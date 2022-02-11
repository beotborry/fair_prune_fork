from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

from utils import get_accuracy
from networks.mlp.mlp import MLP
import trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.adv_lambda = args.adv_lambda
        self.adv_lr = args.eta
        self.decouple = args.decouple
        self.feature_adv = args.feature_adv
        self.target_criterion = args.target_criterion

        self.film = args.film
        self.no_film_residual = args.no_film_residual

        self.no_groupmask = args.no_groupmask
        self.mask_step = args.mask_step
        param_m = [param for name, param in self.model.named_parameters() if 'mask' in name] \
            if not args.no_groupmask and self.decouple else None
        self.mask_optimizer = optim.Adam(param_m, lr=args.mask_lr, weight_decay=args.weight_decay) \
            if not args.no_groupmask and self.decouple else None
        self.scheduler_mask = ReduceLROnPlateau(self.mask_optimizer, patience=5) \
            if not args.no_groupmask and self.decouple else None

    def train(self, model, train_loader, test_loader, epochs):
        #model = self.model
        #model.train()
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes
        self._init_adversary(num_groups, num_classes, train_loader)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_deopp = \
                self.evaluate(model, self.sa_clf, test_loader, self.criterion, self.adv_criterion, decouple=self.decouple)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test Adv loss: {:.3f} Test Adv Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)
                self.adv_scheduler.step(eval_adv_loss)

        print('Training Finished!')
        return model

    def _train_epoch(self, epoch, train_loader, model):
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        model.train()

        running_acc = 0.0
        running_loss = 0.0
        running_adv_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            # groups = groups.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()

            labels = labels.long()
            groups = groups.long()

            get_inter = True if self.feature_adv else False
            if self.decouple:
                if self.film and self.no_film_residual:
                    outputs = model(inputs, groups, get_inter=get_inter, no_film_residual=self.no_film_residual)
                else:
                    outputs = model(inputs, groups, get_inter=get_inter)
            else:
                outputs = model(inputs, get_inter=get_inter)


            inputs_for_adv = outputs[-2] if get_inter else outputs
            # print(inputs_for_adv)
            logits = outputs[-1] if get_inter else outputs

            adv_inputs = None
            if self.target_criterion =='eo':
                repeat_times = num_classes if not self.feature_adv else outputs[-2].shape[-1]
                input_loc = F.one_hot(labels.long(), num_classes).repeat_interleave(repeat_times, dim=1)
                adv_inputs = inputs_for_adv.repeat(1, repeat_times) * input_loc
                adv_inputs = torch.cat((inputs_for_adv, adv_inputs), dim=1)

            elif self.target_criterion == 'dp':
                adv_inputs = inputs_for_adv

            elif self.target_criterion == 'eopp':
                y1_idx = torch.where(labels.long() == 1)[0]
                adv_inputs = inputs_for_adv[y1_idx]

            adv_preds = self.sa_clf(adv_inputs)
            if self.target_criterion != "eopp":
                adv_loss = self.adv_criterion(self.sa_clf, adv_preds, groups)
            else:
                if len(adv_preds.shape) != 2:
                    adv_preds = torch.unsqueeze(adv_preds, 0)
                adv_loss = self.adv_criterion(self.sa_clf, adv_preds, groups[y1_idx])

            self.optimizer.zero_grad()
            self.adv_optimizer.zero_grad()
            if not self.no_groupmask and self.decouple:
                self.mask_optimizer.zero_grad()

            adv_loss.backward(retain_graph=True)
            for n, p in model.named_parameters():
                unit_adv_grad = p.grad / (p.grad.norm() + torch.finfo(torch.float32).tiny)
                p.grad += torch.sum(p.grad * unit_adv_grad) * unit_adv_grad # gradients are already reversed

            loss = self.criterion(model, logits, labels)

#             loss = loss + adv_loss
            loss.backward()

            self.optimizer.step()
#             adv_loss.backward()
            self.adv_optimizer.step()
            if not self.no_groupmask and self.decouple:
                if i % self.mask_step == 0:
                    self.mask_optimizer.step()

            running_loss += loss.item()
            running_adv_loss += adv_loss.item()
            # binary = True if num_classes ==2 else False
            running_acc += get_accuracy(outputs, labels)

#             self.optimizer.step()
#             self.adv_optimizer.step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print_statement = '[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Adv Loss: {:.3f} Train Acc: {:.2f} [{:.2f} s/batch]'\
                    .format(epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term,
                            running_adv_loss / self.term,running_acc / self.term, avg_batch_time / self.term)
                print(print_statement)

                running_loss = 0.0
                running_acc = 0.0
                running_adv_loss = 0.0
                batch_start_time = time.time()


    def evaluate(self, model, adversary, loader, criterion, adv_criterion, decouple=False):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        eval_acc = 0
        eval_adv_acc = 0
        eval_loss = 0
        eval_adv_loss = 0
        eval_eopp_list = torch.zeros(num_groups, num_classes).cuda()
        eval_data_count = torch.zeros(num_groups, num_classes).cuda()

        if 'Custom' in type(loader).__name__:
            loader = loader.generate()
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, classes, _ = eval_data
                #
                labels = classes
                groups = groups.long()
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    groups = groups.cuda()

                labels = labels.long()

                get_inter = True if self.feature_adv else False
                if self.decouple:
                    if self.film and self.no_film_residual:
                        outputs = model(inputs, groups, get_inter=get_inter, no_film_residual=self.no_film_residual)
                    else:
                        outputs = model(inputs, groups, get_inter=get_inter)
                else:
                    outputs = model(inputs, get_inter=get_inter)

                inputs_for_adv = outputs[-2] if get_inter else outputs
                logits = outputs[-1] if get_inter else outputs

                adv_inputs = None
                if self.target_criterion == 'eo':
                    repeat_times = num_classes if not self.feature_adv else outputs[-2].shape[-1]
                    input_loc = F.one_hot(labels.long(), num_classes).repeat_interleave(repeat_times, dim=1)
                    adv_inputs = inputs_for_adv.repeat(1, repeat_times) * input_loc
                    adv_inputs = torch.cat((inputs_for_adv, adv_inputs), dim=1)

                elif self.target_criterion == 'dp':
                    adv_inputs = inputs_for_adv
                
                elif self.target_criterion == 'eopp':
                    y1_idx = torch.where(labels.long() == 1)[0]
                    adv_inputs = inputs_for_adv[y1_idx]


                loss = criterion(model, logits, labels)
                eval_loss += loss.item() * len(labels)
                # binary = True if num_classes == 2 else False
                # acc = get_accuracy(outputs, labels, binary=binary, reduction='none')
                acc = get_accuracy(outputs, labels, reduction='none')
                eval_acc += acc.sum()

                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
                        eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

                adv_preds = adversary(adv_inputs)
                # groups = groups.float() if num_groups == 2 else groups.long()
                groups = groups.long()
                if self.target_criterion != "eopp":
                    adv_loss = adv_criterion(model, adv_preds, groups)
                    eval_adv_loss += adv_loss.item() * len(labels)
                    eval_adv_acc += get_accuracy(adv_preds, groups)
                else:
                    if len(adv_preds.shape) != 2:
                        adv_preds = torch.unsqueeze(adv_preds, 0)
                    adv_loss = adv_criterion(model, adv_preds, groups[y1_idx])
                    eval_adv_loss += adv_loss.item() * len(labels[y1_idx])
                    eval_adv_acc += get_accuracy(adv_preds, groups[y1_idx])
                # binary = True if num_groups == 2 else False

            eval_loss = eval_loss / eval_data_count.sum()
            eval_acc = eval_acc / eval_data_count.sum()
            if self.target_criterion != "eopp":
                eval_adv_loss = eval_adv_loss / eval_data_count.sum()
                eval_adv_acc = eval_adv_acc / eval_data_count.sum()
            else:
                eval_adv_loss = eval_adv_loss / eval_data_count[:, 1].sum()
                eval_adv_acc = eval_adv_acc / eval_data_count[:, 1].sum()
            eval_eopp_list = eval_eopp_list / eval_data_count
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_max_eopp = torch.max(eval_max_eopp).item()
        model.train()
        return eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_max_eopp

    def _init_adversary(self, num_groups, num_classes, dataloader):
        self.model.eval()
        if self.feature_adv:
            with torch.no_grad():
                for data in dataloader:
                    dummy_inputs, groups, _, _ = data
                    outputs = self.model(dummy_inputs.cuda(), groups.long().cuda(), get_feature=True)
                    if self.target_criterion == 'eo':
                        feature_size = outputs[-2].shape[-1] * (outputs[-2].shape[-1] + 1)
                    elif self.target_criterion == 'dp':
                        feature_size = outputs[-2].shape[-1]
                        break
        else:
            if self.target_criterion == 'eo':
                feature_size = num_classes * (num_classes + 1)
            elif self.target_criterion == 'dp':
                feature_size = num_classes
            elif self.target_criterion == 'eopp':
                feature_size = num_classes


#         if self.adv_perhead:
#             self.sa_clf = []
#             self.adv_optimizer = []
#             self.adv_scheduler = []
#             for _ in range(num_groups):
#                 sa_clf = MLP(feature_size=num_classes, hidden_dim=32, num_classes=num_groups, num_layer=2,
#                              adv=True, adv_lambda=self.adv_lambda)
#                 if self.cuda:
#                     sa_clf.cuda()
#                 sa_clf.train()
#                 self.sa_clf.append(sa_clf)
#                 adv_optimizer = optim.Adam(sa_clf.parameters(), lr=self.adv_lr)
#                 self.adv_optimizer.append(adv_optimizer)
#                 self.adv_scheduler.append(ReduceLROnPlateau(adv_optimizer, patience=5))

#         else:
        sa_clf = MLP(feature_size=feature_size, hidden_dim=32, num_classes=num_groups,
                     num_layer=2, adv=True, adv_lambda=self.adv_lambda)
        if self.cuda:
            sa_clf.cuda()
        sa_clf.train()
        self.sa_clf = sa_clf
        self.adv_optimizer = optim.Adam(sa_clf.parameters(), lr=self.adv_lr)
        self.adv_scheduler = ReduceLROnPlateau(self.adv_optimizer, patience=5)

        self.adv_criterion = self.criterion

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
