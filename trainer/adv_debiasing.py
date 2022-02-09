from __future__ import print_function

import torch
import torch.nn as nn
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

    def train(self, model, train_loader, test_loader, epochs):
        #model = self.model
        #model.train()
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes
        self._init_adversary(num_groups, num_classes)
        sa_clf_list = self.sa_clf_list
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)


        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, sa_clf_list)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_deopp, eval_adv_loss_list = \
                self.evaluate(model, sa_clf_list, test_loader, self.criterion, self.adv_criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test Adv Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_adv_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)
            if len(self.adv_scheduler_list) != 0:
                for c in range(num_classes):
                    self.adv_scheduler_list[c].step(eval_adv_loss_list[c])

        print('Training Finished!')
        return model

    def _train_epoch(self, epoch, train_loader, model, sa_clf):
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        model.train()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            groups = groups.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()

            outputs = model(inputs)

            adv_loss = 0
            for c in range(num_classes):
                if sum(labels == c) == 0:
                    continue
                adv_inputs = outputs[labels == c].clone()
                adv_preds = sa_clf[c](adv_inputs)
                adv_loss += self.adv_criterion(model, adv_preds, groups[labels==c], num_groups)

            loss = self.criterion(model, outputs, labels, num_classes)

            self.optimizer.zero_grad()
            for c in range(num_classes):
                self.adv_optimizer_list[c].zero_grad()

            loss = loss + adv_loss
            loss.backward()

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.step()
            for c in range(num_classes):
                self.adv_optimizer_list[c].step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print_statement = '[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} [{:.2f} s/batch]'\
                    .format(epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                            avg_batch_time / self.term)
                print(print_statement)

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

    def evaluate(self, model, adversary, loader, criterion, adv_criterion):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        eval_acc = 0
        eval_adv_acc = 0
        eval_loss = 0
        eval_adv_loss = 0
        eval_adv_loss_list = torch.zeros(num_classes)
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

                outputs = model(inputs)

                loss = criterion(model, outputs, labels, num_classes)
                eval_loss += loss.item()
                binary = True if num_classes == 2 else False
                acc = get_accuracy(outputs, labels, binary=binary, reduction='none')
                eval_acc += acc

                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
                        eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

                for c in range(num_classes):
                    if sum(labels==c)==0:
                        continue
                    adv_preds = adversary[c](outputs[labels==c])
                    adv_loss = adv_criterion(model, adv_preds, groups[labels==c], num_classes=num_groups)
                    eval_adv_loss += adv_loss.item()
                    eval_adv_loss_list[c] += adv_loss.item()
                    # print(c, adv_preds.shape)
                    binary = True if num_groups == 2 else False
                    eval_adv_acc += get_accuracy(adv_preds, groups[labels==c], binary=binary)

            eval_loss = eval_loss / eval_data_count.sum()
            eval_acc = eval_acc / eval_data_count.sum()
            eval_adv_loss = eval_adv_loss / eval_data_count.sum()
            eval_adv_loss_list = eval_adv_loss_list / eval_data_count.sum()
            eval_adv_acc = eval_adv_acc / eval_data_count.sum()
            eval_eopp_list = eval_eopp_list / eval_data_count
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_max_eopp = torch.max(eval_max_eopp).item()
        model.train()
        return eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_max_eopp, eval_adv_loss_list

    def _init_adversary(self, num_groups, num_classes):
        self.model.eval()
        self.sa_clf_list = []
        self.adv_optimizer_list = []
        self.adv_scheduler_list = []
        for _ in range(num_classes):
            sa_clf = MLP(feature_size=num_classes, hidden_dim=32, num_classes=num_groups, num_layer=2,
                         adv=True, adv_lambda=self.adv_lambda)
            if self.cuda:
                sa_clf.cuda()
            sa_clf.train()
            self.sa_clf_list.append(sa_clf)
            adv_optimizer = optim.Adam(sa_clf.parameters(), lr=self.adv_lr)
            self.adv_optimizer_list.append(adv_optimizer)
            self.adv_scheduler_list.append(ReduceLROnPlateau(adv_optimizer, patience=5))

        self.adv_criterion = self.criterion

    def criterion(self, model, outputs, labels, num_classes=2):
        if num_classes == 2:
            return nn.BCEWithLogitsLoss()(outputs, labels)
        else:
            return nn.CrossEntropyLoss()(outputs, labels)
