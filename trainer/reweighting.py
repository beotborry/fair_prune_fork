from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from utils import get_accuracy
from collections import defaultdict
import trainer

from torch.utils.data import DataLoader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.eta = args.eta
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.reweighting_target_criterion = args.reweighting_target_criterion #
        self.decouple = args.decouple

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
        log_set = defaultdict(list)
        model = self.model if model is None else model
        model.train()
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes
        
        extended_multipliers = torch.zeros((num_groups, num_classes))                  #

        # Full batch statistics
        _, Y_train, S_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)  #
        
        eta_learning_rate = self.eta
        print('eta_learning_rate : ', eta_learning_rate)
        n_iters = self.iteration
        print('n_iters : ', n_iters)
        
        violations = 0
        
        for iter_ in range(n_iters):
            start_t = time.time()
            # update weight (normalization from w_tilde)
            weight_set = self.debias_weights(Y_train, S_train, extended_multipliers, num_groups, num_classes)   #
            
            for epoch in range(epochs):
                lb_idx = self._train_epoch(epoch, train_loader, model, weight_set)

                eval_start_time = time.time()
                eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion, decouple=self.decouple)   # factory
                eval_end_time = time.time()
                print('[{}/{}] Method: {} '
                      'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                      (epoch + 1, epochs, self.method,
                       eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

                if self.scheduler != None:
                    self.scheduler.step(eval_loss)

            end_t = time.time()
            train_t = int((end_t - start_t) / 60)
            print('Training Time : {} hours {} minutes / iter : {}/{}'.format(int(train_t / 60), (train_t % 60),
                                                                              (iter_ + 1), n_iters))

            # 모델결과 통계
            Y_pred_train, Y_train, S_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                                 num_workers=self.num_workers, model=model)  #
            
            print('Y_pred_train',Y_pred_train)
            print('Y_train',Y_train)
            
            # violation 계산 (for each class y)
            if self.reweighting_target_criterion == 'dp':
                acc, violations = self.get_error_and_violations_DP(Y_pred_train, Y_train, S_train, num_groups, num_classes)
            elif self.reweighting_target_criterion == 'eo':
                acc, violations = self.get_error_and_violations_EO(Y_pred_train, Y_train, S_train, num_groups, num_classes)
            elif self.reweighting_target_criterion == 'eopp':
                acc, violations = self.get_error_and_violations_EOPP(Y_pred_train, Y_train, S_train, num_groups, num_classes)
            #print(eta_learning_rate)
            extended_multipliers -= eta_learning_rate * violations
            print("extended_multipliers:", extended_multipliers,"weight:",  weight_set)

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, weight_set):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        avg_batch_time = 0.0

        num_batches = len(train_loader)
        num_classes = train_loader.dataset.num_classes

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, indexes = data

            labels = targets
            # labels = labels.float() if num_classes == 2 else labels.long()
            labels = labels.long()

            weights = weight_set[indexes[0]]
            #print("train weights:", weights)

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
                groups = groups.cuda()

            if self.decouple:
                if self.film and self.no_film_residual:
                    outputs = model(inputs, groups, no_film_residual=self.no_film_residual)
                else:
                    outputs = model(inputs, groups)
            else:
                outputs = model(inputs)


            # if num_classes == 2:
            #     loss = torch.mean(weights * nn.BCEWithLogitsLoss(reduction='none')(outputs, labels))
            # else:
            loss = torch.mean(weights * nn.CrossEntropyLoss(reduction='none')(outputs, labels))

            running_loss += loss.item()
            # binary = True if num_classes == 2 else False
            # running_acc += get_accuracy(outputs, labels, binary=binary)
            running_acc += get_accuracy(outputs, labels)

            # zero the parameter gradients + backward + optimize
            self.optimizer.zero_grad()
            if not self.no_groupmask and self.decouple:
                self.mask_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not self.no_groupmask and self.decouple:
                if i % self.mask_step == 0:
                    self.mask_optimizer.step()

            batch_end_time = time.time()
            avg_batch_time += batch_end_time - batch_start_time

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                avg_batch_time = 0.0

        last_batch_idx = i
        return last_batch_idx

    def get_statistics(self, dataset, batch_size=128, num_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        num_classes = dataloader.dataset.num_classes

        if model != None:
            model.eval()

        Y_pred_set = []
        Y_set = []
        S_set = []

        for i, data in enumerate(dataloader):
            inputs, _, sen_attrs, targets, indexes = data
            Y_set.append(targets)
            S_set.append(sen_attrs)

            if self.cuda:
                inputs = inputs.cuda()
                groups = sen_attrs.cuda()
            if model != None:
                outputs = model(inputs) if not self.decouple else model(inputs, groups)
                # Y_pred_set.append(torch.argmax(outputs, dim=1) if num_classes >2 else (torch.sigmoid(outputs) >= 0.5).float())
                Y_pred_set.append(torch.argmax(outputs, dim=1))

        Y_set = torch.cat(Y_set)
        S_set = torch.cat(S_set)
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        return Y_pred_set.long(), Y_set.long().cuda(), S_set.long().cuda()

    # Vectorized version for DP & multi-class
    def get_error_and_violations_DP(self, y_pred, label, sen_attrs, num_groups, num_classes):
        acc = torch.mean((y_pred == label).float())
        total_num = len(y_pred)
        violations = torch.zeros((num_groups, num_classes))

        for g in range(num_groups):
            for c in range(num_classes):
                pivot = len(torch.where(y_pred==c)[0])/total_num
                group_idxs=torch.where(sen_attrs == g)[0]
                group_pred_idxs = torch.where(torch.logical_and(sen_attrs == g, y_pred == c))[0]
                violations[g, c] = len(group_pred_idxs)/len(group_idxs) - pivot
        return acc, violations

    # Vectorized version for EO & multi-class
    def get_error_and_violations_EO(self, y_pred, label, sen_attrs, num_groups, num_classes):
        acc = torch.mean((y_pred == label).float())
        total_num = len(y_pred)
        violations = torch.zeros((num_groups, num_classes)) 
        for g in range(num_groups):
            for c in range(num_classes):
                class_idxs = torch.where(label==c)[0]
                pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
                pivot = len(pred_class_idxs)/len(class_idxs)
                group_class_idxs=torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
                group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
                violations[g, c] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
        print('violations',violations)
        return acc, violations

    def get_error_and_violations_EOPP(self, y_pred, label, sen_attrs, num_groups=2, num_classes=2):
        acc = torch.mean((y_pred == label).float())
        total_num = len(y_pred)
        violations = torch.zeros((num_groups, num_classes))
        for g in range(num_groups):
            c = 1
            class_idxs = torch.where(label==c)[0]
            pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
            pivot = len(pred_class_idxs) / len(class_idxs)
            group_class_idxs = torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
            group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
            violations[g, 0] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
            violations[g, 1] = violations[g, 0]
        print('violations', violations)
        return acc, violations


    # update weight
    def debias_weights(self, label, sen_attrs, extended_multipliers, num_groups, num_classes):  #
        weights = torch.zeros(len(label))
        w_matrix = torch.sigmoid(extended_multipliers) # g by c
        weights = w_matrix[sen_attrs, label]
       # for i in range(num_groups):
       #      group_idxs = torch.where(sen_attrs == i)[0]
       #      w_tilde = torch.exp(extended_multipliers[i])
       #      weights[group_idxs] += w_tilde[label[group_idxs]]
       #      weights[group_idxs] /= torch.sum(torch.exp(extended_multipliers), axis=0)[label[group_idxs]] #
                
        return weights


    def criterion(self, model, outputs, labels):
        # if num_classes == 2:
        #     return nn.BCEWithLogitsLoss()(outputs, labels)
        # else:
        return nn.CrossEntropyLoss()(outputs, labels)
