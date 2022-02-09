from __future__ import print_function

import time
import os

from utils import get_accuracy, print_logs_n_init
import trainer
import torch
import torch.nn as nn
from trainer.mmd_utils import MMDLoss
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR



class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.lamb = args.lamb
        self.sigma = args.sigma
        self.kernel = 'rbf'

    def train(self, model, train_loader, test_loader, epochs, dummy_loader=None):
        log_set = defaultdict(list)
        optimizer = self.optimizer

        model = self.model
        model.train()

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        if 'adam' in optimizer.__module__:
            self.scheduler = ReduceLROnPlateau(optimizer, patience=5)
        else:
            self.scheduler = MultiStepLR(optimizer, [epochs//3, epochs//3 * 2], gamma=0.1)

        MMDLoss_ = MMDLoss(self.lamb, 1, num_groups, num_classes)

        for epoch in range(epochs):
            train_acc, train_total_loss, train_mmd_loss = self._train_epoch(epoch, train_loader, model, distiller=MMDLoss_)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Multi' not in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

            log_set['train_acc'].append(train_acc)
            log_set['train_total_loss'].append(train_total_loss)
            log_set['train_mmd_loss'].append(train_mmd_loss)
            log_set['eval_acc'].append(eval_acc)
            log_set['eval_loss'].append(eval_loss.item())
            log_set['eval_deopp'].append(eval_deopp)


        print('Training Finished!')
        torch.save(log_set, os.path.join(self.log_dir, self.log_name + '_log.pt'))
        return model

    def _train_epoch(self, epoch, train_loader, model, distiller):

        model.train()
        num_classes = train_loader.dataset.num_classes
        running_acc = 0.0
        running_total_loss = 0.0
        running_mmd_loss = 0.0

        epoch_total_loss = 0.0
        epoch_mmd_loss = 0.0
        epoch_acc = 0.0
        num_data = 0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader, 1):
            # Get the inputs
            inputs, groups, targets, _ = data

            labels = targets
            labels = labels.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()

            outputs = model(inputs, get_inter=True)
            predictions = outputs[-1]
            loss = self.criterion(model, predictions, labels)

            f_s = outputs[-2]
            mmd_loss = distiller.forward(f_s, groups, labels)
            loss = loss + mmd_loss

            running_total_loss += loss.item()
            running_mmd_loss += mmd_loss

            # binary = True if num_classes == 2 else False
            # acc = get_accuracy(predictions, labels, binary=binary, sigmoid_output=False)
            acc = get_accuracy(predictions, labels)
            running_acc += acc
            try:
                epoch_mmd_loss += mmd_loss.item() * len(labels)
            except:
                epoch_mmd_loss += mmd_loss * len(labels)

            epoch_total_loss += loss.item() * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == 0:  # print every self.term mini-batches
                running_total_loss, running_acc, batch_start_time, running_mmd_loss = print_logs_n_init(
                    epoch, self.epochs, i, batch_start_time, running_total_loss,
                    running_acc, self.term, mmd_loss=mmd_loss)

        return epoch_acc / num_data, epoch_total_loss / num_data, epoch_mmd_loss / num_data


    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
