from __future__ import print_function

import time
import os
from utils import get_accuracy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import trainer

from data_handler.dataset_factory import DatasetFactory
from data_handler.mixmatch_utils import MixMatchLoader, get_mixmatch_loss, cross_entropy_prob, get_unlabeled_loader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, model, train_loader, test_loader, epochs):
        log_set = defaultdict(list)
        optimizer = self.optimizer
        scheduler = self.scheduler

        # num_classes = train_loader.dataset.num_classes

        model.train()
        unlabeled_loader = None
        weights=None
        if self.apply_mixmatchloss or self.apply_consistencyloss:
            # TODO : data augment for tabular dataset
            # the dataset below is used for unlabeled dataset
            transform_twice = self.apply_consistencyloss
            unlabeled_dataset = DatasetFactory.get_dataset(self.args.dataset, split='train', target=self.args.target,
                                                           group_mode=self.args.group, sen_attr=self.args.sen_attr,
                                                           skew_ratio=self.args.skew_ratio, get_the_others=True,
                                                           transform_twice=transform_twice)

            print('# data of unlabeled ', len(unlabeled_dataset))
            if self.apply_mixmatchloss:
                train_loader = MixMatchLoader(train_loader, unlabeled_dataset, model,
                                              output_transform=nn.Softmax(dim=1), mixmatch=self.mixmatch)
            else:
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.args.batch_size,
                                              shuffle=True, num_workers=self.args.num_workers,
                                              pin_memory=True, drop_last=True)
                # weights = torch.Tensor([[0.0, 0.40, 0.21, 0.39], [0.37, 0.0, 0.16, 0.47],
                #                         [0.48, 0.24, 0.0, 0.28], [0.54, 0.37, 0.09, 0.0]])
                # weights = weights[self.group].cuda()

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(epoch, train_loader, model, optimizer, self.criterion,
                                                      unlabeled_loader=unlabeled_loader, weights=weights)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.3f} Test DEopp {:.3f} [{:.3f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if scheduler is not None and 'Reduce' in type(scheduler).__name__:
                scheduler.step(eval_loss)
            else:
                scheduler.step()

            log_set['train_loss'].append(train_loss)
            log_set['train_acc'].append(train_acc)
            log_set['test_loss'].append(eval_loss.cpu().numpy())
            log_set['test_acc'].append(eval_acc)
            log_set['eval_deopp'].append(eval_deopp)

        print('Training Finished!')
        torch.save(log_set, os.path.join(self.log_dir, self.log_name + '_log.pt'))

        return model

    def _train_epoch(self, epoch, train_loader, model, optimizer, criterion, unlabeled_loader=None, weights=None):
        model.train()
        # print(next(iter(train_loader)))
        running_acc = 0.0
        running_loss = 0.0

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_data = 0

        # num_classes = train_loader.dataset.num_classes
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
        batch_start_time = time.time()
        for i, data in enumerate(train_loader, 1):
            # Get the inputs
            inputs, groups, labels, _ = data
            labels = labels.long() if not self.apply_mixmatchloss else labels

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(model, outputs, labels, apply_mixmatchloss=self.apply_mixmatchloss)
            running_loss += loss.item()
            acc = get_accuracy(outputs, labels, mixmatched=self.apply_mixmatchloss)
            running_acc += acc

            if unlabeled_iter is not None:
                u_inputs, u_groups, _, _ = next(unlabeled_iter)
                if self.lamb != 0:
                    loss += self.compute_consistency_loss(model, u_inputs, u_groups=u_groups, weights=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            if i % self.term == 0: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.3f} '
                      '[{:.3f} s/batch]'.format
                      (epoch + 1, self.epochs, i, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        return epoch_loss / num_data, epoch_acc / num_data

    def criterion(self, model, outputs, labels, apply_mixmatchloss=False):
        if apply_mixmatchloss:
            cross_entropy_loss = cross_entropy_prob if self.mixmatch else nn.CrossEntropyLoss()

            return get_mixmatch_loss(criterion_labeled=cross_entropy_loss,
                                     output_transform=nn.Softmax(dim=1),
                                     K=2,
                                     weight_unlabeled=self.lamb,
                                     criterion_unlabeled=nn.MSELoss()
                                     )(outputs, labels)
        else:
            return nn.CrossEntropyLoss()(outputs, labels)

    def compute_consistency_loss(self, model, unlabeled_data, u_groups=None, weights=None):
        with torch.no_grad():
            u_input1 = unlabeled_data[0].cuda() if self.cuda else unlabeled_data[0]
            logits1 = model(u_input1)
            prob1 = F.softmax(logits1, dim=1)

        u_input2 = unlabeled_data[1].cuda() if self.cuda else unlabeled_data[1]
        logits2 = model(u_input2)
        log_prob2 = F.log_softmax(logits2, dim=1)

        if weights is None:
            consistency_loss = F.kl_div(log_prob2, prob1)
            print(self.lamb * consistency_loss)
            return self.lamb * consistency_loss
        else:
            consistency_loss = torch.index_select(weights, dim=0, index=u_groups.long().cuda()) \
                               * F.kl_div(log_prob2, prob1, reduction='none').sum(dim=1)
            return self.lamb * consistency_loss.mean()