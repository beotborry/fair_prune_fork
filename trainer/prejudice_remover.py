from __future__ import print_function

import time
import os
from utils import get_accuracy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.nn.functional as F
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.lamb
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
        optimizer = self.optimizer
        scheduler = self.scheduler

        num_classes = train_loader.dataset.num_classes

        model.train()

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(epoch, train_loader, model, optimizer, self.criterion)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion, decouple=self.decouple)
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

    def _train_epoch(self, epoch, train_loader, model, optimizer, criterion):
        model.train()
        # print(next(iter(train_loader)))
        running_acc = 0.0
        running_loss = 0.0

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_data = 0

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        batch_start_time = time.time()
        for i, data in enumerate(train_loader, 1):
            # Get the inputs
            inputs, groups, labels, _ = data
            labels = labels.long()
            if self.cuda:
                inputs = inputs.cuda()
                groups = groups.cuda()
                labels = labels.cuda()
            outputs = model(inputs, group=groups) if self.decouple else model(inputs)
            loss = criterion(model, outputs, labels)

            p = F.softmax(outputs, dim=1)

            # compute p(y|s), (num_groups, num_classes)
            q = []
            for g in range(num_groups):
                if torch.sum(groups == g) == 0:
                    q.append(0.)
                else:
                    q.append(p[groups == g].sum(dim=0) / torch.sum(groups == g))
            q = torch.stack(q, dim=0)
            q = q.cuda() if self.cuda else q

            # compute p(y), (num_classes,)
            r = p.sum(dim=0) / len(p)
            reg_pr = torch.sum(p * torch.log(torch.index_select(q, dim=0, index=groups.long()) / r.unsqueeze(0)))
            loss += self.lamb * reg_pr


            running_loss += loss.item()
            acc = get_accuracy(outputs, labels)
            running_acc += acc

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

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
