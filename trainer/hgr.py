from __future__ import print_function

import time
import os
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from utils import get_accuracy
from scipy.io import savemat
import trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.hgr_utils import chi_squared_kde, chi_squared_kde_dp
from collections import defaultdict


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.batch_size = args.batch_size

        ## Below follows the setting mentioned in their paper.
        self.batchRenyi = args.batchRenyi
        self.lambda_renyi = args.lamb * self.batchRenyi / self.batch_size

        self.target_criterion = args.target_criterion
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
        model = self.model
        model.train()

        train_data = train_loader.dataset

        log_set = defaultdict(list)
        optimizer = self.optimizer
        scheduler = self.scheduler

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(epoch, train_loader, model, optimizer, self.criterion, train_data)

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

    def _train_epoch(self, epoch, train_loader, model, optimizer, criterion, train_data):
        model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_data = 0

        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader,1):
            # Get the inputs
            inputs, groups, labels, _ = data

            labels = labels.long()
            if self.cuda:
                inputs = inputs.cuda()
                groups = groups.cuda()
                labels = labels.cuda()

            if self.decouple:
                if self.film and self.no_film_residual:
                    outputs = model(inputs, groups, no_film_residual=self.no_film_residual)
                else:
                    outputs = model(inputs, groups)
            else:
                outputs = model(inputs)
            # Select a renyi regularization mini batch and compute the value of the model on it
            frac = self.batchRenyi / len(train_data)
            foo = torch.bernoulli(frac * torch.ones(len(train_data))).bool()
            ren_idxs = torch.nonzero(foo).squeeze()

            if self.args.dataset == 'utkface':
                br = []
                pr = []
                yr = []
                for ren_idx in ren_idxs:
                    data_renyi = train_data[ren_idx]
                    br.append(data_renyi[0])
                    pr.append(torch.as_tensor(data_renyi[2]))
                    yr.append(torch.as_tensor(data_renyi[3]))
                br = torch.stack(br).float()
                pr = torch.stack(pr).float()
                yr = torch.stack(yr).float()
            else:
                br = torch.from_numpy(train_data.features[foo]).float()
                pr = torch.from_numpy(train_data.groups[foo]).float()
                yr = torch.from_numpy(train_data.labels[foo]).float()

            if self.cuda:
                br = br.cuda()
                pr = pr.cuda()
                yr = yr.cuda()

            if self.decouple:
                if self.film and self.no_film_residual:
                    ren_outs = model(br, pr.long(), no_film_residual=self.no_film_residual)
                else:
                    ren_outs = model(br, pr.long())
            else:
                ren_outs = model(br)

            # Compute the usual loss of the prediction
            loss = criterion(model, outputs, labels)

            # Compte the fairness penalty for positive labels only since we optimize for DEO
            delta = F.softmax(ren_outs, dim=1)
            # delta = delta[torch.arange(delta.size(0)), yr.long()]
            # r2 = chi_squared_kde( delta, pr[yr==1.])
            r2 = 0.
            if self.target_criterion == 'eo':
                delta = delta[torch.arange(delta.size(0)), yr.long()]
                r2 = chi_squared_kde(delta, pr, yr).sum()
            elif self.target_criterion == 'dp':
                delta = delta[torch.arange(delta.size(0)), 1]
                r2 = chi_squared_kde_dp(delta, pr).sum()

            loss += self.lambda_renyi * r2

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
