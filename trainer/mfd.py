from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import time
import os
from utils import get_accuracy
from scipy.io import savemat
from utils import print_logs_n_init
from collections import defaultdict
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.regressor = None
        self.lamb = args.lamb
        self.sigma = args.sigma
        self.kernel = 'rbf'

        self.decouple = args.decouple

        self.film = args.film
        self.no_film_residual = args.no_film_residual

        self.mask_step = args.mask_step
        self.no_groupmask = args.no_groupmask
        param_m = [param for name, param in self.model.named_parameters() if 'mask' in name] \
            if not args.no_groupmask and self.decouple else None
        self.mask_optimizer = optim.Adam(param_m, lr=args.mask_lr, weight_decay=args.weight_decay) \
            if not args.no_groupmask and self.decouple else None
        self.scheduler_mask = ReduceLROnPlateau(self.mask_optimizer, patience=5) \
            if not args.no_groupmask and self.decouple else None

    def train(self, model, train_loader, test_loader, epochs):
        log_set = defaultdict(list)
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        optimizer = self.optimizer

        model = self.model
        teacher = self.teacher
        model.train()
        
        if 'adam' in optimizer.__module__:
            self.scheduler = ReduceLROnPlateau(optimizer, patience=5)
        else:
            self.scheduler = MultiStepLR(optimizer, [epochs//3, epochs//3 * 2], gamma=0.1)
        
        distiller = MMDLoss(self.lamb, self.sigma, num_classes, num_groups)

        for epoch in range(self.epochs):
            train_acc, train_total_loss, train_mmd_loss = self._train_epoch(epoch, train_loader, model, teacher, distiller)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(self.model, test_loader, self.criterion, decouple=self.decouple)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller):
        model.train()
        teacher.eval()

        num_classes = train_loader.dataset.num_classes
        running_acc = 0.0
        running_total_loss = 0.0
        running_mmd_loss = 0.0

        epoch_total_loss = 0.0
        epoch_mmd_loss = 0.0
        epoch_acc = 0.0
        num_data = 0

        batch_start_time = time.time()        
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, groups, targets, _ = data
            labels = targets.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.long().cuda()
#             t_inputs = inputs.to(self.t_device)

            if self.decouple:
                if self.film and self.no_film_residual:
                    outputs = model(inputs, groups, get_inter=True, no_film_residual=self.no_film_residual)
                else:
                    outputs = model(inputs, groups, get_inter=True)
            else:
                outputs = model(inputs, get_inter=True)

            stu_logits = outputs[-1]

#             t_outputs = teacher(t_inputs, get_inter=True)
            with torch.no_grad():
                t_outputs = teacher(inputs, get_inter=True)

            loss = self.criterion(model, stu_logits, labels)

            f_s = outputs[-2]
            f_t = t_outputs[-2]
            mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels)

            loss = loss + mmd_loss
            
            running_total_loss += loss.item()
            running_mmd_loss += mmd_loss

            acc = get_accuracy(stu_logits, labels)
            running_acc += acc
            try:
                epoch_mmd_loss += mmd_loss.item() * len(labels)
            except:
                epoch_mmd_loss += mmd_loss * len(labels)

            epoch_total_loss += loss.item() * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            self.optimizer.zero_grad()
            if not self.no_groupmask and self.decouple:
                self.mask_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if not self.no_groupmask and self.decouple:
                if i % self.mask_step == 0:
                    self.mask_optimizer.step()

            if i % self.term == 0:  # print every self.term mini-batches
                running_total_loss, running_acc, batch_start_time, running_mmd_loss = print_logs_n_init(
                    epoch, self.epochs, i, batch_start_time, running_total_loss,
                    running_acc, self.term, mmd_loss=mmd_loss)

        return epoch_acc / num_data, epoch_total_loss / num_data, epoch_mmd_loss / num_data
            
    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)


class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel='rbf'):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        with torch.no_grad():
            _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)
        for c in range(self.num_classes):
            if len(teacher[labels==c]) == 0:
                continue
            for g in range(self.num_groups):
                if len(student[(labels==c) * (groups == g)]) == 0:
                    continue
                K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                             sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                     sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                     sigma_avg=sigma_avg, kernel=self.kernel)

                mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1/2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg
