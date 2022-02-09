from __future__ import print_function

import time
import os
from utils import get_accuracy, print_logs_n_init
import trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from trainer.mmd_utils import MMDLoss
from collections import defaultdict


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.lamb
        self.sigma = args.sigma
        self.model_type = args.model
        self.kernel = 'rbf'
        self.group_estimator = args.group_estimator
        self.esti_method = args.esti_method
        self.with_trueid = args.with_trueid
        self.mask_step = args.mask_step
        self.no_groupmask = args.no_groupmask
        # self.lr_decay_step = 30

        param_theta = [param for name, param in self.model.named_parameters() if 'mask' not in name]
        self.theta_optimizer = optim.Adam(param_theta, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler_theta = ReduceLROnPlateau(self.theta_optimizer, patience=5)

        param_m = [param for name, param in self.model.named_parameters() if 'mask' in name] \
            if not args.no_groupmask else None
        self.mask_optimizer = optim.Adam(param_m, lr=args.mask_lr, weight_decay=args.weight_decay) \
            if not args.no_groupmask else None
        # self.mask_optimizer = optim.SGD(param_m, lr=args.mask_lr)
        self.scheduler_mask = ReduceLROnPlateau(self.mask_optimizer, patience=5) \
            if not args.no_groupmask else None
        # self.scheduler_mask = StepLR(self.mask_optimizer, step_size=self.lr_decay_step, gamma=0.01)

    def train(self, model, train_loader, test_loader, epochs):
        log_set = defaultdict(list)

        optimizers = [self.theta_optimizer, self.mask_optimizer]
        schedulers = [self.scheduler_theta, self.scheduler_mask]

        model = self.model
        model.train()

        # Make mmd loss
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes

        MMDLoss_ = MMDLoss(self.lamb, 1, num_groups, num_classes)

        group_probs, group_probs_test = None, None
        if self.group_estimator != 'true':
            group_probs, group_probs_test = self._load_pretrained_group_prediction()
            if self.cuda:
                group_probs, group_probs_test = group_probs.cuda(), group_probs_test.cuda()

        for epoch in range(epochs):
            train_acc, train_total_loss, train_mmd_loss = self._train_epoch(epoch, train_loader, model, optimizers, MMDLoss_, group_probs)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion,
                                                            group_probs=group_probs_test,
                                                            esti_method=self.esti_method)
            log_set['train_acc'].append(train_acc)
            log_set['train_total_loss'].append(train_total_loss)
            log_set['train_mmd_loss'].append(train_mmd_loss)
            log_set['eval_acc'].append(eval_acc)
            log_set['eval_loss'].append(eval_loss.item())
            log_set['eval_deopp'].append(eval_deopp)

            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.3f} Test DEopp {:.3f} [{:.3f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            for s in schedulers:
                if s is not None:
                    if 'Reduce' in type(s).__name__:
                        s.step(eval_loss)
                    else:
                        s.step()

        print('Training Finished!')
        torch.save(log_set, os.path.join(self.log_dir, self.log_name + '_log.pt'))
        return model

    def _train_epoch(self, epoch, train_loader, model, optimizers, MMDLoss, group_probs=None):
        model.train()
        num_classes = train_loader.dataset.num_classes

        optimizer_theta = optimizers[0]
        optimizer_mask = optimizers[1]
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
            inputs, groups, targets, idxs = data
            labels = targets
            labels = labels.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()
            # theta and mask update simutaneously
            total_loss, outputs, loss, mmd_loss = self.compute_loss(model, inputs, groups, labels,
                                                                    MMDLoss, group_probs, idxs[0])

            optimizer_theta.zero_grad()
            if not self.no_groupmask:
                optimizer_mask.zero_grad()

            total_loss.backward()

            optimizer_theta.step()
            if not self.no_groupmask:
                if i % self.mask_step == 0:
                    optimizer_mask.step()

            running_total_loss += loss.item()
            running_mmd_loss += mmd_loss

            # binary = True if num_classes == 2 else False
            # acc = get_accuracy(outputs, labels, binary=binary, sigmoid_output=True)
            acc = get_accuracy(outputs, labels)
            running_acc += acc
            try:
                epoch_mmd_loss += mmd_loss.item() * len(labels)
            except:
                epoch_mmd_loss += mmd_loss * len(labels)
            epoch_total_loss += total_loss.item() * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            if i % self.term == 0:  # print every self.term mini-batches
                running_total_loss, running_acc, batch_start_time, running_mmd_loss = print_logs_n_init(
                    epoch, self.epochs, i, batch_start_time, running_total_loss,
                    running_acc, self.term, mmd_loss=mmd_loss)

        return epoch_acc / num_data, epoch_total_loss / num_data, epoch_mmd_loss / num_data

    def compute_loss(self, model, inputs, groups, labels, MMDLoss, group_probs=None, idxs=None):
        if group_probs is None:
            outputs = model(inputs, group=groups, get_inter=True)
        else:
            assert idxs is not None
            idxs = idxs.cuda() if self.cuda else idxs
            true_groups = groups if self.with_trueid else None
            group_ids = group_probs[idxs] if self.esti_method == 'prob' else group_probs[idxs].argmax(dim=1)
            outputs = model(inputs, group=group_ids, get_inter=True, true_group=true_groups)
        predictions = outputs[-1]
        loss = self.criterion(model, predictions, labels, esti_method=self.esti_method)

        features = outputs[-2]
        mmd_loss = MMDLoss.forward(features, groups, labels) if self.lamb != 0 else 0
        total_loss = loss + mmd_loss
        return total_loss, predictions, loss, mmd_loss

    def criterion(self, model, outputs, labels, esti_method=None):
        if esti_method == 'prob':
            return nn.NLLLoss()(torch.log(outputs + 1e-20), labels)
        else:
            return nn.CrossEntropyLoss()(outputs, labels)

    def _load_pretrained_group_prediction(self):
        try:
            path = os.path.join('./results', 'group_clf', self.args.dataset)
            train_probs_file = os.path.join(path, 'group_probs_seed{}_train.pt'.format(self.args.seed))
            test_probs_file = os.path.join(path, 'group_probs_seed{}_test.pt'.format(self.args.seed))
            group_probs = torch.load(train_probs_file)
            group_probs_test = torch.load(test_probs_file)
        except:
            raise FileNotFoundError('Estimation file does not exist')

        return group_probs, group_probs_test



