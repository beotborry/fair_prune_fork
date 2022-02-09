from __future__ import print_function

import time
import os
from utils import get_accuracy, print_logs_n_init
import trainer

import torch
import torch.nn as nn
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
        self.no_film_residual = args.no_film_residual
        self.gamma = args.gamma
        self.mhe = args.mhe

    def train(self, model, train_loader, test_loader, epochs):
        log_set = defaultdict(list)

        optimizer = self.optimizer
        scheduler = self.scheduler
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
            train_total_loss, train_mmd_loss, train_cos_loss, train_acc = self._train_epoch(epoch, train_loader, model, optimizer, MMDLoss_, group_probs)

            eval_start_time = time.time()

            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion,
                                                            group_probs=group_probs_test,
                                                            esti_method=self.esti_method)

            log_set['train_acc'].append(train_acc)
            log_set['train_total_loss'].append(train_total_loss)
            log_set['train_mmd_loss'].append(train_mmd_loss)
            log_set['train_cos_loss'].append(train_cos_loss)
            log_set['eval_acc'].append(eval_acc)
            log_set['eval_loss'].append(eval_loss.item())
            log_set['eval_deopp'].append(eval_deopp)

            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.3f} Test DEopp {:.3f} [{:.3f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if scheduler is not None:
                if 'Reduce' in type(scheduler).__name__:
                    scheduler.step(eval_loss)
                else:
                    scheduler.step()

        print('Training Finished!')
        torch.save(log_set, os.path.join(self.log_dir, self.log_name + '_log.pt'))
        return model

    def _train_epoch(self, epoch, train_loader, model, optimizer, MMDLoss, group_probs=None):
        model.train()
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        running_acc = 0.0
        running_total_loss = 0.0
        running_mmd_loss = 0.0
        running_cos_loss = 0.0

        epoch_total_loss = 0.0
        epoch_mmd_loss = 0.0
        epoch_cos_loss = 0.0
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
            total_loss, outputs, loss, mmd_loss, cos = self.compute_loss(model, inputs, groups, labels, MMDLoss, group_probs, idxs[0],
                                                                    num_groups, num_classes)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_mmd_loss += mmd_loss
            running_cos_loss += cos

            acc = get_accuracy(outputs, labels)
            running_acc += acc
            try:
                epoch_mmd_loss += mmd_loss.item() * len(labels)
                epoch_cos_loss += cos.item() * len(labels)
            except:
                epoch_mmd_loss += mmd_loss * len(labels)
                epoch_cos_loss += cos * len(labels)

            epoch_total_loss += total_loss.item() * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            if i % self.term == 0:  # print every self.term mini-batches
                running_total_loss, running_acc, batch_start_time, running_mmd_loss, running_cos_loss = print_logs_n_init(
                    epoch, self.epochs, i, batch_start_time, running_total_loss,
                    running_acc, self.term, mmd_loss=running_mmd_loss, cos_loss=running_cos_loss)

        return epoch_total_loss / num_data, epoch_mmd_loss / num_data, epoch_cos_loss / num_data, epoch_acc / num_data

    def compute_loss(self, model, inputs, groups, labels, MMDLoss, group_probs=None, idxs=None,
                     num_groups=2, num_classes=2):
        if group_probs is None:
            outputs = model(inputs, group=groups, get_inter=True, no_film_residual=self.no_film_residual)
        else:
            assert idxs is not None
            idxs = idxs.cuda() if self.cuda else idxs
            true_groups = groups if self.with_trueid else None
            group_ids = group_probs[idxs] if self.esti_method == 'prob' else group_probs[idxs].argmax(dim=1)
            outputs = model(inputs, group=group_ids, get_inter=True, true_group=true_groups,
                            no_film_residual=self.no_film_residual)
        predictions = outputs[-1]
        loss = self.criterion(model, predictions, labels, esti_method=self.esti_method)

        features = outputs[-2]
        mmd_loss = MMDLoss.forward(features, groups, labels) if self.lamb != 0 else 0
        total_loss = loss + mmd_loss

        inter_loss = 0
        if self.gamma != 0:
            fixed = torch.ones((num_classes, 1), requires_grad=False).cuda()
            for g in range(num_groups):
                if self.model_type == 'mlp':
                    angleW = model.head[g].weight
                else:
                    angleW = model.fc[g].weight
                # norm = angleW.norm(2, dim=1, keepdim=True).expand(angleW.data.size())
                # angleW = angleW / norm # n_class x d
                if self.mhe:
                    angleWt = angleW.t()
                    inner = torch.matmul(angleW, angleWt)
                    inner.fill_diagonal_(0.)
                    inner = (1/(torch.acos(inner))).sum(dim=1)
                    inner = torch.index_select(inner, dim=0, index=labels)
                    inter_loss += self.gamma * (torch.mean(inner) / (num_classes-1))

                else:
                    angleWt = angleW.t()  # d x n_class
                    W = angleWt.mm(fixed)  # fixed: n_class x 1    W: d x 1
                    Wt = W.t()  # 1 x d
                    cos_g = Wt.mm(W)  # 1 x 1
                    cos = cos_g.sum() * (self.gamma / (num_classes ** 2))
                    inter_loss += cos

            total_loss += inter_loss

        return total_loss, predictions, loss, mmd_loss, inter_loss

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



