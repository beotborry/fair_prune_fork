from __future__ import print_function

import time
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from utils import get_accuracy
from collections import defaultdict
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import trainer
import numpy as np
import torch.nn.functional as F


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        if 'adam' in self.optimizer.__module__:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)
        else:
            self.scheduler = MultiStepLR(self.optimizer, [30, 60, 90], gamma=0.1) if self.args.method != 'ns' else \
                    MultiStepLR(self.optimizer, [self.epochs//3, self.epochs//3 * 2], gamma=0.1)
        # for making a loader, keep the information of dataloader
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.decouple = args.decouple
        self.no_groupmask = args.no_groupmask
        self.film = args.film
        self.no_film_residual = args.no_film_residual
        self.mask_step = args.mask_step
        param_m = [param for name, param in self.model.named_parameters() if 'mask' in name] \
            if not args.no_groupmask and self.decouple else None
        self.mask_optimizer = optim.Adam(param_m, lr=args.mask_lr, weight_decay=args.weight_decay) \
            if not args.no_groupmask and self.decouple else None
        self.scheduler_mask = ReduceLROnPlateau(self.mask_optimizer, patience=5) \
            if not args.no_groupmask and self.decouple else None

    def train(self, model, train_loader, test_loader, epochs):
        from torch.utils.data import DataLoader
        dummy_loader = DataLoader(train_loader.dataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.num_workers, 
                                          pin_memory=True, drop_last=False)

        
        log_set = defaultdict(list)
        model.train()
        
        for epoch in range(epochs):
            train_acc, train_loss = self._train_epoch(epoch, train_loader, model, dummy_loader, self.optimizer, self.criterion)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion, decouple=self.decouple)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.3f} Test DEopp {:.3f} [{:.3f} s]'.format
                  (epoch + 1, epochs, self.method, 
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

            if self.measure_loss:
                eval_loss, eval_acc, _ = self.evaluate(model, test_loader, self.ce, groupwise=True)
                log_set['test_loss'].append(eval_loss.cpu().numpy())
                log_set['test_acc'].append(eval_acc.cpu().numpy())
                eval_loss, eval_acc, _ = self.evaluate(model, train_loader, self.ce, groupwise=True)
                log_set['train_loss'].append(eval_loss.cpu().numpy())
                log_set['train_acc'].append(eval_acc.cpu().numpy())

        print('Training Finished!')
        if self.measure_loss:
            torch.save(log_set, os.path.join(self.log_dir, self.log_name + '_log.pt'))

        return model


    def _train_epoch(self, epoch, train_loader, model, dummy_loader, optimizer, criterion):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        num_classes = train_loader.dataset.num_classes
        batch_start_time = time.time()
        self.adjust_lambda(model, train_loader, dummy_loader)
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, groups, labels, _ = data
            if self.cuda:
                inputs = inputs.cuda().squeeze()
                labels = labels.cuda().squeeze()
                groups = groups.cuda()

            # labels = labels.float() if num_classes == 2 else labels.long()
            labels = labels.long()

            if self.decouple:
                if self.film and self.no_film_residual:
                    outputs = model(inputs, groups, no_film_residual=self.no_film_residual)
                else:
                    outputs = model(inputs, groups)
            else:
                outputs = model(inputs)

            loss = criterion(model, outputs, labels)
            running_loss += loss.item()
            # binary = True if num_classes ==2 else False
            running_acc += get_accuracy(outputs, labels)

            optimizer.zero_grad()
            if not self.no_groupmask and self.decouple:
                self.mask_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not self.no_groupmask and self.decouple:
                if i % self.mask_step == 0:
                    self.mask_optimizer.step()

            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.3f} '
                      '[{:.3f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
                
        return running_acc / self.term, running_loss / self.term
    
    def adjust_lambda(self, model, train_loader, dummy_loader):
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are decribed in the paper.
        """
        
        model.eval()
        
        logits = []
        labels = []
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        
        sampler = train_loader.sampler
        with torch.no_grad():
            for i, data in enumerate(dummy_loader):
                input, groups, label, tmp = data
                if self.cuda:
                    input = input.cuda()
                    label = label.cuda()
                    groups = groups.cuda()

                if self.decouple:
                    if self.film and self.no_film_residual:
                        output = model(input, groups, no_film_residual=self.no_film_residual)
                    else:
                        output = model(input, groups)
                else:
                    output = model(input)

                logits.append(output)
                labels.append(label)

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        labels = labels.long()
        # TO DO
        # We should use BCELoss if a model outputs one-dim vecs
        # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        if sampler.fairness_type == 'eqopp':
            
            yhat_yz = {}
            yhat_y = {}
                        
#             eo_loss = criterion ((F.tanh(logits)+1)/2, (labels+1)/2)
            eo_loss = criterion(logits, labels)
            
            for tmp_yz in sampler.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[sampler.yz_index[tmp_yz]])) / sampler.yz_len[tmp_yz]
                
            for tmp_y in sampler.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[sampler.y_index[tmp_y]])) / sampler.y_len[tmp_y]
            
            # lb1 * loss_z1 + (1-lb1) * loss_z0
            
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                sampler.lb1 += sampler.alpha
            else:
                sampler.lb1 -= sampler.alpha
                
            if sampler.lb1 < 0:
                sampler.lb1 = 0
            elif sampler.lb1 > 1:
                sampler.lb1 = 1 
                
        elif sampler.fairness_type == 'eo':
            
            yhat_yz = {}
            yhat_y = {}
                        
#             eo_loss = criterion ((F.tanh(logits)+1)/2, (labels+1)/2)

            eo_loss = criterion(logits, labels.long())
            
            for tmp_yz in sampler.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(eo_loss[sampler.yz_index[tmp_yz]])) / sampler.yz_len[tmp_yz]
                
            for tmp_y in sampler.y_item:
                yhat_y[tmp_y] = float(torch.sum(eo_loss[sampler.y_index[tmp_y]])) / sampler.y_len[tmp_y]

            max_diff = 0
            pos = (0, 0)

            for _l in range(num_classes):
                # max_diff = 0
                # pos = 0
                for _g in range(1,num_groups):
                    tmp_diff = abs(yhat_yz[(_l, _g)] - yhat_yz[(_l, _g-1)])
                    if max_diff < tmp_diff:
                        max_diff = tmp_diff
                        pos = (_l, _g) if yhat_yz[(_l, _g)] >= yhat_yz[(_l, _g-1)] else (_l, _g-1)

                # # lb update per label
                #     #find plus position
                # if yhat_yz[(_l, pos)] > yhat_yz[(_l, pos-1)]:
                #     target = pos-1
                # else:
                #     target = pos

            pos_label = pos[0]
            pos_group = pos[1]
            for _g in range(num_groups):
                if _g == pos_group:
                    sampler.lbs[pos_label][_g] += sampler.alpha
                else:
                    sampler.lbs[pos_label][_g] -= sampler.alpha / (num_groups-1)
                if sampler.lbs[pos_label][_g] > 1:
                    sampler.lbs[pos_label][_g] = 1
                elif sampler.lbs[pos_label][_g] < 0:
                    sampler.lbs[pos_label][_g] = 0

            #normalize
            sampler.lbs[_l] = [i / sum(sampler.lbs[_l]) for i in sampler.lbs[_l]]
                
            
#             y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
#             y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0
            
#             if y1_diff > y0_diff:
#                 if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
#                     sampler.lb1 += sampler.alpha
#                 else:
#                     sampler.lb1 -= sampler.alpha
#             else:
#                 if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
#                     sampler.lb2 += sampler.alpha
#                 else:
#                     sampler.lb2 -= sampler.alpha
                    
                
#             if sampler.lb1 < 0:
#                 sampler.lb1 = 0
#             elif sampler.lb1 > 1:
#                 sampler.lb1 = 1
                
#             if sampler.lb2 < 0:
#                 sampler.lb2 = 0
#             elif sampler.lb2 > 1:
#                 sampler.lb2 = 1
                
        elif sampler.fairness_type == 'dp':
            yhat_yz = {}
            yhat_y = {}
            
            ones_array = np.ones(len(sampler.y_data))
            ones_tensor = torch.FloatTensor(ones_array).cuda()
#             dp_loss = criterion((F.tanh(logits)+1)/2, ones_tensor) # Note that ones tensor puts as the true label
            dp_loss = criterion(logits, ones_tensor.long())
            
            for tmp_yz in sampler.yz_tuple:
                yhat_yz[tmp_yz] = float(torch.sum(dp_loss[sampler.yz_index[tmp_yz]])) / sampler.z_len[tmp_yz[1]]


            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])
            
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    sampler.lbs[1][1] += sampler.alpha
                    sampler.lbs[1][0] -= sampler.alpha
                else:
                    sampler.lbs[1][1] -= sampler.alpha
                    sampler.lbs[1][0] += sampler.alpha

            else:
                if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
                    sampler.lbs[0][1] -= sampler.alpha
                    sampler.lbs[0][0] += sampler.alpha

                else:
                    sampler.lbs[0][1] += sampler.alpha
                    sampler.lbs[0][0] -= sampler.alpha

            # sum to c?
            if sampler.lbs[1][1] < 0:
                sampler.lbs[1][1] = 0
                sampler.lbs[1][0] = 1
            elif sampler.lbs[1][1] > 1:
                sampler.lbs[1][1] = 1
                sampler.lbs[1][0] = 0

            if sampler.lbs[0][1] < 0:
                sampler.lbs[0][1] = 0
                sampler.lbs[0][0] = 1
            elif sampler.lbs[0][1] > 1:
                sampler.lbs[0][1] = 1
                sampler.lbs[0][0] = 0

        model.train()

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)


