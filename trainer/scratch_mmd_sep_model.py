from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import time
import os
import numpy as np
from utils import get_accuracy
from scipy.io import savemat
from utils import print_logs_n_init
from collections import defaultdict
import trainer
from sklearn.metrics import confusion_matrix


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.lamb = args.lamb
        self.sigma = args.sigma
        self.kernel = 'rbf'
        self.optimizer = []
        self.scheduler = []
        for g in range(self.num_groups):
            g_optim = optim.Adam(self.model[g].parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.optimizer.append(g_optim)

            g_scheduler = ReduceLROnPlateau(g_optim, patience=5)
            self.scheduler.append(g_scheduler)

    def train(self, model, train_loader, test_loader, epochs):
        log_set = defaultdict(list)
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        optimizer = self.optimizer

        model = self.model
        for g in range(self.num_groups):
            model[g].train()

        distiller = MMDLoss(self.lamb, self.sigma, num_classes, num_groups)

        for epoch in range(self.epochs):
            train_acc, train_total_loss, train_mmd_loss = self._train_epoch(epoch, train_loader, model, distiller)
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss.sum().item(), eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                for g in range(self.num_groups):
                    self.scheduler[g].step(eval_loss[g])

            # log_set['train_acc'].append(train_acc)
            # log_set['train_total_loss'].append(train_total_loss)
            # log_set['train_mmd_loss'].append(train_mmd_loss)
            # log_set['eval_acc'].append(eval_acc)
            # log_set['eval_loss'].append(eval_loss.item())
            # log_set['eval_deopp'].append(eval_deopp)

        print('Training Finished!')
        torch.save(log_set, os.path.join(self.log_dir, self.log_name + '_log.pt'))
        return model

    def _train_epoch(self, epoch, train_loader, model, distiller):
        for g in range(self.num_groups):
            model[g].train()

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
            labels = targets.long()

            with torch.no_grad():
                t_labels = []
                t_outputs = []
                for g in range(self.num_groups):
                    if (groups == g).sum() == 0:
                        continue
                    t_labels.append(labels[groups == g])
                    g_inputs = inputs[groups == g].cuda()
                    g_outputs = model[g](g_inputs)
                    t_outputs.append(g_outputs.cpu())
                t_labels = torch.cat(t_labels, dim=0)
                t_outputs = torch.cat(t_outputs, dim=0)

            loss = []
            total_loss = 0.
            acc = 0.
            for g in range(self.num_groups):
                if (groups == g).sum() == 0:
                    continue
                g_inputs = inputs[groups == g].cuda()
                g_labels = labels[groups == g].cuda()
                g_outputs = model[g](g_inputs)
                s_logits = g_outputs
                ce_loss = self.criterion(model[g], s_logits, g_labels)
                g_acc = get_accuracy(s_logits, g_labels)
                acc += g_acc

                f_s = g_outputs
                mmd_loss = distiller.forward(f_s, f_s_all=t_outputs.cuda(),
                                             labels=g_labels, f_s_all_labels=t_labels.cuda())
                loss.append(ce_loss + mmd_loss)
                total_loss += (ce_loss + mmd_loss).item()
                running_mmd_loss += mmd_loss.item()
                epoch_mmd_loss += mmd_loss.item() * len(g_labels)

                self.optimizer[g].zero_grad()
                loss[g].backward()
                self.optimizer[g].step()

            acc /= self.num_groups
            running_total_loss += total_loss
            running_acc += acc
            epoch_total_loss += total_loss * len(labels)
            epoch_acc += acc * len(labels)
            num_data += len(labels)

            if i % self.term == 0:  # print every self.term mini-batches
                running_total_loss, running_acc, batch_start_time, running_mmd_loss = print_logs_n_init(
                    epoch, self.epochs, i, batch_start_time, running_total_loss,
                    running_acc, self.term, mmd_loss=running_mmd_loss)

        return epoch_acc / num_data, epoch_total_loss / num_data, epoch_mmd_loss / num_data

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    def evaluate(self, model, loader, criterion, groupwise=False, group_probs=None, esti_method='prob', decouple=False):

        num_groups = self.num_groups
        num_classes = self.num_classes

        for g in range(num_groups):
            model[g].eval()

        eval_acc = 0
        eval_loss = torch.zeros(num_groups)
        eval_eopp_list = torch.zeros(num_groups, num_classes)
        eval_data_count = torch.zeros(num_groups, num_classes)

        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, groups, targets, idxs = eval_data
                #
                # labels = labels.long() if num_classes >2 else labels.float()
                labels = targets.long()

                for g in range(num_groups):
                    if (groups == g).sum() == 0:
                        continue
                    g_inputs = inputs[groups == g].cuda()
                    g_labels = labels[groups == g].cuda()
                    g_outputs = model[g](g_inputs)
                    loss = criterion(model[g], g_outputs, g_labels)

                    eval_loss[g] += loss.item() * len(labels)
                    # binary = True if num_classes == 2 else False
                    # sigmoided = True if self.method =='decouple' else False
                    acc = get_accuracy(g_outputs, g_labels, reduction='none')

                    eval_acc += acc.sum().item()

                    for l in range(num_classes):
                        if (g_labels == l).sum() == 0:
                            continue
                        try:
                            eval_eopp_list[g, l] += acc[g_labels == l].sum().item()
                        except:
                            eval_eopp_list[g, l] += acc.unsqueeze(0)[g_labels == l].sum().item()
                        eval_data_count[g, l] += torch.sum(g_labels == l).item()

            eval_loss = eval_loss / eval_data_count.sum(dim=1)
            eval_acc = eval_acc / eval_data_count.sum()
            eval_eopp_list = eval_eopp_list / eval_data_count
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_max_eopp = torch.max(eval_max_eopp).item()

        for g in range(num_groups):
            model[g].train()
        return eval_loss, eval_acc, eval_max_eopp

    def save_model(self, save_dir, log_name="", model=None):
        for g in range(self.num_groups):
            model_to_save = self.model[g]
            model_savepath = os.path.join(save_dir, log_name + '_group{}.pt'.format(g))
            torch.save(model_to_save.state_dict(), model_savepath)
            print('Group {} Model saved to {}'.format(g, model_savepath))

    def compute_confusion_matix(self, dataloader, dataset='test', log_dir="", log_name="", model=None,
                                group_probs=None, decouple=False):
        from scipy.io import savemat
        from collections import defaultdict

        for g in range(self.num_groups):
            self.model[g].eval()

        confu_mat = defaultdict(lambda: np.zeros((num_classes, num_classes)))
        # print('# of {} data : {}'.format(dataset, len(dataloader.dataset)))
        num_classes = self.num_classes
        predict_mat = {}
        output_set = torch.tensor([])
        group_set = torch.tensor([], dtype=torch.long)
        target_set = torch.tensor([], dtype=torch.long)

        total = 0
        total_ans = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get the inputs
                inputs, groups, targets, idxs = data
                labels = targets
                groups = groups.long()

                for g in range(self.num_groups):
                    if (groups == g).sum() == 0:
                        continue
                    g_inputs = inputs[groups == g].cuda()
                    g_labels = labels[groups == g].cuda()
                    g_outputs = self.model[g](g_inputs)
                    g_pred = torch.argmax(g_outputs, 1)

                    group_set = torch.cat((group_set, groups[groups == g].cpu()))
                    target_set = torch.cat((target_set, g_labels.cpu()))
                    output_set = torch.cat((output_set, g_outputs.cpu()))
                    total_ans += (g_pred == g_labels).cpu().sum()
                    confu_mat[str(g)] += confusion_matrix(g_labels.cpu().numpy(), g_pred.cpu().numpy(),
                                                          labels=[i for i in range(num_classes)])
                total += inputs.shape[0]

        predict_mat['group_set'] = group_set.numpy()
        predict_mat['target_set'] = target_set.numpy()
        predict_mat['output_set'] = output_set.numpy()
        print(log_name)
        savepath = os.path.join(log_dir, log_name + '_{}_confu'.format(dataset))
        print('savepath', savepath)
        savemat(savepath, confu_mat, appendmat=True)
        print('total : ', total)

        savepath_pred = os.path.join(log_dir, log_name + '_{}_pred'.format(dataset))
        savemat(savepath_pred, predict_mat, appendmat=True)

        print('Computed confusion matrix for {} dataset successfully!'.format(dataset))
        print('Accuracy : ', total_ans / float(total))
        return confu_mat


class MMDLoss(nn.Module):
    def __init__(self, lamb, sigma, num_groups, num_classes, kernel='rbf'):
        super(MMDLoss, self).__init__()
        self.lamb = lamb
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_s_all, labels, f_s_all_labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
        else:
            student = f_s.view(f_s.shape[0], -1)
            f_s_all = f_s_all.view(f_s_all.shape[0], -1)

        mmd_loss = 0

        with torch.no_grad():
            _, sigma_avg = self.pdist(f_s_all, student, sigma_base=self.sigma, kernel=self.kernel)
        for c in range(self.num_classes):
            target_joint = f_s_all[f_s_all_labels == c].clone().detach()

            if len(student[labels == c]) == 0:
                continue

            K_SSg, _ = self.pdist(target_joint, student[labels == c],
                                          sigma_base=self.sigma, kernel=self.kernel)

            K_SgSg, _ = self.pdist(student[labels == c], student[labels == c],
                                   sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            K_SS, _ = self.pdist(target_joint, target_joint,
                                 sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss = torch.clamp(K_SS.mean() + K_SgSg.mean() - 2 * K_SSg.mean(), 0.0, np.inf).mean()

        loss = (1 / 2) * self.lamb * mmd_loss

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
                res = torch.exp(-res / (2 * (sigma_base) * sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg