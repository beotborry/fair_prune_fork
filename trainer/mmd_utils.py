import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


def psiConverter(fea, D, W, b):
    gamma = 1

    fea = fea.view(fea.shape[0], -1)
    psi_fea = ((2.0 / D) ** (1 / 2)) * torch.cos(((2.0 / gamma) ** (1 / 2)) * torch.matmul(fea, W) + b)

    return psi_fea


def FastMMDEstimator(model, support_loader, D, hook, num_classes, num_groups, lamb, cuda=True, num_batch=10):
    mmd_loss = 0

    n_in = hook.feature.view(hook.feature.shape[0], -1).shape[-1]
    W = torch.normal(mean=torch.zeros(n_in, D), std=torch.ones(n_in, D)).cuda()
    b = torch.FloatTensor(D).uniform_(0., 2. * math.pi).cuda()

    n_batch_dict = defaultdict(lambda: 0)
    fea_dict = defaultdict(lambda: torch.zeros(D).cuda())

    for j, spt_data in enumerate(support_loader, 1):
        spt_inputs, _, spt_groups, spt_labels, _ = spt_data

        if cuda:
            spt_inputs = spt_inputs.cuda()
            spt_labels = spt_labels.cuda()
            spt_groups = spt_groups.cuda()

        hook.renew()
        _ = model(spt_inputs)
        
        psi_feature = psiConverter(hook.feature, D, W, b)
        
        for c in range(num_classes):
            if sum(spt_labels == c) == 0:
                continue
            fea_dict[c] += torch.sum(psi_feature[spt_labels == c], dim=0).clone().detach()
            n_batch_dict[c] += sum(spt_labels == c)

            for g in range(num_groups):
                if sum((spt_labels == c) * (spt_groups == g)) == 0:
                    continue
                fea_dict[(c, g)] += torch.sum(psi_feature[(spt_labels == c) * (spt_groups == g)], dim=0)
                n_batch_dict[(c, g)] += sum((spt_labels == c) * (spt_groups == g))
        
        if j == num_batch:
            break

    for c in range(num_classes):
        for g in range(num_groups):
            mmd_loss += (fea_dict[c] / n_batch_dict[c] - fea_dict[(c, g)] / n_batch_dict[(c, g)]).norm(2)

    mmd_loss = lamb * mmd_loss
    
    return mmd_loss


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

        mmd_loss = 0

        with torch.no_grad():
            _, sigma_avg = self.pdist(student, student, sigma_base=self.sigma, kernel=self.kernel)
        for c in range(self.num_classes):
            target_joint = f_s_all[f_s_all_labels == c].clone().detach()

            if len(student[labels == c]) == 0:
                continue

            K_SSg, sigma_avg = self.pdist(target_joint, student[labels == c],
                                          sigma_base=self.sigma, kernel=self.kernel)

            K_SgSg, _ = self.pdist(student[labels == c], student[labels == c],
                                   sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            K_SS, _ = self.pdist(target_joint, target_joint,
                                 sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss = torch.clamp(K_SS.mean() + K_SgSg.mean() - 2 * K_SSg.mean(), 0.0, np.inf).mean()

        loss = (1/2) * self.lamb * mmd_loss

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
