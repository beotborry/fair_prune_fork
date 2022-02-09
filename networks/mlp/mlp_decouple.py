import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network_utils import Mask
from torch.distributions.normal import Normal

norm_mean, norm_var = 0.0, 1.0


class MLP_decouple(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_classes=None, num_layer=3, num_groups=2,
                 no_groupmask=False, decouple_at=-1):
        super(MLP_decouple, self).__init__()
        try: #list
            in_features = self.compute_input_size(feature_size)
        except : #int
            in_features = feature_size

        self.num_layer = num_layer
        self.num_groups = num_groups
        self.num_classes = num_classes
        self._make_layer(in_features, hidden_dim, num_classes, num_layer, num_groups, no_groupmask, decouple_at)

    def forward(self, feature, group=None, get_inter=False, true_group=None, get_logit=False):
        feature = torch.flatten(feature, 1)
        h = self.features(feature)
        out = 0
        tmp_h = 0
        for i in range(self.num_groups):
            tmp_out = self.mask[i](h) if not self.no_groupmask else h
            tmp_out = self.head[i](tmp_out)
            tmp_logit = tmp_out

            if len(group.shape) > 1:
                # when prob is given
                tmp_out = F.softmax(tmp_out, dim=1)

            group_for_inter_feature = group if true_group is None else true_group
            tmp_h += tmp_logit * (group_for_inter_feature == i)[:, None]
            out += tmp_out * (group == i)[:, None] if not len(group.shape) > 1 else tmp_out * group[:, i].unsqueeze(-1)

        h = tmp_h if get_logit else h
        if get_inter or get_logit:
            return h, out
        else:
            return out

    def compute_input_size(self, feature_size):
        in_features = 1
        for size in feature_size:
            in_features = in_features * size

        return in_features
    
    def _make_layer(self, in_dim, h_dim, num_classes, num_layer, num_groups, no_groupmask, decouple_at):
        
        if num_layer == 1:
            self.features = nn.Identity()
            h_dim = in_dim
        else:
            features = []
            for i in range(num_layer + decouple_at):
                features.append(nn.Linear(in_dim, h_dim) if i == 0 else nn.Linear(h_dim, h_dim))
                features.append(nn.ReLU())
            self.features = nn.Sequential(*features)

        self.head = nn.ModuleList()
        for i in range(num_groups):
            head_layers = []
            for j in range(-decouple_at):
                head_layers.append(nn.Linear(h_dim, num_classes) if j + 1 == -decouple_at else nn.Linear(h_dim, h_dim))

            self.head.append(nn.Sequential(*head_layers))

        self.no_groupmask = no_groupmask
        if not no_groupmask:
            self.mask = nn.ModuleList()
            for g in range(num_groups):
                m = Normal(torch.tensor([norm_mean] * h_dim), torch.tensor([norm_var] * h_dim)).sample()
                self.mask.append(Mask(m, for_fc=True))

