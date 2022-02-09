import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network_utils import weight_norm

class MLP_decouple_Film(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_classes=None, num_layer=3, num_groups=2, emb_dim=16,
                 normalize_weight=False, decouple_at=-1):
        super(MLP_decouple_Film, self).__init__()
        try:  # list
            in_features = self.compute_input_size(feature_size)
        except:  # int
            in_features = feature_size

        self.h_dim = hidden_dim
        self.num_layer = num_layer
        self.num_groups = num_groups
        self.num_classes = num_classes
        self._make_layer(in_features, hidden_dim, num_classes, num_layer, num_groups, emb_dim,
                         normalize_weight, decouple_at)

    def forward(self, feature, group=None, get_inter=False, true_group=None, no_film_residual=False, get_logit=False):
        feature = torch.flatten(feature, 1)
        h = self.features(feature)

        # group_emb = self.emb(group.long()) if true_group is None else self.emb(true_group.long())
        # if true_group is not None:
        #     selected_group = F.one_hot(true_group.long(), self.num_groups)
        # else:
        selected_group = F.one_hot(group.long(), self.num_groups) if not len(group.shape) > 1 else group

        # group_emb = self.emb(group.long()) if true_group is None else self.emb(true_group.long())
        group_emb = self.emb(selected_group.float())
        film = self.film(group_emb)
        gamma = film[:, :self.h_dim]
        beta = film[:, self.h_dim:]
        gamma_norm = gamma.norm(p=2, dim=1, keepdim=True)
        beta_norm = beta.norm(p=2, dim=1, keepdim=True)
        gamma = gamma.div(gamma_norm).view(film.size(0), -1)
        beta = beta.div(beta_norm).view(film.size(0), -1)
        h_new = gamma * h + beta

        h = F.relu(h_new) if no_film_residual else F.relu(h) + F.relu(h_new)

        out = 0
        tmp_h = 0

        for i in range(self.num_groups):
            tmp_out = self.head[i](h)
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

    def _make_layer(self, in_dim, h_dim, num_classes, num_layer, num_groups, emb_dim,
                    normalize_weight=False, decouple_at=-1):

        if num_layer == 1:
            self.features = nn.Identity()
            h_dim = in_dim
        else:
            features = []
            for i in range(num_layer + decouple_at):
                features.append(nn.Linear(in_dim, h_dim) if i == 0 else nn.Linear(h_dim, h_dim))
                if i+1 != num_layer + decouple_at:
                    features.append(nn.ReLU())
            self.features = nn.Sequential(*features)

        # self.emb = torch.nn.Embedding(num_groups, emb_dim)
        self.emb = torch.nn.Linear(num_groups, emb_dim)
        self.film = nn.Linear(emb_dim, h_dim*2)

        self.head = nn.ModuleList()
        for i in range(num_groups):
            head_layers = []
            for j in range(-decouple_at):
                if j+1 == -decouple_at:
                    if normalize_weight:
                        head_layers.append(weight_norm(nn.Linear(h_dim, num_classes, bias=False), name='weight', dim=0))
                    else:
                        head_layers.append(nn.Linear(h_dim, num_classes))
                else:
                    head_layers.append(nn.Linear(h_dim, h_dim))
            self.head.append(nn.Sequential(*head_layers))
