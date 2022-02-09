import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.resnet.resnet_util import conv3x3
from networks.resnet.resnet import ResNet
from networks.network_utils import weight_norm


class noReLUBasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(noReLUBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class ResNet_decouple_Film(ResNet):

    def __init__(self, block, layers, num_classes=1000,
                 width_per_group=64, num_groups=2, emb_dim=32, normalize_weight=False, for_cifar=False, decouple_at=-1):
        super(ResNet_decouple_Film, self).__init__(block=block, layers=layers,
                                                   num_classes=num_classes, width_per_group=width_per_group,
                                                   for_cifar=for_cifar)
        self.num_groups = num_groups

        # self.emb = torch.nn.Embedding(num_groups, emb_dim)
        self.emb = torch.nn.Linear(num_groups, emb_dim)

        self.decouple_at = decouple_at
        if decouple_at == -1:
            self.film = nn.Linear(emb_dim, 512 * 2)

            self.fc = nn.ModuleList()
            for g in range(num_groups):
                if normalize_weight:
                    self.fc.append(weight_norm(nn.Linear(512 * block.expansion, num_classes, bias=False)))
                else:
                    self.fc.append(nn.Linear(512 * block.expansion, num_classes))

        elif decouple_at == -2:
            self.film = nn.Linear(emb_dim, 256 * 2)
            self.head = nn.ModuleList()
            for g in range(num_groups):
                head = []
                head.append(self._make_layer(block, 512, layers[3], stride=2, dilate=False, inplanes=256 * block.expansion))
                head.append(nn.ReLU())
                head.append(nn.AdaptiveAvgPool2d((1, 1)))
                head.append(nn.Flatten())
                head.append(nn.Linear(512 * block.expansion, num_classes))
                self.head.append(nn.Sequential(*head))

    def _forward_impl(self, x, group=None, get_inter=False, true_group=None, no_film_residual=False, get_logit=False):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        b1 = F.relu(self.layer1(x))
        b2 = F.relu(self.layer2(b1))
        if self.decouple_at == -1:
            b3 = F.relu(self.layer3(b2))
            b4 = self.layer4(b3)

            # if true_group is not None:
            #     selected_group = F.one_hot(true_group.long(), self.num_groups)
            # else:
            selected_group = F.one_hot(group.long(), self.num_groups) if not len(group.shape) > 1 else group
            group_emb = self.emb(selected_group.float())

            film = self.film(group_emb)
            gamma = film[:, : 512]  # .view(film4.size(0),-1,1,1)
            beta = film[:, 512:]  # .view(film4.size(0),-1,1,1)
            gamma_norm = gamma.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta.norm(p=2, dim=1, keepdim=True).detach()

            gamma = gamma.div(gamma_norm).view(film.size(0), -1, 1, 1)
            beta = beta.div(beta_norm).view(film.size(0), -1, 1, 1)
            b4_new = gamma * b4 + beta

            if no_film_residual:
                b4 = F.relu(b4_new)
            else:
                b4 = F.relu(b4_new) + F.relu(b4)

            h = self.avgpool(b4)
            h = torch.flatten(h, 1)

            temp_b4 = 0
            out = 0
            for i in range(self.num_groups):
                tmp_out = self.fc[i](h)
                tmp_logit = tmp_out

                if len(group.shape) > 1:
                    # when prob is given
                    tmp_out = F.softmax(tmp_out, dim=1)

                group_for_inter_feature = group if true_group is None else true_group
                temp_b4 += tmp_logit * (group_for_inter_feature == i)[:, None]
                out += tmp_out * (group == i)[:, None] if not len(group.shape) > 1 else tmp_out * group[:, i].unsqueeze(-1)

            b4 = temp_b4 if get_logit else b4
            if get_inter or get_logit:
                return b1, b2, b3, b4, out
            else:
                return out

        elif self.decouple_at == -2:
            b3 = self.layer3(b2)
            # if true_group is not None:
            #     selected_group = F.one_hot(true_group.long(), self.num_groups)
            # else:
            selected_group = F.one_hot(group.long(), self.num_groups) if not len(group.shape) > 1 else group
            group_emb = self.emb(selected_group.float())

            film = self.film(group_emb)
            gamma = film[:, : 256]  # .view(film4.size(0),-1,1,1)
            beta = film[:, 256:]  # .view(film4.size(0),-1,1,1)
            gamma_norm = gamma.norm(p=2, dim=1, keepdim=True).detach()
            beta_norm = beta.norm(p=2, dim=1, keepdim=True).detach()

            gamma = gamma.div(gamma_norm).view(film.size(0), -1, 1, 1)
            beta = beta.div(beta_norm).view(film.size(0), -1, 1, 1)
            b3_new = gamma * b3 + beta

            if no_film_residual:
                b3 = F.relu(b3_new)
            else:
                b3 = F.relu(b3_new) + F.relu(b3)

            temp_b3 = 0
            out = 0
            for i in range(self.num_groups):
                tmp_out = self.head[i](b3)
                tmp_logit = tmp_out

                if len(group.shape) > 1:
                    # when prob is given
                    tmp_out = F.softmax(tmp_out, dim=1)

                group_for_inter_feature = group if true_group is None else true_group
                temp_b3 += tmp_logit * (group_for_inter_feature == i)[:, None]
                out += tmp_out * (group == i)[:, None] if not len(group.shape) > 1 else tmp_out * group[:, i].unsqueeze(
                    -1)

            b3 = temp_b3 if get_logit else b3
            if get_inter or get_logit:
                return b1, b2, b3, b3, out
            else:
                return out

    def forward(self, x, group=None, get_inter=False, true_group=None, no_film_residual=False, get_logit=False):
        return self._forward_impl(x, group, get_inter, true_group, no_film_residual, get_logit)


def resnet18_decouple_film(**kwargs):
    return ResNet_decouple_Film(noReLUBasicBlock, [2, 2, 2, 2], **kwargs)
