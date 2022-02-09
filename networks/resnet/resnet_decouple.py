import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.network_utils import Mask
from torch.distributions.normal import Normal
from networks.resnet.resnet import ResNet
from networks.resnet.resnet_util import conv3x3
from networks.sequential import Sequential
norm_mean, norm_var = 0.0, 1.0


class SparseBasicBlock_CM(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, num_groups=2):
        super(SparseBasicBlock_CM, self).__init__()
        # self.num_groups = num_groups
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # self.mask1 = torch.nn.ModuleList()
        # for g in range(self.num_groups):
        #     m = Normal(torch.tensor([norm_mean] * planes), torch.tensor([norm_var] * planes)).sample()
        #     self.mask1.append(Mask(m, planes=True))

        # m = Normal(torch.tensor([norm_mean] * planes), torch.tensor([norm_var] * planes)).sample()
        # self.mask1 = Mask(m, planes=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # self.mask2 = torch.nn.ModuleList()
        # for g in range(self.num_groups):
        #     m = Normal(torch.tensor([norm_mean] * planes), torch.tensor([norm_var] * planes)).sample()
        #     self.mask2.append(Mask(m, planes=True))

        # m = Normal(torch.tensor([norm_mean] * planes), torch.tensor([norm_var] * planes)).sample()
        # self.mask2 = Mask(m, planes=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # tmp = self.mask1[0](out) * (group == 0)[:,None,None,None]
        # for i in range(1,self.num_groups):
        #     tmp += self.mask1[i](out) * (group == i)[:,None,None,None]
        # out = tmp

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        # tmp = self.mask2[0](out) * (group == 0)[:,None,None,None]
        # for i in range(1,self.num_groups):
        #     tmp += self.mask2[i](out) * (group == i)[:,None,None,None]
        # out = tmp

        # out = self.mask2(out)

        return out


class ResNet_decouple(ResNet):

    def __init__(self, block, layers, num_classes=1000,
                 width_per_group=64, num_groups=2, no_groupmask=False, for_cifar=False, decouple_at=-1):
        super(ResNet_decouple, self).__init__(block=block, layers=layers,
                                              num_classes=num_classes, width_per_group=width_per_group,
                                              for_cifar=for_cifar)
        self.num_groups = num_groups

        self.no_groupmask = no_groupmask
        self.decouple_at = decouple_at
        if decouple_at == -1:
            if not no_groupmask:
                self.mask = nn.ModuleList()
                for g in range(num_groups):
                    m = Normal(torch.tensor([norm_mean] * 512), torch.tensor([norm_var] * 512)).sample()
                    self.mask.append(Mask(m, planes=True))
            self.fc = nn.ModuleList()
            for g in range(num_groups):
                self.fc.append(nn.Linear(512 * block.expansion, num_classes))

        elif decouple_at == -2:
            if not no_groupmask:
                self.mask = nn.ModuleList()
                for g in range(num_groups):
                    m = Normal(torch.tensor([norm_mean] * 256), torch.tensor([norm_var] * 256)).sample()
                    self.mask.append(Mask(m, planes=True))

            self.head = nn.ModuleList()
            for g in range(num_groups):
                head = []
                head.append(self._make_layer(block, 512, layers[3], stride=2, dilate=False, inplanes=256 * block.expansion))
                head.append(nn.AdaptiveAvgPool2d((1, 1)))
                head.append(nn.Flatten())
                head.append(nn.Linear(512 * block.expansion, num_classes))
                self.head.append(nn.Sequential(*head))


    def _forward_impl(self, x, group=None, get_inter=False, true_group=None, get_logit=False):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)

        if self.decouple_at == -1:
            b4 = self.layer4(b3)

            if not self.no_groupmask:
                tmp = 0
                for i in range(self.num_groups):
                    if true_group is None:
                        tmp += self.mask[i](b4) * (group == i)[:, None, None, None]
                    else:
                        tmp += self.mask[i](b4) * (true_group == i)[:, None, None, None]
                b4 = tmp

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
            if not self.no_groupmask:
                tmp = 0
                for i in range(self.num_groups):
                    if true_group is None:
                        tmp += self.mask[i](b3) * (group == i)[:, None, None, None]
                    else:
                        tmp += self.mask[i](b3) * (true_group == i)[:, None, None, None]
                b3 = tmp

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


    def forward(self, x, group=None, get_inter=False, true_group=None, get_logit=False):
        return self._forward_impl(x, group, get_inter, true_group, get_logit)


def resnet18_decouple(**kwargs):
    return ResNet_decouple(SparseBasicBlock_CM, [2, 2, 2, 2], **kwargs)
