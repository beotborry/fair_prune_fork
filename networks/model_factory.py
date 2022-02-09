import torch.nn as nn


from networks.mlp.mlp import MLP
from networks.mlp.mlp_decouple import MLP_decouple
from networks.mlp.mlp_decouple_film import MLP_decouple_Film

from networks.resnet.resnet import resnet18
from networks.resnet.resnet_decouple import resnet18_decouple
from networks.resnet.resnet_decouple_film import resnet18_decouple_film

from networks.shufflenet.shufflenet import shufflenet_v2_x1_0


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(target_model, num_classes, img_size, pretrained=False, method=None, num_groups=2, no_groupmask=False,
                  num_layer=2, hidden_nodes=50, emb_dim=16, weight_norm=False, decouple=False, film=False,
                  for_cifar=False, decouple_at=-1):

        if target_model == 'mlp':
            model = get_mlp(method, img_size, hidden_nodes=hidden_nodes, num_layer=num_layer, num_classes=num_classes,
                            num_groups=num_groups, no_groupmask=no_groupmask, emb_dim=emb_dim, weight_norm=weight_norm,
                            decouple=decouple, film=film, decouple_at=decouple_at)

        elif target_model.startswith('resnet'):
            model = get_resnet(method, num_classes, pretrained=pretrained,
                               num_groups=num_groups, no_groupmask=no_groupmask, emb_dim=emb_dim,
                               weight_norm=weight_norm, decouple=decouple, film=film, for_cifar=for_cifar,
                               decouple_at=decouple_at)

        elif target_model == 'shufflenet':
            model = get_shufflenet(method, num_classes, pretrained=pretrained)

        else:
            raise NotImplementedError

        return model


def get_mlp(method, img_size, hidden_nodes, num_layer, num_classes, num_groups=2,
            no_groupmask=False, emb_dim=32, weight_norm=False, decouple=False, film=False, decouple_at=-1):

    if method == 'decouple':
        return MLP_decouple(feature_size=img_size, hidden_dim=hidden_nodes, num_classes=num_classes,
                            num_layer=num_layer, num_groups=num_groups, no_groupmask=no_groupmask,
                            decouple_at=decouple_at)
    elif method == 'decouple_film':
        return MLP_decouple_Film(feature_size=img_size, hidden_dim=hidden_nodes, num_classes=num_classes,
                                 num_layer=num_layer, num_groups=num_groups, emb_dim=emb_dim,
                                 normalize_weight=weight_norm, decouple_at=decouple_at)
    else:
        if decouple:
            if film:
                return MLP_decouple_Film(feature_size=img_size, hidden_dim=hidden_nodes, num_classes=num_classes,
                                         num_layer=num_layer, num_groups=num_groups, emb_dim=emb_dim,
                                         normalize_weight=weight_norm, decouple_at=decouple_at)
            else:
                return MLP_decouple(feature_size=img_size, hidden_dim=hidden_nodes, num_classes=num_classes,
                                    num_layer=num_layer, num_groups=num_groups, no_groupmask=no_groupmask,
                                    decouple_at=decouple_at)
        else:
            # print(img_size, hidden_nodes, num_classes, num_layer)
            return MLP(feature_size=img_size, hidden_dim=hidden_nodes, num_classes=num_classes,
                       num_layer=num_layer)



def get_resnet(method, num_classes, pretrained=False, num_groups=2, no_groupmask=False, emb_dim=32, weight_norm=False,
               decouple=False, film=False, for_cifar=False, decouple_at=-1):

    if pretrained:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    else:
        if method == 'decouple':
            model = resnet18_decouple(num_classes=num_classes, num_groups=num_groups, no_groupmask=no_groupmask,
                                      for_cifar=for_cifar, decouple_at=decouple_at)
        elif method == 'decouple_film':
            model = resnet18_decouple_film(num_classes=num_classes, num_groups=num_groups, emb_dim=emb_dim,
                                           normalize_weight=weight_norm, for_cifar=for_cifar, decouple_at=decouple_at)
        else:
            if decouple:
                if film:
                    model = resnet18_decouple_film(num_classes=num_classes, num_groups=num_groups, emb_dim=emb_dim,
                                                   normalize_weight=weight_norm, for_cifar=for_cifar, decouple_at=decouple_at)
                else:
                    model = resnet18_decouple(num_classes=num_classes, num_groups=num_groups, no_groupmask=no_groupmask,
                                              for_cifar=for_cifar, decouple_at=decouple_at)
            else:
                model = resnet18(pretrained=False, num_classes=num_classes, for_cifar=for_cifar)

    return model


def get_shufflenet(method, num_classes, pretrained=False):
    if pretrained:
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    else:
        model = shufflenet_v2_x1_0(pretrained=False, num_classes=num_classes)

    return model
