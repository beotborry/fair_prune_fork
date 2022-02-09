import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')
    parser.add_argument('--dataset', required=True, default='', choices=['utkface', 'celeba',
                                                                         'adult', 'compas', 'cifar10s', 'cifar10cg',
                                                                         'credit', 'bank'])
    parser.add_argument('--sen-attr', default='sex', type=str, choices=['sex', 'race', 'marital', 'age'],
                        help='sensitive attribute for dataset')

    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--modelpath', default=None, action='append')
    parser.add_argument('--evalset', default='all', choices=['all', 'train', 'test'])
    parser.add_argument('--img-size', default=176, type=int, help='img size for preprocessing')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--mask-lr', default=0.001, type=float, help='mask learning rate')
    parser.add_argument('--epochs', '--epoch', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio for cifar10s')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--date', default='20200101', type=str, help='experiment date')
    parser.add_argument('--method', default='mp', type=str, required=True,
                        choices=['mfd', 'fairbatch', 'decouple', 'scratch', 'adv_debiasing', 'reweighting', 'scratch_mmd',
                                 'decouple_film', 'hgr', 'pr', 'mfd_decouple', 'mfdf_decouple'])
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD',
                                 'SGD_momentum_decay',
                                 'Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--labelwise', default=False, action='store_true', help='labelwise loader')
    parser.add_argument('--model', default='', required=True, choices=['resnet18', 'shufflenet', 'mlp'])
    parser.add_argument('--num-layer', default=3, type=int, help='number of layers for mlp')

    parser.add_argument('--decouple-at', default=-1, type=int, help='from which layer start to decouple')

    parser.add_argument('--hidden-nodes', default=64, type=int, help='number of hidden nodes for mlp')

    parser.add_argument('--parallel', default=False, action='store_true', help='data parallel')
    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--target', default='Attractive', type=str, help='target attribute for celeba')
    parser.add_argument('--measure-loss', action='store_true', default=False, help='record loss at every epoch')

    parser.add_argument('--lamb', default=1, type=float, help='mmd strength')
    parser.add_argument('--mu', default=1, type=float, help='sparsity strength')
    parser.add_argument('--sigma', default=1, type=float, help='rbf parameter')

    # mask_mmd arguments
    parser.add_argument('--mask-step', default=1, type=int, help='mask update period')
    parser.add_argument('--weight-decay', type=float, default=0., help='The weight decay of loss.')
    parser.add_argument('--resize-ratio', default=None, type=float, help='Ratio for the number of convolutional channels of resized model')
    parser.add_argument('--eta', default=0.001, type=float, help='adversary training learning rate')
    parser.add_argument('--adv-lambda', default=2.0, type=float, help='adversary loss strength')
    
    # decouple test
    parser.add_argument('--group', default=-1, type=int, help='dataloader for specific group only')
    parser.add_argument('--group-estimator', default='true', type=str, choices=['true', 'esti'],
                        help='whether to use group estimator')
    parser.add_argument('--esti-method', default='argmax', choices=['argmax', 'prob'], type=str,
                        help='how to handle group probs')
    parser.add_argument('--no-groupmask', default=False, action='store_true',
                        help='do not insert groupwise mask layer')
    parser.add_argument('--esti-truemmd', default=False, action='store_true',
                        help='how to handle group probs')
    parser.add_argument('--emb-dim', default=16, type=int, help='embedding dimension for group id')
    parser.add_argument('--with-trueid', default=False, action='store_true',
                        help='how to handle group probs')
    parser.add_argument('--no-film-residual', default=False, action='store_true',
                        help='do not add film layer output to original')
    parser.add_argument('--gamma', default=0., type=float, help='separability strength')
    parser.add_argument('--weight-norm', default=False, action='store_true', help='separability strength')
    parser.add_argument('--mhe', default=False, action='store_true', help='mhe style')
    parser.add_argument('--decouple', default=False, action='store_true', help='decouple for no baseline models')
    parser.add_argument('--film', default=False, action='store_true', help='film for no baseline models')
    parser.add_argument('--feature-adv', default=False, action='store_true', help='adversary at feature')
#     parser.add_argument('--adv-perhead', default=False, action='store_true', help='decouple for no baseline models')


    # for fair batch
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='learning rate for lambda in FairBatch(default: 0.0001)')
    # reweighting
    parser.add_argument('--iteration', type=int, default=10, help='number of iteration for reweighting')
    parser.add_argument('--reweighting-target-criterion', default='eopp', choices=['dp', 'eo', 'eopp'], type=str, help='target fairness criterion of reweighting method')
    
    #for MFD
    parser.add_argument('--teacher-type', default=None, choices=['resnet12', 'resnet18', 'resnet34', 'resnet50',
                                                                 'mobilenet', 'shufflenet', 'cifar_net', 'mlp'])
    parser.add_argument('--teacher-path', default=None, help='teacher model path')
    parser.add_argument('--target-criterion', default='eo', choices=['dp', 'eo'], type=str,
                        help='target fairness criterion mfd')
    parser.add_argument('--get-teacher-weight', action='store_true', default=False,
                        help='init student model with teacher weight')

    # for hgr
    parser.add_argument('--batchRenyi', type=int, default=128,
                        help='batch size for renyi estimation')

    # for mixmatch
    parser.add_argument('--mixmatch', action='store_true', default=False,
                        help='apply mixmatch approach that includes mixing up samples for decoupled model')
    parser.add_argument('--apply_mixmatchloss', action='store_true', default=False,
                        help='apply mixmatch loss for decoupled model')
    parser.add_argument('--apply-consistencyloss', action='store_true', default=False, help='apply consistency loss')



    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.no_groupmask = True if args.film else args.no_groupmask
    args.apply_mixmatchloss = True if args.mixmatch else args.apply_mixmatchloss
    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')

    return args
