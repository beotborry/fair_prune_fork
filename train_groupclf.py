import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import data_handler
from utils import check_log_dir, make_log_name, set_seed, get_accuracy, print_logs_n_init
from networks.resnet.resnet import resnet18
from networks.mlp.mlp_groupclf import MLP
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from scipy.io import savemat
from collections import defaultdict
from sklearn.metrics import confusion_matrix

import time
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')

    parser.add_argument('--dataset', required=True, default='',
                        choices=['utkface', 'celeba', 'adult', 'compas', 'bank'])
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--img-size', default=224, type=int, help='img size for preprocessing')
    parser.add_argument('--target', default='Attractive', type=str, help='target attribute for celeba')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-decay-step', default=100, type=float, help='decay step')

    parser.add_argument('--epochs', '--epoch', default=70, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')

    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD',
                                 'SGD_momentum_decay',
                                 'Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--date', default='group_clf', type=str, help='experiment folder name / date')
    parser.add_argument('--labelwise', default=False, action='store_true', help='labelwise loader')
    parser.add_argument('--balanced', default=False, action='store_true', help='balanced cross entropy loss')
    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--weight-decay', type=float, default=2e-4, help='The weight decay of loss.')
    parser.add_argument('--modelpath', default=None)
    parser.add_argument('--sen-attr', default='sex', choices=['sex', 'race', 'age'])

    parser.add_argument('--num-layer', default=3, type=int, help='number of layers for mlp')
    parser.add_argument('--hidden-nodes', default=64, type=int, help='number of hidden nodes for mlp')
    parser.add_argument('--gamma', default=1.0, type=float, help='focal loss hyperparam')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')

    return args


args = get_args()

def evaluate(model, loader, criterion, cuda=True,
             confusion=False, log_dir='./', log_name='confu', dataset='test'):
    model.eval()
    eval_acc, eval_loss = 0, 0
    eval_data_count = 0
    num_groups = loader.dataset.num_groups

    if confusion:
        num_classes = loader.dataset.num_groups
        confu_mat = defaultdict(lambda: np.zeros((num_classes, num_classes)))

    with torch.no_grad():
        for j, eval_data in enumerate(loader):
            # Get the inputs
            inputs, _, groups, _, _ = eval_data
            labels = groups.long() 
#             if num_groups >2 else groups.float()
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, 1) 
#             if num_groups >2 else (torch.sigmoid(outputs) >= 0.5).float()
            acc = (preds == labels).float().squeeze()
            eval_acc += acc.sum()
            eval_data_count += len(groups)

            if confusion:
                confu_mat['0'] += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(),
                                                   labels=[i for i in range(num_classes)])

        eval_loss = eval_loss / eval_data_count
        eval_acc = eval_acc / eval_data_count

        if confusion:
            savepath = os.path.join(log_dir, log_name + f'_{dataset}_confu')
            print('savepath', savepath)
            savemat(savepath, confu_mat, appendmat=True)

    model.train()
    return eval_loss, eval_acc


def train_epoch(epoch, model, train_loader, optimizer, args, criterion):
    model.train()
    running_acc = 0.0
    running_loss = 0.0
    batch_start_time = time.time()
    num_groups = train_loader.dataset.num_groups

    for i, data in enumerate(train_loader, 1):
        # Get the inputs
        inputs, _, groups, _, _ = data
        targets = groups.long() 
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs)
#         if args.balanced:
#             outputs = outputs.unsqueeze(-1) if num_groups ==2 else outputs
#             targets = targets.long()
        loss = criterion(outputs, targets)
        running_loss += loss.item()
#         binary = True if num_groups == 2 else False
        binary = False
        running_acc += get_accuracy(outputs, targets, binary)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.term == 0:  # print every self.term mini-batches
            running_loss, running_acc, batch_start_time = print_logs_n_init(epoch, args.epochs, i, batch_start_time,
                                                                            running_loss, running_acc, args.term)


def estimate_group_probs(model, loader, cuda=True):
    model.eval()
    with torch.no_grad():
        num_groups = loader.dataset.num_groups
        group_probs = torch.zeros((len(loader.dataset), num_groups))
        group_probs = group_probs.cuda() if cuda else group_probs
        for i, data in enumerate(loader, 1):
            inputs, _, _, _, idx_tuple = data
            if cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1) 
#             if num_groups >2 else torch.stack((1-torch.sigmoid(outputs),torch.sigmoid(outputs)), dim=1)
            idxs = idx_tuple[0]
            group_probs[idxs] = outputs

    return group_probs


def get_weights(loader, cuda=True):
    num_groups = loader.dataset.num_groups
    data_counts = torch.zeros(num_groups)
    data_counts = data_counts.cuda() if cuda else data_counts

    for data in loader:
        inputs, _, groups, _, _ = data
        for g in range(num_groups):
            data_counts[g] += torch.sum((groups == g))

    weights = data_counts / data_counts.min()
    return weights, data_counts


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)
    

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def main():

    torch.backends.cudnn.enabled = True
    seed = args.seed
    set_seed(seed)
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    model_type = 'resnet18' if args.dataset =='utkface' else 'mlp'
    log_name = '{}_seed{}_epochs{}_bs{}_lr{}_decay{}'.format(model_type, args.seed, args.epochs, args.batch_size, args.lr, args.weight_decay)
    log_name += '_labelwise' if args.labelwise else ''
    log_name += '_balanced_gamma{}'.format(args.gamma) if args.balanced else ''
    dataset = args.dataset
    save_dir = os.path.join('./trained_models', args.date, dataset)
    log_dir = os.path.join('./results', args.date, dataset)
    check_log_dir(save_dir)
    check_log_dir(log_dir)

    ########################## get dataloader ################################
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset, img_size=args.img_size,
                                                        batch_size=args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers,
                                                        target=args.target,
                                                        labelwise=args.labelwise,
                                                        drop_last=False,
                                                        sen_attr=args.sen_attr
                                                        )
    num_classes, num_groups, train_loader, test_loader, _ = tmp
    
    ########################## get model & optimizer ##################################

    num_groups = train_loader.dataset.num_groups
    if args.dataset == 'utkface':
        group_identifier = resnet18(num_classes=num_groups)
    else:
        # group_identifier = MLP(feature_size=args.img_size, hidden_dim=args.hidden_nodes,
        #                        num_classes=num_groups, num_layer=args.num_layer, use_norm=args.balanced, no_binarized=True)
        group_identifier = MLP(feature_size=args.img_size, hidden_dim=args.hidden_nodes,
                               num_classes=num_groups, num_layer=args.num_layer)

    if args.modelpath is not None:
        group_identifier.load_state_dict(torch.load(args.modelpath))

    if args.cuda:
        group_identifier.cuda()
    group_identifier.train()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(group_identifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    else:
        optimizer = optim.SGD(group_identifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)

    weights = None
    if args.balanced:
        weights, data_counts = get_weights(train_loader, cuda=args.cuda)
        cls_num_list = []
        for i in range(num_groups):
            cls_num_list.append(data_counts[i].item())
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    ########################## train & evaluate ##################################
    if args.mode != 'eval':
        for epoch in range(args.epochs):
            if args.balanced:
#                 adjust_learning_rate(optimizer, epoch, args)
#                 idx = epoch // 160
#                 betas = [0, 0.9999]
#                 effective_num = 1.0 - np.power(betas[idx], cls_num_list)
#                 per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
#                 per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
#                 per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
#                 criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, cls_num_list)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
#                 per_cls_weights = None
#                 criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
                criterion = FocalLoss(weight=per_cls_weights, gamma=args.gamma).cuda()
                
            train_epoch(epoch, group_identifier, train_loader, optimizer, args, criterion)
            eval_loss, eval_acc = evaluate(group_identifier, test_loader, criterion, cuda=args.cuda)
            print('[{}/{}] Test Loss: {:.3f} Test Acc: {:.3f}'.format
                  (epoch + 1, args.epochs, eval_loss, eval_acc))
        scheduler.step(eval_loss)

    evaluate(group_identifier, train_loader, criterion, cuda=args.cuda, confusion=True,
             log_dir=log_dir, log_name=log_name, dataset='train')
    evaluate(group_identifier, test_loader, criterion, cuda=args.cuda, confusion=True,
             log_dir=log_dir, log_name=log_name, dataset='test')

    # estimate group probs
    group_probs_train = estimate_group_probs(group_identifier, train_loader, cuda=args.cuda)
    group_probs_test = estimate_group_probs(group_identifier, test_loader, cuda=args.cuda)
    torch.save(group_probs_train.cpu(), os.path.join(log_dir, 'group_probs_seed{}_train.pt'.format(args.seed)))
    torch.save(group_probs_test.cpu(), os.path.join(log_dir, 'group_probs_seed{}_test.pt'.format(args.seed)))
    model_savepath = os.path.join(save_dir, log_name + '.pt')
    torch.save(group_identifier.state_dict(), model_savepath)
    print('Saved estimated group id')
    print('Done!')

if __name__ == '__main__':
    main()
