import torch
import numpy as np
import random
import os
import time


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_accuracy(outputs, labels, binary=False, sigmoid_output=False, reduction='mean', mixmatched=False):
    with torch.no_grad():
        # if multi-label classification
        if len(labels.size())>1 and not mixmatched:
            outputs = (outputs>0.0).float()
            correct = ((outputs==labels)).float().sum()
            total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
            avg = correct / total
            return avg.item()
        if binary:
            if sigmoid_output:
                predictions = (outputs >= 0.5).float()
            else:
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
        else:
            predictions = torch.argmax(outputs, 1)

        if mixmatched:
            labels = torch.argmax(labels, 1)
        c = (predictions == labels).float().squeeze()

        if reduction == 'none':
            return c
        else:
            accuracy = torch.mean(c)
            return accuracy.item()


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def make_log_name(args):
    log_name = args.model

    if args.mode == 'eval' and args.method != 'mfd_decouple':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]

    else:
        if (args.decouple or args.method.startswith('decouple')):
            log_name += '_decouple'
            log_name += '_at{}'.format(args.decouple_at)

        if (args.decouple or args.method == 'decouple') and not args.no_groupmask:
            log_name += '_masklr{}'.format(args.mask_lr)
            log_name += '_maskstep{}'.format(args.mask_step)

        log_name += '_film' if args.film else ''
        log_name += '_{}'.format(args.sen_attr) if args.dataset in ['adult', 'compas', 'bank', 'credit'] else ''
        log_name += '_seed{}_epochs{}_bs{}_lr{}_decay{}'.format(args.seed, args.epochs, args.batch_size, args.lr, args.weight_decay)

        if args.labelwise:
            log_name += '_labelwise'

        if args.method =='adv_debiasing':
            log_name += '_adv_lamb{}_eta{}_constraint{}'.format(args.adv_lambda, args.eta, args.target_criterion)

        elif args.method.startswith('decouple'):
            log_name += '_{}'.format(args.group_estimator)
            log_name += '_{}'.format(args.esti_method) if args.group_estimator == 'esti' else ''
            log_name += '_lamb{}'.format(args.lamb)
            log_name += '_with_trueid' if args.with_trueid else ''
            if args.method == 'decouple_film':
                log_name += '_gamma{}'.format(args.gamma) if args.gamma != 0 else ''
                log_name += '_no_resi' if args.no_film_residual else ''

        elif args.method == 'fairbatch':
            log_name += '_alpha{}'.format(args.alpha)

        elif args.method == 'reweighting':
            log_name += '_constraint{}_eta{}_iter{}'.format(args.reweighting_target_criterion, args.eta, args.iteration)

        elif args.method == 'scratch_mmd' or args.method.startswith('mfd'):
            log_name += '_lamb{}'.format(args.lamb)
            if args.get_teacher_weight:
                log_name += '_tweight_init'

        elif args.method == 'hgr':
            log_name += '_lamb{}_renbatch{}'.format(args.lamb, args.batchRenyi)

        elif args.method == 'pr':
            log_name += '_lamb{}'.format(args.lamb)

        elif args.group != -1:
            log_name += '_group{}'.format(str(args.group))

            if args.apply_mixmatchloss or args.apply_consistencyloss:
                log_name += '_lamb{}'.format(args.lamb)

    if args.resize_ratio is not None:
        log_name += '_resizeratio{}'.format(args.resize_ratio)
               
    return log_name


def print_logs_n_init(epoch, total_epochs, it, bst, loss, acc, term, mmd_loss=None, cos_loss=None):
    avg_batch_time = time.time() - bst
    print_state = '[{}/{}, {:5d}] Method: decouple Train Loss: {:.3f} Train Acc: {:.3f}'.format(
        epoch + 1, total_epochs, it, loss / term, acc / term)
    print_state += ' MMD Loss {:.3f}'.format(mmd_loss / term) if mmd_loss is not None else ''
    print_state += ' Cos Loss {:.3f}'.format(cos_loss / term) if cos_loss is not None else ''
    print_state += ' [{:.3f} s/batch]'.format(avg_batch_time / term)
    print(print_state)
    running_loss = 0.0
    running_acc = 0.0
    running_mmd_loss = 0.0
    running_cos_loss = 0.0
    bst = time.time()
    if mmd_loss is not None:
        if cos_loss is not None:
            return running_loss, running_acc, bst, running_mmd_loss, running_cos_loss
        else:
            return running_loss, running_acc, bst, running_mmd_loss
    else:
        return running_loss, running_acc, bst


