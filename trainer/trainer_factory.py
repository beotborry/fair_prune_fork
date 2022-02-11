import torch
import numpy as np
import os
import importlib

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR
from sklearn.metrics import confusion_matrix
from utils import get_accuracy

method_dict = {'scratch': 'trainer.vanilla_train',
               'decouple': 'trainer.decouple',
               'decouple_film': 'trainer.decouple_film',
               'fairbatch': 'trainer.fairbatch',
               'adv_debiasing': 'trainer.adv_debiasing_revised',
               'reweighting': 'trainer.reweighting',
               'mfd': 'trainer.mfd',
               'scratch_mmd': 'trainer.scratch_mmd',
               'hgr': 'trainer.hgr',
               'pr': 'trainer.prejudice_remover',
               'mfd_decouple': 'trainer.mfd_decouple_sep_model',
               'mfdf_decouple': 'trainer.scratch_mmd_sep_model',}


class TrainerFactory:

    def __init__(self):
        pass
    @staticmethod
    def get_trainer(method, **kwargs):

        if method not in method_dict.keys():
            raise Exception('Not allowed method')
        trainer = importlib.import_module(method_dict[method])
        return trainer.Trainer(**kwargs)


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''
    def __init__(self, model, args, log_dir=None, log_name=None, num_classes=None, num_groups=None):
        self.cuda = args.cuda
        self.term = args.term
        self.lr = args.lr
        self.epochs = args.epochs
        self.model = model
        if not (args.method == 'mfd_decouple' or args.method == 'mfdf_decouple'):
            if 'SGD' in args.optimizer:
                self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                self.scheduler = MultiStepLR(self.optimizer, [self.epochs//3, self.epochs//3 * 2], gamma=0.1)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
                #self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)

        self.optim_type = args.optimizer

        self.img_size = args.img_size if not 'cifar10' in args.dataset else 32
        self.ce = torch.nn.CrossEntropyLoss()
        self.measure_loss = args.measure_loss
        self.method = args.method

        self.log_dir = log_dir
        self.log_name = log_name

        self.args = args
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.mixmatch = args.mixmatch
        self.group = args.group
        # TODO : implemenatation for applying only mixmatchloss (should prepare unlabeled dataloader)
        self.apply_mixmatchloss = args.apply_mixmatchloss
        self.apply_consistencyloss = args.apply_consistencyloss
        if self.apply_consistencyloss or self.apply_mixmatchloss:
            self.lamb = args.lamb

    def evaluate(self, model, loader, criterion, groupwise=False, group_probs=None, esti_method='prob', decouple=False):
        model.eval()
        num_groups = self.num_groups
        num_classes = self.num_classes

        eval_acc = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda()
        eval_loss = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda()
        eval_eopp_list = torch.zeros(num_groups, num_classes).cuda()
        eval_data_count = torch.zeros(num_groups, num_classes).cuda()
        
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, targets, idxs = eval_data
                #
                # labels = labels.long() if num_classes >2 else labels.float()
                labels = targets.long()
                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    groups = groups.cuda()

                if self.method.startswith('decouple'):
                    if group_probs is None:
                        if self.method == 'decouple_film':
                            outputs = model(inputs, groups, no_film_residual=self.no_film_residual)
                        else:
                            outputs = model(inputs, groups)
                    else:
                        assert idxs is not None
                        idxs = idxs[0].cuda() if self.cuda else idxs[0]
                        estimated_group = group_probs[idxs] if esti_method == 'prob' else group_probs[idxs].argmax(dim=1)
                        if self.method == 'decouple_film':
                            outputs = model(inputs, group=estimated_group, no_film_residual=self.no_film_residual)
                        else:
                            outputs = model(inputs, group=estimated_group)
                    loss = criterion(model, outputs, labels, esti_method=esti_method)
                else:
                    outputs = model(inputs) if not decouple else model(inputs, groups)
                    loss = criterion(model, outputs, labels)

                eval_loss += loss.item() * len(labels)
                # binary = True if num_classes == 2 else False
                # sigmoided = True if self.method =='decouple' else False
                acc = get_accuracy(outputs, labels, reduction='none')

                eval_acc += acc.sum()

                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
                        eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

            eval_loss = eval_loss / eval_data_count.sum() if not groupwise else eval_loss / eval_data_count
            eval_acc = eval_acc / eval_data_count.sum() if not groupwise else eval_acc / eval_data_count
            eval_eopp_list = eval_eopp_list / eval_data_count

            # for eopp
            #group0_fn = eval_data_count[0, 1] - eval_eopp_list[0, 1]
            #group0_tp = eval_eopp_list[0, 1]
            #group1_fn = eval_data_count[1, 1] - eval_eopp_list[1, 1]
            #group1_tp = eval_eopp_list[1, 1]

            #pivot = (group0_tp + group1_tp) / (group0_fn + group0_tp + group1_fn + group1_tp)
            #group0_tpr = group0_tp / (group0_fn + group0_tp)
            #group1_tpr = group1_tp / (group1_fn + group1_tp)
            #eval_max_eopp = max(abs(group0_tpr - pivot), abs(group1_tpr - pivot))
            
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_max_eopp = torch.max(eval_max_eopp).item()

        model.train()
        return eval_loss, eval_acc, eval_max_eopp
    
    def save_model(self, save_dir, log_name="", model=None):
        model_to_save = self.model if model is None else model
        model_savepath = os.path.join(save_dir, log_name + '.pt')
        torch.save(model_to_save.state_dict(), model_savepath)

        print('Model saved to %s' % model_savepath)

    def compute_confusion_matix(self, dataloader, dataset='test', log_dir="", log_name="", model=None,
                                group_probs=None, decouple=False):
        from scipy.io import savemat
        from collections import defaultdict

        if model is None:
            model = self.model
        model.eval()

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
                inputs, _, groups, targets, idxs = data
                labels = targets
                groups = groups.long()

                if self.cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    groups = groups.cuda()
                # forward

                if self.method.startswith('decouple'):
                    if group_probs is None:
                        if self.method == 'decouple_film':
                            outputs = model(inputs, groups, no_film_residual=self.no_film_residual)
                        else:
                            outputs = model(inputs, groups)
                    else:
                        assert idxs is not None
                        idxs = idxs[0].cuda() if self.cuda else idxs[0]
                        estimated_group = group_probs[idxs] if self.esti_method == 'prob' else group_probs[idxs].argmax(dim=1)
                        if self.method == 'decouple_film':
                            outputs = model(inputs, group=estimated_group, no_film_residual=self.no_film_residual)
                        else:
                            outputs = model(inputs, group=estimated_group)
                else:
                    outputs = model(inputs) if not decouple else model(inputs, groups)

                group_set = torch.cat((group_set, groups.cpu()))
                target_set = torch.cat((target_set, targets))
                output_set = torch.cat((output_set, outputs.cpu()))
                # if num_classes > 2:
                pred = torch.argmax(outputs, 1)
                # else:
                #     if self.method == 'decouple':
                #         pred = (outputs >= 0.5).float()
                #     else:
                #         pred = (torch.sigmoid(outputs) >= 0.5).float()

                total += inputs.shape[0]
                total_ans += (pred==labels).cpu().sum()
                group_element = list(torch.unique(groups.cpu()).numpy())
                for i in group_element:
                    mask = groups == i
                    if len(labels[mask]) != 0:
                        confu_mat[str(i)] += confusion_matrix(
                            labels[mask].cpu().numpy(), pred[mask].cpu().numpy(),
                            labels=[i for i in range(num_classes)])

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
        print('Accuracy : ', total_ans/float(total))
        return confu_mat
