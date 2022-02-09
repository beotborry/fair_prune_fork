import torch
import numpy as np
import networks
import data_handler
import trainer
from utils import check_log_dir, make_log_name, set_seed
from arguments import get_args
import time
import os 

args = get_args()

def main():

    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join('./trained_models', args.date, dataset, args.method)
    log_dir = os.path.join('./results', args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(log_dir)

    ########################## get dataloader ################################
    kwargs = {'name': args.dataset,
              'batch_size':args.batch_size,
              'seed':args.seed,
              'num_workers':args.num_workers,
              'target':args.target,
              'labelwise':args.labelwise,
              'group_mode':args.group,
              'sen_attr':args.sen_attr,
              'skew_ratio':args.skew_ratio,
              }
    
    if args.method == 'fairbatch':
        fairbatch_kwargs = {'alpha' : args.alpha,
                  'target_fairness' : args.target_criterion}
        kwargs.update(fairbatch_kwargs)

    tmp = data_handler.DataloaderFactory.get_dataloader(**kwargs)
    num_classes, num_groups, train_loader, valid_loader, test_loader = tmp
    
    ########################## get model ##################################
    for_cifar = True if args.dataset.startswith('cifar') else False
    if args.method == 'mfd_decouple' or args.method == 'mfdf_decouple':
        model = []
        for i in range(num_groups):
            group_model = networks.ModelFactory.get_model(args.model, num_classes, args.img_size, pretrained=args.pretrained,
                                                          method=args.method, num_groups=num_groups, num_layer=args.num_layer,
                                                          hidden_nodes=args.hidden_nodes, for_cifar=for_cifar)

            if args.get_teacher_weight:
                group_model.load_state_dict(torch.load(args.teacher_path))
            # group_model.to((i % 2) +1)
            group_model.cuda()
            model.append(group_model)

    else:
        model = networks.ModelFactory.get_model(args.model,num_classes, args.img_size,pretrained=args.pretrained,
                                                method=args.method, num_groups=num_groups,
                                                no_groupmask=args.no_groupmask, num_layer=args.num_layer,
                                                hidden_nodes=args.hidden_nodes, emb_dim=args.emb_dim,
                                                weight_norm=args.weight_norm, decouple=args.decouple, film=args.film,
                                                for_cifar=for_cifar, decouple_at=args.decouple_at)
        model.cuda()
    if args.modelpath is not None:
        if args.method == 'mfd_decouple' and len(args.modelpath) > 0:
            for i in range(num_groups):
                pretrained_dict = torch.load(args.modelpath[i], map_location='cuda:{}'.format((i % 2) + 1))
                model[i].load_state_dict(pretrained_dict)
        else:
            pretrained_dict = torch.load(args.modelpath[0], map_location='cuda:{}'.format(torch.cuda.current_device()))
            if args.method != 'decouple':
                model.load_state_dict(pretrained_dict, strict=False)

            elif args.method == 'decouple':
                dummy_keys = [i for i in model.state_dict().keys() if 'mask' not in i]
                for key1, key2 in zip(dummy_keys, pretrained_dict.keys()):
                    if model.state_dict()[key1].shape == torch.tensor(1).shape:
                        model.state_dict()[key1] = pretrained_dict[key2]
                    else:
                        model.state_dict()[key1][:] = pretrained_dict[key2][:]
            elif args.decouple:
                model_dict = model.state_dict()
                body_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
                model_dict.update(body_dict)
                if args.model == 'resnet18':
                    clf_weight = pretrained_dict['fc.weight']
                    clf_bias = pretrained_dict['fc.bias']
                    for k, v in model_dict.items():
                        if k.startswith('fc'):
                            if 'weight' in k:
                                model_dict[k] = clf_weight
                            else:
                                model_dict[k] = clf_bias
                        else:
                            continue
                model.load_state_dict(model_dict)
    teacher = None
    if args.method.startswith('mfd') and args.mode != 'eval' and args.method != 'mfdf_decouple':
        teacher = networks.ModelFactory.get_model(args.teacher_type, num_classes, args.img_size,
                                                  pretrained=args.pretrained, num_groups=num_groups,
                                                  no_groupmask=True, num_layer=args.num_layer,
                                                  hidden_nodes=args.hidden_nodes, film=False)

        teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cuda')))
#         teacher.load_state_dict(torch.load(args.teacher_path))        
        teacher.cuda() if args.method == 'mfd' else teacher.to(0)

    ########################## get trainer ##################################
    trainer_ = trainer.TrainerFactory.get_trainer(args.method, args=args, model=model,
                                                  log_dir=log_dir, log_name=log_name,
                                                  num_classes=num_classes, num_groups=num_groups)
    if args.method.startswith('mfd'):
        trainer_.teacher = teacher

    ####################### start training or evaluating ####################
    if args.mode == 'train':
        start_t = time.time()
        model = trainer_.train(model, train_loader, test_loader, args.epochs)
        trainer_.save_model(save_dir, log_name, model=model)

        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Total Pruning Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))

    group_probs, group_probs_test = None, None
    if (args.method.startswith('decouple') and args.group_estimator !='true'):
        group_probs, group_probs_test = trainer_._load_pretrained_group_prediction()
        if args.cuda:
            group_probs, group_probs_test = group_probs.cuda(), group_probs_test.cuda()

    if args.evalset == 'all':
        trainer_.compute_confusion_matix(train_loader, 'train', log_dir, log_name, model=model,
                                         group_probs=group_probs, decouple=args.decouple)
        trainer_.compute_confusion_matix(test_loader, 'test',  log_dir, log_name, model=model,
                                         group_probs=group_probs_test, decouple=args.decouple)
    else:
        eval_loader = train_loader if args.evalset == 'train' else test_loader
        eval_probs = group_probs if args.evalset == 'train' else group_probs_test
        trainer_.compute_confusion_matix(eval_loader, args.evalset, log_dir, log_name, model=model,
                                         group_probs=eval_probs, decouple=args.decouple)

    print('Done!')


if __name__ == '__main__':
    main()
