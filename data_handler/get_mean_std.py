def get_mean_std(dataset, skew_ratio=0.8, group=-1):
    mean, std = None, None

    if dataset == 'utkface':
        if group==-1:
            mean = [0.5960, 0.4573, 0.3921]
            std = [0.2586, 0.2314, 0.2275]
        elif group==0:
            mean = [0.6262, 0.4836, 0.4209]
            std = [0.2473, 0.2217, 0.2199]
        elif group==1:
            mean = [0.5053, 0.3745, 0.3101]
            std = [0.2536, 0.2220, 0.2149]
        elif group==2:
            mean = [0.6317, 0.4979, 0.4341]
            std = [0.2727, 0.2492, 0.2441]
        elif group==3:
            mean = [0.5927, 0.4485, 0.3751]
            std = [0.2564, 0.2270, 0.2209]

    elif dataset == 'cifar10s':
        # from data_handler.cifar10s import CIFAR10_S
        # preprocessing = transforms.Compose([transforms.ToTensor()])
        # pre_dataset = CIFAR10_S(root='./data/cifar10', split='train', transform=preprocessing,
        #                         skewed_ratio=skew_ratio, labelwise=False)
        # mean = tuple(np.mean(pre_dataset.dataset['image'] / 255., axis=(0, 1, 2)))
        # std = tuple(np.std(pre_dataset.dataset['image'] / 255., axis=(0, 1, 2)))
        if group == -1:
            mean = [0.4871, 0.4811, 0.4632] # for skew 0.8
            std = [0.2431, 0.2414, 0.2506] # for skew 0.8
        elif group == 0:
            mean = [0.4818, 0.4818, 0.4818]
            std = [0.2376, 0.2376, 0.2376]
        elif group == 1:
            mean = [0.4924, 0.4804, 0.4446]
            std = [0.2483, 0.2451, 0.2617]

    elif dataset == 'cifar10cg':
        if group == 0:
            mean = [0.4809, 0.4809, 0.4809]
            std = [0.2392, 0.2392, 0.2392]
        elif group == 1:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]

    elif dataset == 'celeba':
        # default target is 'Attractive'
        mean = [0.5063, 0.4258, 0.3832]
        std = [0.3107, 0.2904, 0.2897]

    elif dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return mean, std

