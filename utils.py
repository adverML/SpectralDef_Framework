#!/usr/bin/env python3

""" helper functions

author Peter Lorenz
"""
import os
import sys
import re
import datetime
import json
import pdb

import numpy

import time
import math
from inspect import currentframe, getframeinfo

import torch.nn.init as init

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


from datasets.celbahq import CelebaDataset, CelebaDatasetPath
from datasets import smallimagenet

from conf import settings

from collections import OrderedDict

from models.vgg_cif10 import VGG
from models.vgg import vgg16_bn
from models.wideresidual import WideResNet, WideBasic
from models.orig_resnet import wide_resnet50_2


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log, mode='a'):
        str_to_log = str(str_to_log)
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, mode) as f:
                f.write(str_to_log + '\n')
                f.flush()


def save_args_to_file(args, path):
    args_dct = args.__dict__
    with open(path + os.sep + 'args.txt', 'w') as f:
        json_dumps_str = json.dumps(args_dct, indent=4)
        json_string = json.dumps(args_dct, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(json_string)
    

def get_debug_info(msg):
    frameinfo = getframeinfo(currentframe())
    print(msg, ", filename: ", frameinfo.filename, ", line_nr: ", frameinfo.lineno)


def aa_get_mode(args):
    mode = 'std'
    if args.individual:
        mode = 'ind'
    return mode


def get_compose(mean, std):
    compose = [transforms.ToTensor()]
    if not mean == None:
        compose = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return compose


def get_appendix(num_classes, max_num_classes):
    appendix = ''
    if not num_classes == max_num_classes:
        appendix = '_' + str(num_classes)
    return appendix


def get_num_classes(args):
    if args.net == 'cif10' or args.net == 'cif10vgg':
        num_classes = settings.MAX_CLASSES_CIF10
    elif args.net == 'cif100' or args.net == 'cif100vgg':
        num_classes = settings.MAX_CLASSES_CIF100
    elif args.net == 'imagenet' or args.net == 'imagenet64' or args.net == 'imagenet128':
        num_classes = settings.MAX_CLASSES_IMAGENET
    elif args.net == 'imagenet32':
        num_classes = args.num_classes
    elif args.net == 'celebaHQ32' or args.net == 'celebaHQ64' or args.net == 'celebaHQ128':
        num_classes = settings.MAX_CLASSES_CELEBAHQ
    return num_classes


def get_celeba_path(args):
    if args.img_size == 32:
        img_dir = settings.CELEBAHQ32_PATH
    elif args.img_size == 64:
        img_dir = settings.CELEBAHQ64_PATH
    elif args.img_size == 128:
        img_dir = settings.CELEBAHQ128_PATH
    elif args.img_size == 256:
        img_dir = settings.CELEBAHQ256_PATH
    return img_dir


layer_name_cif10 = [
                'conv2_0WB', 'conv2_1WB', 'conv2_2WB', 'conv2_3WB',
                'conv3_0WB', 'conv3_1WB', 'conv3_2WB', 'conv3_3WB',
                'conv4_0WB', 'conv4_1WB', 'conv4_2WB', 'conv4_3WB',
                'almost_last'
                ]


layer_name_cif10vgg = [
            '2_relu', '5_relu', '9_relu', '12_relu',
            '16_relu', '19_relu', '22_relu', '26_relu',
            '29_relu', '32_relu', '36_relu', '39_relu',
            'almost_last'
            ]


def print_infos():
    print( 'FileName: ', sys.argv[0] )
    print( 'Date:     ', datetime.now() )
    print( 'GPU:      ', torch.cuda.get_device_name(torch.cuda.current_device()) )


def get_network(args):
    """ return given network
    """
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_class=settings.NUM_CLASSES)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_class=settings.NUM_CLASSES)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_class=settings.NUM_CLASSES)
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_class=settings.NUM_CLASSES)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=settings.NUM_CLASSES)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=settings.NUM_CLASSES)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    elif args.net == 'wrn2810':
        from models.wideresidual import WideResNet, WideBasic
        depth=28
        widen_factor=10
        net = WideResNet(num_classes=settings.NUM_CLASSES, block=WideBasic, depth=depth, widen_factor=widen_factor)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        # net = net.cuda()
        if args.parallel:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # net = nn.parallel.DistributedDataParallel(net, device_ids=list(range(torch.cuda.device_count())))
            # net = nn.parallel.DataParallel(net, device_ids=[0,1])
            net = nn.parallel.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

        # net = nn.parallel.DistributedDataParallel(net, device_ids=[0,1,2])
        # net = nn.DataParallel(net)
        # net.to(device)
        net = net.cuda()
    return net


def get_training_dataloader(mean=None, std=None, data='Gender', img_dir='./data', batch_size=64, num_workers=8, shuffle=True):
    DATA_SPLIT = '70' # 80 90

    if data == 'Gender' or 'Smiling':
        csv_path = settings.CELEBA_CSV_PATH + 'train_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'
    else: # hair
        csv_path = settings.CELEBA_CSV_PATH + 'train_' + data.lower() + '_hq_ext_' + DATA_SPLIT + '.csv'

    train_transform = transforms.Compose(get_compose(mean, std))

    train_dataset = CelebaDataset(  csv_path=csv_path,
                                    img_dir=img_dir,
                                    data=data,
                                    transform=train_transform)

    train_loader = DataLoader(  dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    return train_loader


def get_validation_dataloader(mean=None, std=None, data='Gender', img_dir='./data', batch_size=64, num_workers=8, shuffle=False):
    DATA_SPLIT = '70' # 80 90

    if data == "Hair_Color":
        csv_path = settings.CELEBA_CSV_PATH + 'valid_' + data.lower() + '_hq_' + DATA_SPLIT + '_long.csv'
    else:
        csv_path = settings.CELEBA_CSV_PATH + 'valid_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'

    val_transform = transforms.Compose(get_compose(mean, std))

    val_dataset = CelebaDataset(csv_path=csv_path,
                                img_dir=img_dir,
                                data=data,
                                transform=val_transform)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return val_loader


def get_test_dataloader(mean=None, std=None, data='Gender', img_dir='./data', batch_size=64, num_workers=8, shuffle=False):
    
    DATA_SPLIT = '70' # 80 90
    
    csv_path = settings.CELEBA_CSV_PATH + 'test_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'
    test_transform = transforms.Compose(get_compose(mean, std))

    test_dataset = CelebaDataset(   csv_path=csv_path,
                                    img_dir=img_dir,
                                    data=data,
                                    transform=test_transform)

    test_loader = DataLoader(   dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    return test_loader


def get_clean_data_dataloader(mean, std, data='Gender', batch_size=64, num_workers=8, shuffle=False):

    DATA_SPLIT = '70' # 80 90
    tmp_normalized = '_norm'
    IMAGE_SIZE = '32'

    csv_path = settings.CELEBA_CSV_PATH + 'classified_' + data.lower() + '_hq_' + DATA_SPLIT + IMAGE_SIZE + tmp_normalized + '.csv'

    clean_data_transform = transforms.Compose(get_compose(mean, std))

    test_dataset = CelebaDataset(   csv_path=csv_path,
                                    img_dir=IMG_DIR,
                                    data=data,
                                    transform=clean_data_transform)

    test_loader = DataLoader(   dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)

    return test_loader


def get_validation_dataloader_path(mean, std, data='Gender', batch_size=64, num_workers=8, shuffle=False):
    DATA_SPLIT = '70' # 80 90

    csv_path = settings.CELEBA_CSV_PATH + 'valid_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'

    validation_transform = transforms.Compose(get_compose(mean, std))

    val_dataset = CelebaDatasetPath(csv_path=csv_path,
                                    img_dir=IMG_DIR,
                                    data=data,
                                    transform=validation_transform)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=False)

    return val_loader


def get_test_dataloader_path(mean, std, data='Gender', batch_size=64, num_workers=8, shuffle=False):
    DATA_SPLIT = '70' # 80 90

    csv_path = settings.CELEBA_CSV_PATH + 'test_' + data.lower() + '_hq_' + DATA_SPLIT + '.csv'

    test_transform = transforms.Compose(get_compose(mean, std))

    test_dataset = CelebaDatasetPath(   csv_path=csv_path,
                                        img_dir=IMG_DIR,
                                        data=data,
                                        transform=test_transform)

    test_loader = DataLoader(   dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=True)

    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
    
    TOTAL_BAR_LENGTH = 65.


'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def getdevicename():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    ret = device_name.split('-')[0].lower()
    # print("device_name: ", ret)
    return ret


def get_dataloader(dataset_type, root_dir, is_train, batch_size, workers, resolution=32, classes=1000, preprocessing=None, shuffle=True, **kwargs):

    print("root_dir: ", root_dir)
    print("is_train: ", is_train)
    print("batch_size: ", batch_size)
    print("resolution: ", resolution)

    normalize = False
    if not preprocessing == None:
        normalize = True
        if normalize or is_train:
            normalize_transfrom = transforms.Normalize(mean=preprocessing['mean'], std=preprocessing['std'])

    # import pdb; pdb.set_trace()

    transformations = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transfrom,
    ] if is_train else [ 
            transforms.ToTensor(), normalize_transfrom
        ] if normalize else [
            transforms.ToTensor()
        ] 
    
    print("normalize: ", normalize)
    print("transformations: ", transformations)

    trans = transforms.Compose(transformations)
    dataset = smallimagenet.SmallImagenet(root=root_dir, size=resolution, train=is_train, transform=trans,
                                          classes=range(classes)) if dataset_type == "SmallImageNet" else tinyimagenet.TinyImageNet(
                                          root=root_dir, train=is_train, transform=trans)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)

    # pdb.set_trace()
    
    return loader


# def check_arg_net_normalization(args):
#     try:
#         x = args.net_normalization
#     except AttributeError:
#         x = False    
#     return x


def get_normalization(args):

    mean = None
    std  = None
    if args.net == 'cif10' or args.net == 'cif10vgg' or args.net == 'cif100':
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
    elif args.net == 'cif100vgg':
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]
    elif args.net == 'imagenet32' or args.net == 'imagenet64' or args.net == 'imagenet128':
        mean = [0.4810, 0.4574, 0.4078]
        std  = [0.2146, 0.2104, 0.2138]
    elif args.net == 'imagenet':        
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    elif args.net == 'celebaHQ32' or args.net == 'celebaHQ64' or args.net == 'celebaHQ128' or args.net == 'celebaHQ256':
            if args.img_size == 32:
                mean = [0.36015135049819946, 0.21252931654453278, 0.11682419478893280]
                std  = [0.24773411452770233, 0.20017878711223602, 0.17963241040706635]
            elif args.img_size == 64:
                mean = [0.36108517646789550, 0.2132178544998169,  0.11681009083986282]
                std  = [0.24751928448677063, 0.19908452033996582, 0.17821228504180908]
            elif args.img_size == 128:
                mean = [0.36175119876861570, 0.21352902054786682, 0.11670646071434021]
                std  = [0.24808333814144135, 0.19945485889911652, 0.17840421199798584]
            elif args.img_size == 256:
                mean = [0.36185416579246520, 0.21353766322135925, 0.11669121682643890]
                std  = [0.24811546504497528, 0.19950547814369202, 0.17830605804920197]
            elif args.img_size == 512:
                mean = [0.36201700568199160, 0.21373045444488525, 0.11688751727342606]
                std  = [0.24830812215805054, 0.19977152347564697, 0.17850320041179657]
            elif args.img_size == 1024:
                mean = [0.36192145943641660, 0.21378295123577118, 0.11703434586524963]
                std =  [0.24930983781814575, 0.20103301107883453, 0.17989256978034973]
    else:
        get_debug_info(msg="Err: normalization not found!")

    return mean, std


def normalize_images(images, args):
    mean, std = get_normalization(args)
    images[:,0,:,:] = (images[:,0,:,:] - mean[0]) / std[0]
    images[:,1,:,:] = (images[:,1,:,:] - mean[1]) / std[1]
    images[:,2,:,:] = (images[:,2,:,:] - mean[2]) / std[2]
    return images


def create_new_state_dict(checkpoint, keyword='net'):

    new_state_dict = OrderedDict()
    for k, v in checkpoint[keyword].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def get_model_info(args):
    # imagenet32 imagenet64 imagenet128 
    # celebaHQ32 celebaHQ64 celebaHQ128
    net = 'wrn'
    depth = 28
    widen_factor = 10

    if args.net == 'imagenet':
        depth = 50
        widen_factor = 2
    elif args.net == 'cif10vgg' or args.net == 'cif100vgg':
        net = 'vgg'
        depth = 16
        widen_factor = 0

    return net, depth, widen_factor


def check_args(args, logger):
    if args.net_normalization:
        if not args.attack == 'std' or not args.attack == 'ind':
            logger.log("Warning: Net normalization must be switched off!")
            args.net_normalization = False
            logger.log("Warning: Net normalization is switched off now!")
    return args


def load_model(args):

    model = None
    preprocessing = None

    # Params for WideResNet
    depth = 28
    widen_factor = 10

    # Check out Normalization
    mean, std     = get_normalization(args)
    preprocessing = dict(mean=mean, std=std, axis=-3)

    # Check if the net should be normalized as for AutoAttack!
    if args.net_normalization:
        net_normalization = preprocessing
        if mean == None:
             net_normalization = {}
        get_debug_info(msg="Info: net normalization!")
    else:
        net_normalization = {}
        get_debug_info(msg="Info: No net normalization!")

    if args.net == 'cif10':
        model = WideResNet(num_classes=settings.MAX_CLASSES_CIF10, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)
        
        ckpt = torch.load(settings.CIF10_CKPT)
        new_state_dict = create_new_state_dict(ckpt)
        model.load_state_dict(new_state_dict)

    elif args.net == 'cif10vgg':
        depth = 16
        widen_factor = 0
        model = VGG('VGG16', preprocessing=net_normalization)
        ckpt = torch.load(settings.CIF10VGG_CKPT)
        new_state_dict = create_new_state_dict(ckpt)
        model.load_state_dict(new_state_dict)


    elif args.net == 'cif100vgg':

        depth = 16
        widen_factor = 0
        model = vgg16_bn( preprocessing=net_normalization ) 
        ckpt = torch.load(settings.CIF100VGG_CKPT)
        model.load_state_dict(ckpt)
        

    elif args.net == 'cif100':
        model = WideResNet(num_classes=settings.MAX_CLASSES_CIF100, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)
        ckpt = torch.load(settings.CIF100_CKPT)
        new_state_dict = create_new_state_dict(ckpt, keyword='state_dict')
        model.load_state_dict(new_state_dict)


    elif args.net == 'imagenet':
        model = wide_resnet50_2(pretrained=True, preprocessing=net_normalization)


    elif args.net == 'imagenet32':
        model = WideResNet(num_classes=args.num_classes, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)
        if args.num_classes == settings.MAX_CLASSES_IMAGENET:
            ckpt = torch.load(settings.IMAGENET32_CKPT_1000)
        elif  args.num_classes == 250: 
            ckpt = torch.load(settings.IMAGENET32_CKPT_250)
        elif  args.num_classes == 100:
            ckpt = torch.load(settings.IMAGENET32_CKPT_100)
        elif  args.num_classes == 75: 
            ckpt = torch.load(settings.IMAGENET32_CKPT_75)
        elif  args.num_classes == 50: 
            ckpt = torch.load(settings.IMAGENET32_CKPT_50)
        elif  args.num_classes == 25: 
            ckpt = torch.load(settings.IMAGENET32_CKPT_25)
        elif  args.num_classes == 10: 
            ckpt = torch.load(settings.IMAGENET32_CKPT_10)

        new_state_dict = create_new_state_dict(ckpt, keyword='state_dict')
        model.load_state_dict(new_state_dict)


    elif args.net == 'imagenet64':
        model = WideResNet(num_classes=settings.MAX_CLASSES_IMAGENET, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)
        ckpt = torch.load(settings.IMAGENET64_CKPT_1000)
        new_state_dict = create_new_state_dict(ckpt, keyword='state_dict')

        model.load_state_dict(new_state_dict)


    elif args.net == 'imagenet128':
        model = WideResNet(num_classes=settings.MAX_CLASSES_IMAGENET, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)
        ckpt = torch.load(settings.IMAGENET128_CKPT_1000)
        new_state_dict = create_new_state_dict(ckpt, keyword='state_dict')
        
        model.load_state_dict(new_state_dict)


    elif args.net == 'celebaHQ32':
        model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)

        chpt_dict = settings.CELEBAHQ32_CKPT_2
        if args.num_classes == 4:
            chpt_dict = settings.CELEBAHQ32_CKPT_4
        ckpt = torch.load(chpt_dict)
        get_debug_info(msg=chpt_dict)

        model.load_state_dict(ckpt)


    elif args.net == 'celebaHQ64':
        model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)

        chpt_dict = settings.CELEBAHQ64_CKPT_2
        if args.num_classes == 4:
            chpt_dict = settings.CELEBAHQ64_CKPT_4
        ckpt = torch.load(chpt_dict)
        get_debug_info(msg=chpt_dict)

        model.load_state_dict(ckpt)


    elif args.net == 'celebaHQ128':
        model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)

        chpt_dict = settings.CELEBAHQ128_CKPT_2
        if args.num_classes == 4:
            chpt_dict = settings.CELEBAHQ128_CKPT_4
        ckpt = torch.load(chpt_dict)
        get_debug_info(msg=chpt_dict)

        model.load_state_dict(ckpt)


    elif args.net == 'celebaHQ256':
        model = WideResNet(num_classes=settings.MAX_CLASSES_CELEBAHQ, block=WideBasic, depth=depth, widen_factor=widen_factor, preprocessing=net_normalization)

        # chpt_dict = settings.CELEBAHQ256_CKPT_2
        if args.num_classes == 4:
            chpt_dict = settings.CELEBAHQ256_CKPT_4
        ckpt = torch.load(chpt_dict)
        get_debug_info(msg=chpt_dict)

        model.load_state_dict(ckpt)


    if model == None:
        get_debug_info(msg="Err: Model is None!")
        assert True

    return model, preprocessing


def make_dir(save_dir='./data/'):
    existed = os.path.isdir(save_dir)
    if not existed:
        os.makedirs(save_dir)
    return existed


def create_save_dir_path(save_dir, args, filename='images'):
    
    save_dir_img = os.path.join(save_dir, filename )
    save_dir_adv = os.path.join(save_dir, filename + '_adv')
    
    get_debug_info(msg=save_dir_img)
    get_debug_info(msg=save_dir_adv)

    return save_dir_img, save_dir_adv



def load_train_set(args, preprocessing=None):
     
    args.batch_size = 128
    
    return load_test_set(args, preprocessing=preprocessing, IS_TRAIN=True)


def load_test_set(args, preprocessing=None, IS_TRAIN=False):
    num_workers = 4;  shuffle = True;  download = True; 

    normalization = []
    if not preprocessing == None:
        normalization = [transforms.Normalize(mean=preprocessing['mean'], std=preprocessing['std'])]

    if args.net == 'cif10' or args.net == 'cif10vgg':
        
        transform_list = [transforms.ToTensor()] + normalization
        transform = transforms.Compose(transform_list)
        item = datasets.CIFAR10(root=settings.CIF10_PATH, train=IS_TRAIN, transform=transform, download=download)
        data_loader = torch.utils.data.DataLoader(item, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)
        

    elif args.net == 'cif100' or args.net == 'cif100vgg':

        transform_list = [transforms.ToTensor()] + normalization
        transform = transforms.Compose(transform_list)
        item = datasets.CIFAR100(root=settings.CIF100_PATH, train=IS_TRAIN, transform=transform, download=download)
        data_loader =  torch.utils.data.DataLoader(item, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)


    elif args.net == 'imagenet':

        transform_list = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()] + normalization
        transform = transforms.Compose(transform_list)

        dataset_dir = os.path.join(settings.IMAGENET_PATH, 'val') if IS_TRAIN else os.path.join(settings.IMAGENET_PATH, 'train')
        data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dir, transform), batch_size=args.batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=True)

        get_debug_info(msg="ImageNet is always shuffled!")

        data_loader


    elif args.net == 'imagenet32' or args.net == 'imagenet64' or args.net == 'imagenet128' or args.net == 'imagenet240':
        

        if args.img_size == 32:
            dataset_dir = os.path.join(settings.IMAGENET32_PATH, 'train_data') if IS_TRAIN else settings.IMAGENET32_PATH           
            data_loader = get_dataloader("SmallImageNet", dataset_dir, is_train=IS_TRAIN, batch_size=args.batch_size, workers=num_workers, 
                            resolution=args.img_size, classes=args.num_classes, preprocessing=preprocessing, shuffle=shuffle)
        elif args.img_size == 64:
            dataset_dir = os.path.join(settings.IMAGENET64_PATH, 'train_data') if IS_TRAIN else settings.IMAGENET64_PATH
            data_loader = get_dataloader("SmallImageNet", dataset_dir, is_train=IS_TRAIN, batch_size=args.batch_size, workers=num_workers, 
                            resolution=args.img_size, classes=settings.MAX_CLASSES_IMAGENET, preprocessing=preprocessing, shuffle=shuffle)
        elif args.img_size == 128: 
            # dataset_dir = os.path.join(settings.IMAGENET128_PATH, 'val/box') if IS_TRAIN else  os.path.join(settings.IMAGENET128_PATH, 'train/box')
            dataset_dir = os.path.join(settings.IMAGENET128_PATH, 'val_data/box') if IS_TRAIN else  os.path.join(settings.IMAGENET128_PATH, 'train_data/box')
            # normalize = transforms.Normalize(mean=[0.4810, 0.4574, 0.4078], std=[0.2146, 0.2104, 0.2138])

            transform_list = [transforms.ToTensor()] + normalization
            transform = transforms.Compose(transform_list)

            data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dir, transform), batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            get_debug_info(msg="Err: Only imagenet-32, 64, 128 implemented!")

    elif args.net == 'celebaHQ32' or args.net == 'celebaHQ64' or args.net == 'celebaHQ128' or args.net == 'celebaHQ256':

        classes = "Smiling"
        if args.num_classes == 4:
            classes = "Hair_Color"

        mean = None
        std  = None
        if not len(normalization) == 0:
            mean = preprocessing['mean']
            std  = preprocessing['std']

        img_dir = get_celeba_path(args)

        if IS_TRAIN:
            celebahq_training_loader = get_training_dataloader(
                mean=mean,
                std=std,
                data=classes,
                img_dir=img_dir,
                num_workers=num_workers,
                batch_size=args.batch_size,
                shuffle=shuffle
            )   
        else:
            data_loader = get_validation_dataloader (
                mean=mean, #settings.CELEABAHQ_TRAIN_MEAN,
                std=std,   #settings.CELEABAHQ_TRAIN_STD,
                data=classes,
                img_dir=img_dir,
                num_workers=num_workers,
                batch_size=args.batch_size,
                shuffle=shuffle
            )

    else:
        print("ERR: Parameter 'args.net' is not known!")

    return data_loader


def create_output_filename(args):
    appendix = get_appendix(args.num_classes, settings.MAX_CLASSES_IMAGENET)
    net, depth, widen_factor = get_model_info(args)
    output_filename = net + '_' + str(depth) + '_' + str(widen_factor) + appendix 

    return output_filename


def create_dir_clean_data(args, root='./data/clean_data/', wait_input=False):
    output_filename = create_output_filename(args)

    output_path_dir = os.path.join(root, 'run_' + str(args.run_nr), args.net, output_filename)

    existed = make_dir(output_path_dir)
    get_debug_info(msg='Info: clean_data_path: ' + output_path_dir + ', existed: ' + str(existed))
    
    if existed and wait_input:
        input(settings.WARN_DIR_EXISTS)

    return output_path_dir


def epsilon_to_float(epsilon):
    try:
        num, denom = epsilon.split('/')
        num = float(num)
        denom = float(denom)
        return num / denom
    except ValueError:
        get_debug_info(msg='Info: clean_data_path: ' + output_path_dir)
        return -1


def epsilon_to_string(epsilon):
    """
        Ex.: 0.5/255 to 05_255
        Ex.: 8./255. to 8_255        
    """
    if epsilon.isdigit():
        get_debug_info("ERR: Input must be string!")
        assert True
 
    return epsilon.replace("/", "_").replace(".", "")


def check_epsilon(args):
    epsilon = ''
    if args.attack == 'std' or args.attack == 'ind':
        epsilon = epsilon_to_string(args.eps)
    
    return epsilon


def check_layer_nr(args):
    layer_nr = ''
    if args.detector == 'LayerPFS' or args.detector == 'LayerMFS':   
        if not args.nr == -1:
                layer_nr = 'layer_' + str(args.nr)
        else:
            get_debug_info(msg="Only WhiteBox Methods possible!")

    return layer_nr
    

def create_dir_attacks(args, root='./data/attacks/', wait_input=False):
    output_filename = create_output_filename(args)
    epsilon = check_epsilon(args)
    output_path_dir = os.path.join(root, 'run_' + str(args.run_nr), args.net, output_filename, args.attack, epsilon)
    existed = make_dir(output_path_dir)
    
    get_debug_info(msg='Info: attack data path: ' + output_path_dir + ', existed: ' + str(existed))

    if existed and wait_input:
        input(settings.WARN_DIR_EXISTS)

    return output_path_dir


def create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False):
    output_filename = create_output_filename(args)
    epsilon = check_epsilon(args)
    layer_nr = check_layer_nr(args)
    output_path_dir = os.path.join(root, 'run_' + str(args.run_nr), args.net, output_filename, args.attack, epsilon, args.detector, layer_nr)

    existed = make_dir(output_path_dir)
    get_debug_info(msg='Info: Extracted Characteristics data path: ' + output_path_dir + ', existed: ' + str(existed))

    if existed and wait_input:
        input(settings.WARN_DIR_EXISTS)

    return output_path_dir


def create_dir_detection(args, root='./data/detection/', wait_input=False):
    output_filename = create_output_filename(args)
    epsilon = check_epsilon(args)
    layer_nr = check_layer_nr(args)
    
    output_path_dir = os.path.join(root, 'run_' + str(args.run_nr), args.net, output_filename, args.attack, epsilon, args.detector, layer_nr, args.clf)


    existed = make_dir(output_path_dir)
    get_debug_info(msg='Info: Detection data path: ' + output_path_dir + ', existed: ' + str(existed))

    if existed and wait_input:
        input(settings.WARN_DIR_EXISTS)

    return output_path_dir


def log_header(logger, args, output_path_dir, sys):
    logger.log('===============================================================================')
    logger.log('FILENAME: ' + str(sys.argv[0]) )
    logger.log('DATE: ' + str(datetime.datetime.now()))
    logger.log('GPU:  ' + getdevicename())
    logger.log('ARGS: ' + str(args.__dict__))
    logger.log('OUTPUT_PATH_DIR: ' + output_path_dir)



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
           
    
    print("TP: ", TP)
    print("FP: ", FP)
    print("TN: ", TN)
    print("FN: ", FN)

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/(TN+FP) 
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # # Negative predictive value
    # NPV = TN/(TN+FN)
    # # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # # False negative rate
    # FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # print("TPR: ", TPR)
    # print("TNR: ", TNR)
    # print("PPV: ", PPV)
    # print("NPV: ", NPV)

    # print("TNR: ", FPR)
    # print("PPV: ", FNR)
    # print("NPV: ", FDR)

    print("ACC: ", ACC)

    return (TP, FP, TN, FN)
