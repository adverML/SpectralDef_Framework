#!/usr/bin/env python3
""" Generate Clean Data

author Peter Lorenz
"""

#this script extracts the correctly classified images
print('INFO: Load modules...')
import pdb
import os, sys
import json
from conf import settings
import argparse
import datetime
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets

from utils import *

from models.vgg_cif10 import VGG
from models.wideresidual import WideResNet, WideBasic

from datasets import smallimagenet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",         default=1,    type=int, help="Which run should be taken?")

    parser.add_argument("--net",            default='cif10',        help=settings.HELP_NET)
    parser.add_argument("--img_size",       default=32,   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",    default=1000, type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--batch_size",     default=1   , type=int, help=settings.HELP_BATCH_SIZE)
    parser.add_argument("--wanted_samples", default=4000, type=int, help=settings.HELP_WANTED_SAMPLES)

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)
    
    args = parser.parse_args()


    if not args.batch_size == 1:
        get_debug_info(msg='Err: Batch size must be always 1!')
        assert True

    output_path_dir = create_dir_clean_data(args, root='./data/clean_data/')

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys)

    logger.log('INFO: Load model...')

    model, preprocessing = load_model(args)
    model.cuda()
    model.eval()
    

    logger.log('INFO: Load dataset...')
    test_loader  = load_test_set(args, preprocessing=None) # Data Normalizations; No Net Normaliztion

    clean_dataset = []
    correct = 0
    total = 0
    i = 0

    logger.log('INFO: Classify images...')

    for images, labels in test_loader:
        if i == 0:
            logger.log( "INFO: tensor size: " + str(images.size()) )

        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
        if (predicted == labels):
            # clean_dataset.append(data)
            clean_dataset.append([images, labels])

        i = i + 1
        if i % 500 == 0:
            acc = (args.wanted_samples, i, 100 * correct / total)
            logger.log('INFO: Accuracy of the network on the %d test images: %d, %d %%' % acc)

        if len(clean_dataset) >= args.wanted_samples:
            break
    
    # pdb.set_trace()    
    logger.log("INFO: initial accuracy: {:.2f}".format(acc[-1]))
    logger.log("INFO: output_path_dir: " + output_path_dir + ", len(clean_dataset) " + str(len(clean_dataset)) )

    torch.save(clean_dataset, output_path_dir + os.sep + 'clean_data', pickle_protocol=4)
    logger.log('INFO: Done extracting and saving correctly classified images!')