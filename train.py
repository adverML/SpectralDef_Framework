# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
from datetime import timedelta
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_validation_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from utils import getdevicename

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_name = ''
if device == 'cuda':
    torch.manual_seed(10)
    gpu_name = getdevicename()


def train(epoch, total_time):
    start = time.time()
    net.train()
    train_correct = 0.0
    for batch_index, (images, labels) in enumerate(celebahq_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        # _, preds_train = outputs.max(1)

        # train_correct += preds_train.eq(labels).sum() 
        # loss.retain_grad()
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(celebahq_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        trained_samples = batch_index * args.b + len(images)
        if batch_index % len(celebahq_training_loader) * 0.5 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                1,  #train_correct.float() / len(celebahq_training_loader.dataset) ,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=trained_samples,
                total_samples=len(celebahq_training_loader.dataset)
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()
    diff = finish - start
    total_time = total_time + diff

    print('epoch {} training time consumed: {:.2f}s, Elapsed time: {}'.format(epoch, diff, str(timedelta(seconds=total_time))))

    return total_time


@torch.no_grad()
def eval_training(epoch=0, total_time=0, tb=True):
    
    # train_loss = 0.0 # cost function error
    # train_correct = 0.0
    # for (images, labels) in celebahq_training_loader:
    #     if args.gpu:
    #         images = images.cuda()
    #         labels = labels.cuda()

    #     outputs = net(images)
    #     loss = loss_function(outputs, labels)

    #     train_loss += loss.item()
    #     _, preds_train = outputs.max(1)
    #     train_correct += preds_train.eq(labels).sum()

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    # for (images, labels) in celebahq_validation_loader:
    for (images, labels) in celebahq_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
#     if args.gpu:
#         print('GPU INFO.....')
#         print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
#     val_acc = compute_accuracy(net, celebahq_validation_loader, args)
#     train_acc = compute_accuracy(net, celebahq_train_loader, args)
    
#     print('Val set: Epoch: {}, Loss {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}, Time consumed:{:.2f}s'.format(
#         epoch,
#         loss_function(outputs, labels),
#         train_acc,
#         val_acc,
#         finish - start
#     ))

    total_time = total_time + (finish - start)
    print('Val set: Epoch: {}, Average loss: {:.4f}, Acc: {:.4f}, Time consumed:{:.2f}s, Elapsed time:{}'.format(
        epoch,
        test_loss / len(celebahq_validation_loader.dataset),
        correct.float() / len(celebahq_validation_loader.dataset),
        finish - start,
        str(timedelta(seconds=total_time))
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Val/Average loss', test_loss / len(celebahq_validation_loader.dataset), epoch)
        writer.add_scalar('Val/Accuracy', correct.float() / len(celebahq_validation_loader.dataset), epoch)

    return correct.float() / len(celebahq_validation_loader.dataset), total_time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int,    default=64 , help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1  , help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-parallel', action='store_true', default=False, help='train on multiple GPUs training')
    args = parser.parse_args()
    
    print(settings.IMAGE_SIZE, settings.DATA, ', NUM_CLASSES: ', settings.NUM_CLASSES, ', Workers: ', settings.NUM_WORKERS, ', Normalized: ', settings.NORMALIZED)
    print(args)

    net = get_network(args)

    mean, std = get_normalization(args)

    

    celebahq_training_loader = get_training_dataloader(
        mean=mean,
        std=std,
        data=settings.DATA,
        num_workers=settings.NUM_WORKERS,
        batch_size=args.b,
        shuffle=True
    )

    celebahq_validation_loader = get_validation_dataloader(
        mean=mean,
        std=std,
        data=settings.DATA,
        num_workers=settings.NUM_WORKERS,
        batch_size=args.b,
        shuffle=False
    )

    celebahq_test_loader = get_test_dataloader(
        mean=mean,
        std=std,
        data=settings.DATA,
        num_workers=settings.NUM_WORKERS,
        batch_size=args.b,
        shuffle=False
    )


    if settings.DATA == "Hair_Color":
        loss_function = nn.CrossEntropyLoss()
        # cw = torch.tensor([0.11810626, 0.14863034, 0.10996354, 0.62329985], dtype=torch.float32).cuda()
        # loss_function = nn.CrossEntropyLoss(weight=cw)
    elif settings.DATA == "Gender":
        cw = torch.tensor([0.65457143, 0.34542857], dtype=torch.float32).cuda()
        loss_function = nn.CrossEntropyLoss(weight=cw)
        # loss_function = nn.CrossEntropyLoss()
    elif  settings.DATA == "Smiling":
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(celebahq_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    if args.resume:
        checkpoint_path = 'checkpoint'
        # checkpoint_path = './checkpoint/resnet50/256x256_4_0.01_Smiling_Friday_14_May_2021_13h_10m_32s/'
        settings.CHECKPOINT_PATH = checkpoint_path

        if settings.IMAGE_SIZE == '256x256':
            recent_folder = '256x256_4_0.01_Smiling_Friday_14_May_2021_13h_10m_32s'
        elif settings.IMAGE_SIZE == '512x512': 
            recent_folder = '512x512_4_0.001_Smiling_Friday_14_May_2021_21h_51m_59s'

    #     # recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)

    #     pdb.set_trace()
    #     recent_folder = most_recent_folder( settings.CHECKPOINT_PATH, fmt=settings.DATE_FORMAT)

    #     if not recent_folder:
    #         raise Exception('no recent folder were found')
    #     # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH)
    #     checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        # import pdb; pdb.set_trace()
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.IMAGE_SIZE + '_' + str(args.b) + '_' + \
        str(args.lr) + '_' + settings.DATA + '_' + gpu_name + '_' + settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.IMAGE_SIZE + '_' + str(args.b) + '_' + str(args.lr) + '_' + \
            settings.DATA + '_' + gpu_name + '_' + settings.TIME_NOW))

    if not args.parallel:
        input_tensor = torch.nn.Parameter(torch.Tensor(1, 3, 32, 32))
        if args.gpu:
            input_tensor = input_tensor.cuda()
        writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    print("checkpoint_path: ", checkpoint_path)
    try:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    except OSError as err:
        print(err)

    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    total_time = 0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        total_time = train(epoch, total_time)
        acc, total_time = eval_training(epoch, total_time)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
