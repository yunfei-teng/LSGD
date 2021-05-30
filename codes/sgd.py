from __future__ import print_function
import os
import time
import datetime

import math
import argparse
from models import EASGD_7_layer, resnet20, vgg16
from optimizers import LARS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
import torch.multiprocessing as mp
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        cur_steps = len(train_loader)* (epoch-1) + batch_idx
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if args.optimizer == 'SGD':
            optimizer.step()
        elif args.optimizer == 'LARS':
            if epoch == 0:
                optimizer.step(0, 1)
            else:
                optimizer.step(cur_steps, args.remaining_steps)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, 100*(1-correct/args.batch_size)

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, 100*(1-correct/len(test_loader.dataset))

def main_worker(gpu, args):
    # load data sets
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    
    kwargs = {'num_workers': 0, 'pin_memory': True} # num_workers is the same as LSGD paper
    transformTR = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(28),
                    transforms.ToTensor(),
                    normalize,
                ])
    transformTE = transforms.Compose([
                    transforms.CenterCrop(28),
                    transforms.ToTensor(),
                    normalize,
                ])
    train_dataset = datasets.CIFAR10(args.datadir, train=True, download=True, transform=transformTR)
    test_dataset = datasets.CIFAR10(args.datadir, train=False, transform=transformTE)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False, **kwargs)
    
    # define model and optimizer
    torch.cuda.set_device(gpu) # depreciated now
    device = torch.device('cuda:%d'%gpu)
    if args.model == 'vgg16':
        model = vgg16().to('cuda:0')
    elif args.model == 'resnet20':
        model = resnet20().to('cuda:0')
    elif args.model == 'cnn7':
        model = EASGD_7_layer(10).to('cuda:0')
    else:
        raise ValueError("wrong model name!")  
    model = model.cuda(gpu)
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    elif args.optimizer == 'LARS':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    else:
        raise ValueError('Could not find optimizer name')

    # initial traininig
    exp_dir = os.path.join(args.checkpoints_dir, args.exp_name)
    file_name = os.path.join(exp_dir, 'messages.csv')
    csv_fields =  "rank,epoch,raw_time,train_loss,train_error,test_loss,test_error"
    message_file = open(file_name, "w")
    message_file.write(csv_fields)
    message_file.close()

    total_time = 0
    strat_time = time.time()
    train_loss, train_error = train(args, model, device, train_loader, optimizer, 0)
    total_time += time.time() - strat_time
    test_loss, test_error = test(args, model, device, test_loader)

    message_file = open(file_name, "a")
    _text = "\n%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f"%(gpu, 0, total_time, train_loss, train_error, test_loss, test_error)
    for i in range(args.num_gpus):
        message_file.write(_text)
    message_file.close()

    expected_epochs = math.ceil((args.hours* 3600 + args.minutes* 60) / total_time)
    expected_epochs = args.epochs if args.use_epochs else expected_epochs
    print('Each epoch takes [%.1fs] and the expected number of epochs is [%d]'%(total_time, expected_epochs))

    # main training
    for epoch in range(1, expected_epochs):
        strat_time = time.time()
        train_loss, train_error = train(args, model, device, train_loader, optimizer, epoch)
        if args.optimizer == 'SGD':
            if args.model == 'vgg16' and total_time >= 500:
                for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr* 0.1       
            elif args.model == 'resnet20' and total_time >= 500:
                for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr* 0.1
        elif args.optimizer == 'LARS':
            pass
        total_time += time.time() - strat_time
        
        if epoch % (128 // args.batch_size) == 0:
            test_loss, test_error = test(args, model, device, test_loader)
            message_file = open(file_name, "a")
            _text = "\n%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f"%(gpu, epoch, total_time, train_loss, train_error, test_loss, test_error)
            for i in range(args.num_gpus):
                message_file.write(_text)
            message_file.close()
            print('(epoch: %3d) [%.3f s] LOSS: %.3f ERROR: %.3f%%'%(epoch, total_time, test_loss, test_error))

    if args.save_model:
        torch.save(model.state_dict(),"%s.pt"%args.exp_name)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Data Parallel Training')    
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--exp_name', type = str, default='LSGD')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

    parser.add_argument('--cur_group', default=0, type=int)
    parser.add_argument('--num_groups', default=1, type=int)
    parser.add_argument('--datadir', type = str, default='./dataset', help='data')
    parser.add_argument('--dataset', type = str, default='cifar10', help='dataset')
    parser.add_argument('--model', type = str, default='resnet20')

    parser.add_argument('--use_epochs', action='store_true') 
    parser.add_argument('--save_model', action='store_true') 
    parser.add_argument('--epochs', type = int, default=30)
    parser.add_argument('--iters',  type = int, default=550)
    parser.add_argument('--minutes', default=0.2, type=float)
    parser.add_argument('--hours'  , default=0.0, type=float)
    parser.add_argument('--check_point_epochs', default=30, type=int)

    parser.add_argument('--batch_size', type = int, default=128)
    parser.add_argument('--lr', type = float, default=0.01)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_iter_decay', type = int, default=-1)
    parser.add_argument('--lr_time_decay', type = float, default=-1)
    parser.add_argument('--lr_decay_coef', type = float, default= 1)
    parser.add_argument('--optimizer', type = str, default= 'SGD')

    parser.add_argument('--wd', type = float, default=1e-4)
    parser.add_argument('--mom', type = float, default=0.0)
    parser.add_argument('--no_eval', action='store_true')

    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--cpu_seed', type = int, default=24)
    parser.add_argument('--gpu_seed', type = int, default=32)

    parser.add_argument('--distributed', default=True, action='store_true')
    parser.add_argument('--num_gpus', default=4, type=int)
    parser.add_argument('--dist_ip', default="216.165.115.98", type=str)
    parser.add_argument('--dist_port', default="2432", type=str)
    parser.add_argument('--dist_backend', default="nccl", type=str)
    parser.add_argument('--random_sampler', action='store_true')
    parser.add_argument('--suffix', default='', type=str) # additional parameters

    args = parser.parse_args()
    try:
        os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name))
    except:
        pass
    message = str(datetime.datetime.now())
    message += '\n----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    
    # save to the disk
    expr_dir = os.path.join(args.checkpoints_dir, args.exp_name)
    file_name = os.path.join(expr_dir, 'args.txt')
    with open(file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
    main_worker(args.gpu, args)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Distributed training takes [%.1fs]'%(time.time() - start_time))