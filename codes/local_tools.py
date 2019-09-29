import os
import time

import torch
import torch.distributed as dist
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as udata
import torchvision.models
from models import EASGD_7_layer

# ---- Usefull Utilities ----
def mkdir(path):
    '''create a single empty directory if it didn't exist
    Parameters: path (str) -- a single directory path'''
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    '''create empty directories if they don't exist
    Parameters: paths (str list) -- a list of directory paths'''
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def tensor2text(tensor):
    '''convert tensor's contents to readable texts'''
    tensor_text = ""
    tensor_type = tensor.type()
    int_tensor_dict = ['torch.cuda.CharTensor', 'torch.cuda.IntTensor']
    float_tensor_dict = ['torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor']
    for var in tensor:
        if tensor_type in float_tensor_dict:
            tensor_text += "%.4f "%var.item()
        elif tensor_type in int_tensor_dict:
            tensor_text += "%d "%var.item()
        else:
            raise ValueError(var, "Undefined tensor type to print")
    return '[' + tensor_text + ']'

# ---- PyTorch Utilities ----
def get_data_loader(args):
    '''return train and test dataloader'''
    # [normalization]
    normalize = None
    if args.dataset.lower() == 'cifar' or args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'fake_cifar':
        args.dataset = 'CIFAR10'
        # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    
    # [transformation]
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

    # [dataset]
    train_dataset = dsets.__dict__[args.dataset](root=args.datadir,
                                                 train=True,
                                                 transform=transformTR,
                                                 download=True)

    test_dataset = dsets.__dict__[args.dataset](root=args.datadir,
                                                train=False,
                                                transform=transformTE,
                                                download=True)

    # [data loader] arguments
    # when 'num_loader' > 0 the loading hangs before each epoch which seems to be a strange bug
    # 'num_loaders = 0' means loading in main process and it works fine
    # raised question by me: https://discuss.pytorch.org/t/training-hangs-for-a-second-at-the-beginning-of-each-epoch/39304
    num_loaders = 0   
    is_pin_mem = True # This will be forced to be True in distributed training
    train_sampler = None
    train_sampler = udata.distributed.DistributedSampler(train_dataset, args.world_size, dist.get_rank())

    # [data loader] set up
    if args.random_sampler:
        train_loader = udata.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=num_loaders,
                                        shuffle=True,
                                        pin_memory=is_pin_mem,
                                        drop_last=True)
    else:
        train_loader = udata.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=num_loaders,
                                        shuffle=False,
                                        pin_memory=is_pin_mem,
                                        sampler=train_sampler,
                                        drop_last=True)
    
    test_loader  = udata.DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=num_loaders,
                                    shuffle=False,
                                    pin_memory=is_pin_mem)

    return train_sampler, train_dataset, train_loader, test_dataset, test_loader

def get_model(args):
    '''return pre-*defined* model'''
    if args.model == 'cnn7':
        return EASGD_7_layer(10)
    else:
        raise ValueError("wrong model name!")    

class AverageMeter(object):
    '''computes and stores the average and current value'''
    def __init__(self, running_size=10):
        self.running_avg = 0
        self.running_size = running_size
        self.running_vec = torch.zeros(running_size)
        self.reset()

    def reset(self):
        self.running_index = 0
        self.running_vec.zero_()
        self.running_avg = 0

    def update(self, val, n=1):
        if self.running_index >= self.running_size:
            self.running_index = 0
        self.running_vec[self.running_index] = val
        self.running_index += 1
        self.running_avg = self.running_vec.mean().item()