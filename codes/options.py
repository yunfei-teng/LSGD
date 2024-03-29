import argparse
import os
import datetime

import torch
import local_tools as utils
import warnings
import torch.backends.cudnn as cudnn

class Options():
    ''' This class defines argsions
        adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options
    '''
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        ''' set up arguments '''
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--exp_name', type = str, default='LSGD')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

        parser.add_argument('--cur_group', default=0, type=int)
        parser.add_argument('--num_groups', default=1, type=int)
        parser.add_argument('--datadir', type = str, default='./dataset', help='data')
        parser.add_argument('--dataset', type = str, default='cifar10', help='dataset')
        parser.add_argument('--model', type = str, default='resnet20')

        parser.add_argument('--use_epochs', action='store_true') 
        parser.add_argument('--epochs', type = int, default=5)
        parser.add_argument('--iters',  type = int, default=550)
        parser.add_argument('--minutes', default=0.2, type=float)
        parser.add_argument('--hours'  , default=0.0, type=float)
        parser.add_argument('--check_point_epochs', default=30, type=int)

        parser.add_argument('--batch_size', type = int, default=128)
        parser.add_argument('--lr', type = float, default=0.01)
        parser.add_argument('--lr_lin', action='store_true')
        parser.add_argument('--lr_pow', action='store_true')

        parser.add_argument('--lr_decay', action='store_true')
        parser.add_argument('--lr_iter_decay', type = int, default=-1)
        parser.add_argument('--lr_time_decay', type = float, default=-1)
        parser.add_argument('--lr_decay_coef', type = float, default= 1)

        parser.add_argument('--wd', type = float, default=1e-4)
        parser.add_argument('--mom', type = float, default=0.0)
        parser.add_argument('--no_eval', action='store_true')

        parser.add_argument('--cpu_seed', type = int, default=24)
        parser.add_argument('--gpu_seed', type = int, default=32)

        parser.add_argument('--distributed', default=True, action='store_true')
        parser.add_argument('--dist_optimizer', type = str, default="lsgd")

        parser.add_argument('--c1', type = float, default=0.1, help='local pulling strength')
        parser.add_argument('--c2', type = float, default=0.1, help='global pulling strength')
        parser.add_argument('--p1', type = float, default=0.1, help='PPA local pulling strength')
        parser.add_argument('--p2', type = float, default=0.1, help='PPA global PPA pulling strength')
        parser.add_argument('--no_prox', default=False, action='store_true')

        parser.add_argument('--alpha', type = float, default=0.9, help='exponential moving averaging')
        parser.add_argument('--gamma', type = float, default=0.9, help='x to mu pulling strength')
        parser.add_argument('--etagamma', type = float, default=0.99, help='SGLD step size')
        parser.add_argument('--weight_averaging', default=False, action='store_true')

        parser.add_argument('--num_gpus', default=4, type=int)
        parser.add_argument('--dist_ip', default="216.165.115.98", type=str)
        parser.add_argument('--dist_port', default="2432", type=str)
        parser.add_argument('--dist_backend', default="nccl", type=str)
        parser.add_argument('--random_sampler', action='store_true')

        parser.add_argument('--l_comm', default=256, type=int)
        parser.add_argument('--g_comm', default=256, type=int)
        parser.add_argument('--avg_size', default= 10, type=int)
        
        parser.add_argument('--landscape',  action='store_true')
        parser.add_argument('--suffix', default='', type=str) # additional parameters
        
        self.initialized = True
        return parser

    def get_argsions(self):
        ''' get argsions from parser '''
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description='Asyncrhonous toy example')
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_argsions(self, args):
        ''' Print and save argsions
            It will print both current argsions and default values(if different).
            It will save argsions into a text file / [checkpoints_dir] / args.txt
        ''' 
        message = str(datetime.datetime.now())
        message += '\n----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(args.checkpoints_dir, args.exp_name)
        utils.mkdirs(expr_dir) # first remove existing directory and then create a new one
        file_name = os.path.join(expr_dir, 'args.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
    
    def parse(self):
        ''' Parse our argsions, create checkpoints directory suffix, and set up gpu device. '''
        args = self.get_argsions()

        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.exp_name = args.exp_name + suffix
        
        # reset options
        if args.g_comm < args.l_comm:
            raise ValueError('global communication period is shorter than local ones')
        elif args.num_groups == 1 or args.g_comm == args.l_comm:
            args.is_lcomm = False
            args.l_comm = args.g_comm
            print("[ONLY global leader is activated!]")
        else:
            args.is_lcomm = True

        if args.dataset.lower() == 'cifar' or args.dataset.lower() == 'cifar10':
            args.dataset = 'CIFAR10'
        elif args.dataset.lower() == 'imagenet':
            args.dataset = 'ImageNet'
        else:
            raise ValueError("wrong dataset name!")

        assert not (args.lr_lin and args.lr_pow)
        assert args.dist_optimizer in ['EASGD', 'LSGD'], 'No such distributed optimizer supported'
        if not args.dist_optimizer == 'LSGD':
            args.no_prox = True

        self.print_argsions(args)
        self.args = args
        return self.args