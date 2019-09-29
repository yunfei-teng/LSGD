import torch
import torch.distributed as dist
import models

from local_tools import *
from dist_tools import *

class WorkerBase():
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This is the base class for master and worker in distributed training '''
        # initialization
        assert torch.cuda.is_available(), "distributed training requires GPU on all machines"
        device = torch.device('cuda:%d'%(cur_worker%torch.cuda.device_count()))

        # cuda randomness options
        '''
        torch.manual_seed(args.cpu_seed)
        torch.cuda.set_device(cur_worker%torch.cuda.device_count()) # depreciated now!
        torch.cuda.manual_seed(args.gpu_seed)
        cudnn.deterministic = True # This will slow down the program
        '''
        
        # assign class variables
        self.args = args
        self.device = device
        self.device_name = torch.cuda.get_device_name(device)
        self.cur_worker = cur_worker
        self.shared_tensor = shared_tensor
        self.shared_lock = shared_lock
        self.shared_queue_r = shared_queue_r
        self.shared_queue_a = shared_queue_a
        
        # training variables and signals
        self.my_gpu  = cur_worker
        self.my_rank = set_global_rank(args, cur_worker)
        self.world_best_worker = 0

        # create model and gradients
        self.model = get_model(args)
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params.data)
        self.model.to(device)
        self.model_center = get_model(args)
        self.model_center.to(device)
        self.model_center.eval()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        if self.my_rank == 0: 
            print('-- The model has following buffers --')
            for name, buf in self.model.named_buffers():
                print(name)
            print()
            print('-- The model has following params --')
            for name, buf in self.model.named_parameters():
                print(name)