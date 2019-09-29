import time
import copy
import torch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from local_tools import *
from dist_tools import *
from worker_base import WorkerBase

class WorkerDistOptim(WorkerBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        '''class for distribured optimization'''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)

        # create groups
        if cur_worker % args.num_gpus == 0:
            print("\n=== Setting up distribution training environment ===")
            print("--[Group] Preparing to crete the group")
        dist.init_process_group(rank=self.my_rank, world_size=args.world_size, 
                                backend=args.dist_backend, init_method=args.dist_url_wrk)
        dist_print(args, "--[Done] Successfully initialized the groups!!\n")

        self.group_dict = {}
        for i in range(args.num_groups):
            _group = [j for j in range(self.args.cur_group* self.args.num_gpus, (self.args.cur_group+1)* self.args.num_gpus)]
            self.group_dict[i] = torch.distributed.new_group(_group)
        print('Create local groups in the group dictonary')
        
        # barrier test
        dist_print(args, "--[Barrier] Now entering barrier test!!")
        dist_dumb_barrier(args, self.device)
        dist_print(args, "--[Done] Passed Barrier Test!!\n")
        dist_print(args, "--[Variables] Initilizing distribution variables")

        # distributed data_loader 
        self.train_sampler, self.train_dataset, self.train_loader,\
            self.test_dataset, self.test_loader = get_data_loader(args)
        ''' debugging: print('Sampler Test', self.train_sampler == self.train_loader.sampler) '''

        # obtain loaders' sizes
        self.train_dataset_size = len(self.train_dataset)
        self.train_loader_size = len(self.train_loader)
        self.test_dataset_size = len(self.test_dataset)
        self.test_loader_size = len(self.test_loader)

        # distributed tensors
        self.dist_params_tensor = ravel_model_params(self.model, is_grad=False, device=self.device)
        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src=0, async_op=False)
        # self.dist_buffers_tensor = ravel_model_buffers(self.model, is_grad=False, device=self.device)
        # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = 0, async_op=False)

        # same model initialization
        unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='copy', model_weight=0)
        # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy', model_weight=0)

        if self.my_rank == 0:
            print('The parameter tensor has length of %d and size of %.3f MB'%(len(self.dist_params_tensor), 32* (1.25e-7)* len(self.dist_params_tensor)))
            # print('The buffer tensor has length of %d and size of %.3f MB'%(len(self.dist_buffers_tensor)), 32* (1.25e-7)* len(self.dist_buffers_tensor))

        # counter
        self.l_comm_counter = 0
        self.dist_cur_lr = self.args.lr

        # define other necessary tensors
        message_size = 3
        self.dist_message_tensor = torch.zeros(message_size, device=self.device) #| rank | loss | LR |
        self.dist_message_list = [torch.zeros(message_size, device=self.device) for i in range(args.world_size)]

        # print success information
        dist_print(args, "[All Preparation Done] Preparation was DONE!")  

    def dist_batch_train(self):
        # 1. update messages
        self.l_comm_counter += 1  
        self.dist_message_tensor[0] = self.my_rank
        self.dist_message_tensor[1] = self.local_train_loss
        self.dist_message_tensor[2] = self.cur_lr
        self.dist_params_req = dist.all_gather_multigpu([self.dist_message_list], [self.dist_message_tensor], async_op=False)
        
        # 2. update parameters and buffers locally
        # local-parameters
        l_best_worker = None # a nonexistent index
        l_lowest_loss = float('inf')  
        for worker in self.dist_message_list[self.args.cur_group* self.args.num_gpus: (self.args.cur_group+1)* self.args.num_gpus]:
            if worker[1].item() < l_lowest_loss:
                l_lowest_loss = worker[1].item()
                l_best_worker = int(worker[0].item())
        assert self.args.cur_group* self.args.num_gpus <= l_best_worker and l_best_worker < (self.args.cur_group+1)* self.args.num_gpus

        if self.my_rank == l_best_worker:
            copy_model_params(self.dist_params_tensor, self.model)
        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src = l_best_worker, group = self.group_dict[self.args.cur_group], async_op=False)
        unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='mix', model_weight=1-self.args.c1)

        # local-buffers
        # copy_model_buffers(self.dist_buffers_tensor, self.model)
        # self.dist_buffers_req = dist.reduce_multigpu([self.dist_buffers_tensor], dst = self.args.cur_group* self.args.num_gpus, group = self.group_dict[self.args.cur_group], async_op=False)
        # if self.my_rank == 0:
        #     self.dist_buffers_tensor.div_(self.args.num_gpus)
        # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = self.args.cur_group* self.args.num_gpus, group = self.group_dict[self.args.cur_group], async_op=False)
        # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy')
        
        # 3. update parameters and buffers globally
        gl_prop = int(max(self.args.g_comm//self.args.l_comm, 1))
        if self.l_comm_counter % gl_prop == 0 and self.args.num_groups > 1:
            # global-parameters
            g_best_worker = None # a nonexistent index
            g_lowest_loss = float('inf')
            for worker in self.dist_message_list:
                if worker[1].item() < g_lowest_loss:
                    g_lowest_loss = worker[1].item()
                    g_best_worker = int(worker[0].item())
            assert 0 <= g_best_worker and  g_best_worker < self.args.num_groups* self.args.num_gpus
            self.world_best_worker = g_best_worker
            if self.my_rank ==  g_best_worker:
                copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src=self.world_best_worker, async_op=False)
            unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='mix', model_weight=1-self.args.c2)

            # global-buffers
            # copy_model_buffers(self.dist_buffers_tensor, self.model)
            # self.dist_buffers_req = dist.reduce_multigpu([self.dist_buffers_tensor], dst = 0, async_op=False)
            # if self.my_rank == 0:
            #     self.dist_buffers_tensor.div_(self.args.world_size)
            # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = 0, async_op=False)
            # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy')