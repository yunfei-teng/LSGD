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
        dist.init_process_group(rank=self.my_rank, world_size=args.world_size, 
                                backend=args.dist_backend, init_method=args.dist_url_wrk)

        self.group_dict = {}
        for i in range(args.num_groups):
            _group = [j for j in range(self.args.cur_group* self.args.num_gpus, (self.args.cur_group+1)* self.args.num_gpus)]
            self.group_dict[i] = torch.distributed.new_group(_group)
        
        # barrier test
        dist_print(args, "--[Barrier] Now entering barrier test!!")
        dist_dumb_barrier(args, self.device)

        # distributed data_loader 
        dist_print(args, "--[Loader] Preparing data sets and data loaders")
        self.train_sampler, self.train_dataset, self.train_loader,\
            self.test_dataset, self.test_loader = get_data_loader(args)
        ''' debugging: print('Sampler Test', self.train_sampler == self.train_loader.sampler) '''

        # obtain loaders' sizes
        self.train_dataset_size = len(self.train_dataset)
        self.train_loader_size = len(self.train_loader)
        self.test_dataset_size = len(self.test_dataset)
        self.test_loader_size = len(self.test_loader)

        # distributed tensors
        dist_print(args, "--[Variables] Initilizing distribution variables")
        self.dist_params_tensor = ravel_model_params(self.model, is_grad=False, device=self.device)
        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src=0, async_op=False)
        self.dist_group_params_tensor = self.dist_params_tensor.clone()
        # self.dist_buffers_tensor = ravel_model_buffers(self.model, is_grad=False, device=self.device)
        # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = 0, async_op=False)

        # same model initialization
        unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='copy', model_weight=0)
        self.dist_params_tensor_msr = self.dist_params_tensor.clone()

        # model buffers
        # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy', model_weight=0)

        if self.my_rank % self.args.num_gpus == 0:
            print('The parameter tensor has length of %d and size of %.3f MB'%(len(self.dist_params_tensor), 32* (1.25e-7)* len(self.dist_params_tensor)))
            # print('The buffer tensor has length of %d and size of %.3f MB'%(len(self.dist_buffers_tensor)), 32* (1.25e-7)* len(self.dist_buffers_tensor))
            print('\n=== Proceeding to main subprocess===')
        
        # counter
        self.l_comm_counter = 0
        self.g_comm_counter = 0
        self.dist_cur_lr = self.args.lr

        # define other necessary tensors
        message_size = 3
        self.dist_message_tensor = torch.zeros(message_size, device=self.device) #| rank | loss | LR |
        self.dist_message_list = [torch.zeros(message_size, device=self.device) for i in range(args.world_size)]

    def dist_lsgd_train(self):
        if self.args.landscape:
            _model_params_tensor = self.dist_group_params_tensor.clone()
            copy_model_params(_model_params_tensor, self.model)

        # 1. update messages
        self.l_comm_counter += 1  
        self.dist_message_tensor[0] = self.my_rank
        self.dist_message_tensor[1] = self.cur_lr
        self.dist_message_tensor[2] = self.local_train_loss
        self.dist_params_req = dist.all_gather_multigpu([self.dist_message_list], [self.dist_message_tensor], async_op=False)
        
        # 2.1 update parameters and buffers locally
        if self.args.is_lcomm:
            # local-parameters
            cur_group_message = self.dist_message_list[self.args.cur_group* self.args.num_gpus: (self.args.cur_group+1)* self.args.num_gpus]
            l_leader = min(cur_group_message, key = lambda x: x[2].item())
            l_leader_rank = int(l_leader[0].item())
            assert self.args.cur_group* self.args.num_gpus <= l_leader_rank and l_leader_rank < (self.args.cur_group+1)* self.args.num_gpus
            self.world_best_worker = l_leader_rank
            if self.my_rank == l_leader_rank:
                copy_model_params(self.dist_group_params_tensor, self.model)
            self.dist_params_req = dist.broadcast_multigpu([self.dist_group_params_tensor], src = l_leader_rank, group = self.group_dict[self.args.cur_group], async_op=False)
            unravel_model_params(self.model, self.dist_group_params_tensor, is_grad=False, operation='mix', model_weight=1-self.args.c1)

            # local-buffers
            # copy_model_buffers(self.dist_buffers_tensor, self.model)
            # self.dist_buffers_req = dist.reduce_multigpu([self.dist_buffers_tensor], dst = self.args.cur_group* self.args.num_gpus, group = self.group_dict[self.args.cur_group], async_op=False)
            # if self.my_rank == 0:
            #     self.dist_buffers_tensor.div_(self.args.num_gpus)
            # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = self.args.cur_group* self.args.num_gpus, group = self.group_dict[self.args.cur_group], async_op=False)
            # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy')
        
        # 3. update parameters and buffers globally
        gl_prop = int(max(self.args.g_comm//self.args.l_comm, 1))
        if self.l_comm_counter % gl_prop == 0:
            self.g_comm_counter += 1
            # global-parameters
            g_leader = min(self.dist_message_list, key = lambda x: x[2].item())
            g_leader_rank = int(g_leader[0].item())
            assert 0 <= g_leader_rank and  g_leader_rank < self.args.num_groups* self.args.num_gpus
            self.world_best_worker = g_leader_rank
            if self.my_rank ==  g_leader_rank:
                copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor], src=g_leader_rank, async_op=False)
            unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='mix', model_weight=1-self.args.c2) 

            # global-buffers
            # copy_model_buffers(self.dist_buffers_tensor, self.model)
            # self.dist_buffers_req = dist.reduce_multigpu([self.dist_buffers_tensor], dst = 0, async_op=False)
            # if self.my_rank == 0:
            #     self.dist_buffers_tensor.div_(self.args.world_size)
            # self.dist_buffers_req = dist.broadcast_multigpu([self.dist_buffers_tensor], src = 0, async_op=False)
            # unravel_model_buffers(self.model, self.dist_buffers_tensor, is_grad=False, operation='copy')
            
    def dist_lsgd_test(self):
        if self.args.is_lcomm:
            copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.all_reduce_multigpu([self.dist_params_tensor], group = self.group_dict[self.args.cur_group], async_op=False)
            self.dist_params_tensor.div_(self.args.num_gpus)
            unravel_model_params(self.model_center, self.dist_params_tensor, is_grad=False, operation='copy')
            # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
            test_loss, test_error = self.local_center_test()
        else:
            copy_model_params(self.dist_params_tensor, self.model)
            self.dist_params_req = dist.all_reduce_multigpu([self.dist_params_tensor], async_op=False)
            self.dist_params_tensor.div_(self.args.world_size)
            unravel_model_params(self.model_center, self.dist_params_tensor, is_grad=False, operation='copy')
            # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
            test_loss, test_error = self.local_center_test()
        return test_loss, test_error
                            
    def dist_easgd_train(self):
        # [easgd] step 1: master gather current information from all workers step 2: master broadcast its parameters from last communication period 
        # The tricky part for EASGD is that: the update of master involves workers x_t instrad of x_{t+1}
        # self.args.beta = 0.43
        self.args.beta = self.args.c2
        copy_model_params(self.dist_params_tensor, self.model)
        self.dist_params_req = dist.reduce_multigpu([self.dist_params_tensor], dst = 0, async_op=False)
        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor_msr], src = 0, async_op=False)
        
        # [easgd] master and workers will be pulled to each other: (a) workers pull to master (b) master pulls to workers
        # The update comes from easgd paper equations (5) and (6)
        alpha = self.args.beta/self.args.world_size            
        unravel_model_params(self.model, self.dist_params_tensor_msr, is_grad=False, operation='mix', model_weight=1-alpha)
        if self.my_rank == 0:
            self.dist_params_tensor.div_(self.args.world_size)
            self.dist_params_tensor_msr.mul_(1-self.args.beta)
            self.dist_params_tensor_msr.add_(self.args.beta, self.dist_params_tensor)
               
    def dist_easgd_test(self):
        # test with the averaged parameters
        self.dist_params_req = dist.broadcast_multigpu([self.dist_params_tensor_msr], src = 0, async_op=False)
        unravel_model_params(self.model_center, self.dist_params_tensor_msr, is_grad=False, operation='copy')
        # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
        test_loss, test_error = self.local_center_test()
        return test_loss, test_error
                            
    def dist_train(self):
        ''' distributed training '''
        # ============== WORK IN PROGRESS (***) ================
        # assign the values of exponetial averaging tensor mu to current model parameters
        # if self.args.weight_averaging:
        #     unravel_model_params(self.model, self.local_mu_tensor, is_grad=False, operation='copy')
        if self.args.weight_averaging:
            self.local_x_tensor.mul_(1-self.args.etagamma).add_(self.args.etagamma, self.local_mu_tensor)
            unravel_model_params(self.model, self.local_x_tensor, is_grad=False, operation='copy')
        # ============== WORK IN PROGRESS (***) ================

        if self.args.dist_optimizer == 'LSGD':
            self.dist_lsgd_train()

        elif self.args.dist_optimizer == 'EASGD':
            self.dist_easgd_train()

        # ============== WORK IN PROGRESS (****) ===============
        # assign current model parameters' values to exponetial averaging tensor mu
        if self.args.weight_averaging:
            copy_model_params(self.local_x_tensor, self.model)
            self.local_mu_tensor.copy_(self.local_x_tensor)
        # ============== WORK IN PROGRESS (****) ===============

    def dist_test(self):
        ''' distributed testing '''
        if self.args.dist_optimizer == 'LSGD':
            test_loss, test_error = self.dist_lsgd_test()

        elif self.args.dist_optimizer == 'EASGD':
            test_loss, test_error = self.dist_easgd_test()

        # not necessary but for safety we'd better zero out the distributed tensors
        self.dist_params_tensor.zero_()
        return test_loss, test_error

    # -----------------------------
    # ----- Debugging Section -----
    # -----------------------------
    ''' The debugging section may be buggy itself... '''
    def dist_lsgd_landscape(self):
        def _plot_landscape1(self, _model_params_tensor):
            '''We will plot landscape on a two-dimensional plane only for ordinary workers
            The plane is expanded by three points: model, local leader and global leader '''
            num_steps = 5
            base_params_tensor = _model_params_tensor.clone()
            # val_params_tensor.copy_(val_params_tensor-0.1* self.dist_params_tensor).mul_(1/0.9)
            dir_tensor1 = self.dist_params_tensor - base_params_tensor
            dir_tensor2 = self.dist_group_params_tensor - base_params_tensor
            # print((dir_tensor1*dir_tensor2).sum().div(dir_tensor1.norm()).div(dir_tensor2.norm()).item())

            base_params_tensor.add_(-1, dir_tensor1)
            base_params_tensor.add_(-1, dir_tensor2)
            dir_tensor1.div_(num_steps)
            dir_tensor2.div_(num_steps)        
            
            start = time.time()
            loss_landscape, error_landscape = [], []
            for step1 in range(3* num_steps):
                for step2 in range(3* num_steps):
                    model_params_tensor = base_params_tensor.add(step1, dir_tensor1).add(step2, dir_tensor2)
                    unravel_model_params(self.model_center, model_params_tensor, is_grad=False, operation='copy')
                    # print('-- [%d] -- '%(step+1), end = '')
                    loss, error = self.local_center_test()
                    loss_landscape  += [loss]
                    error_landscape += [error]
                current_time = time.time() - start
                average_time = current_time/(3* num_steps* (step1+1))
                left_time = (3* num_steps)** 2 * average_time - current_time
                if self.my_rank % self.args.num_gpus == 0:
                    print('\n== Average time is [%.2fs] and the left time is [%.2fs] == '%(average_time, left_time))
            return loss_landscape, error_landscape

        def _plot_landscape2(self, grid_size, dir1, dir2, _model_params_tensor):
            '''We will plot landscape on a two-dimensional plane only for ordinary workers
            The plane is expanded by three points: model, local leader and global leader '''
            num_steps = 16
            base_params_tensor = _model_params_tensor.clone()
            base_params_tensor.add_(-grid_size, dir1)
            base_params_tensor.add_(-grid_size, dir2)
            m_dir1 = dir1.mul(grid_size/num_steps)
            m_dir2 = dir2.mul(grid_size/num_steps)
            
            start = time.time()
            loss_landscape, error_landscape = [], []
            for step1 in range(2* num_steps):
                for step2 in range(2* num_steps):
                    model_params_tensor = base_params_tensor.add(step1, m_dir1).add(step2, m_dir2)
                    unravel_model_params(self.model_center, model_params_tensor, is_grad=False, operation='copy')
                    loss, error = self.local_center_test()
                    loss_landscape  += [loss]
                    error_landscape += [error]
                current_time = time.time() - start
                average_time = current_time/(2* num_steps* (step1 + 1))
                left_time = (2* num_steps)** 2 * average_time - current_time
                if self.my_rank % self.args.num_gpus == 0:
                    print('\n == Average time is [%.2fs] and the left time is [%.2fs] == '%(average_time, left_time))
            return loss_landscape, error_landscape

            if not self.args.landscape:
                return
            # define landscape with all local leaders
            dist_leaders_tensor = torch.zeros(1, device=self.device)
            dist_leaders_list = [torch.zeros(1, device=self.device) for _ in range(self.args.world_size)]
            dist_params_list = [self.dist_params_tensor.clone() for _ in range(self.args.world_size)]

            expr_dir = os.path.join(self.args.checkpoints_dir, self.args.exp_name)
            if self.my_rank !=  l_leader_rank:
                dist_leaders_tensor[0] = 0
            elif self.my_rank !=  g_leader_rank:
                dist_leaders_tensor[0] = 1
            else:
                dist_leaders_tensor[0] = 2
            dist.all_gather_multigpu([dist_leaders_list], [dist_leaders_tensor], async_op=False)
            dist.all_gather_multigpu([dist_params_list], [_model_params_tensor], async_op=False)
            dist_dumb_barrier(self.args, self.device) # synchronization
            
            dir1 = dir2 = None
            if self.my_rank !=  l_leader_rank:
                # 1. unnormalized landscape of current woker to local leader and global leader
                # print('[Worker %d] is entering landscape plots'%self.my_rank)
                # loss_landscape, error_landscape = self._plot_landscape1(_model_params_tensor)
                # torch.save(loss_landscape, os.path.join(expr_dir, 'landscape-loss-w%d-g%d.pth'%(self.my_rank, self.g_comm_counter)))
                # torch.save(error_landscape, os.path.join(expr_dir, 'landscape-error-w%d-g%d.pth'%(self.my_rank, self.g_comm_counter)))
            
                # 2. (a) normalized landscape of current woker to local leader and global leader
                #    (b) the trajectory of one epoch of stochastic optimization will be recorded
                dir1 = self.dist_params_tensor - _model_params_tensor
                dir2 = self.dist_group_params_tensor - _model_params_tensor
            
            else:
                # (a) normalized landscape of current leader to other two arbitrary leaders
                # (b) the trajectory of one epoch of stochastic optimization will be recorded
                m_leaders = []
                for rank, (leader, params) in enumerate(zip(dist_leaders_list, dist_params_list)):
                    if not (leader.item() == 0 or rank == l_leader_rank):
                        m_leaders += [params]
                dir1 = m_leaders[0] - _model_params_tensor
                dir2 = m_leaders[1] - _model_params_tensor

            dir1_norm, dir2_norm = dir1.norm().item(), dir2.norm().item()
            dir1, dir2 = dir1.div(dir1_norm), dir2.div(dir2_norm)
            print('The distances to local leader and global leader are %.2f and %.2f respectively.'%(dir1_norm, dir2_norm))

            # trajectory (current model) always starts from the origin
            grid_size = 3
            dist_dumb_barrier(self.args, self.device) # synchronization
            loss_landscape, error_landscape = self._plot_landscape2(grid_size, dir1, dir2, _model_params_tensor)
            landscape_train_loader = udata.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        num_workers=0,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        drop_last=True)

            m_points = [[0, 0]] # points along the trajectory for interpolation
            cur_model_params_tensor = _model_params_tensor.clone()
            for data, target in landscape_train_loader:
                self.local_batch_train(data, target)    
                copy_model_params(cur_model_params_tensor, self.model)
                m_dir1 = ((cur_model_params_tensor-_model_params_tensor)* dir1).sum().item()
                m_dir2 = ((cur_model_params_tensor-_model_params_tensor)* dir2).sum().item()
                m_points += [[m_dir1, m_dir2]]

            #dist_points_tensor = torch.FloatTensor(m_points).to(self.device)
            #dist_points_list = [dist_points_tensor.clone() for _ in range(self.args.world_size)]
            #dist.all_gather_multigpu([dist_points_list], [dist_points_tensor], async_op=False)
            contour = {'loss': loss_landscape, 'error': error_landscape, 'point': m_points, 'leader':[dir1_norm, dir2_norm], 'gs':grid_size, 'identity':dist_leaders_tensor[0].item()}
            torch.save(contour, os.path.join(expr_dir, 'leader-contour-w%d-g%d.pth'%(self.my_rank, self.g_comm_counter)))
            unravel_model_params(self.model, _model_params_tensor, is_grad=False, operation='copy', model_weight=0)