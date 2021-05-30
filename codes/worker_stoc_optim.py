import torch
import torch.optim as optim
import torch.nn.functional as F
# import optimizers

import torch.distributed as dist
from local_tools import *
from dist_tools import *
from worker_base import WorkerBase

class WorkerStocOptim(WorkerBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' the class for local training and testing '''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)
        self.model.train()
        # define optimizer with learning rate augmentation
        self.cur_lr = args.lr
        if args.lr_lin:
            self.cur_lr = (self.my_rank + 1)* self.cur_lr
        if args.lr_pow:
            self.cur_lr = (2** self.my_rank)* self.cur_lr

        self.local_optimizer = optim.SGD(self.model.parameters(), lr = self.cur_lr, 
                                         weight_decay = args.wd, momentum = args.mom, 
                                         nesterov = (args.mom>0))

        avg_size = self.args.avg_size
        self.avg_loss_meter, self.avg_err_meter = AverageMeter(running_size=avg_size), AverageMeter(running_size=avg_size)
        self.local_train_loss, self.local_train_error = float('inf'), float('inf')
        if self.args.weight_averaging:
            self.local_x_tensor = ravel_model_params(self.model, is_grad=False, device=self.device)
            self.local_mu_tensor = ravel_model_params(self.model, is_grad=False, device=self.device)
            self.local_params_tensor = ravel_model_params(self.model, is_grad=False, device=self.device)

    def local_batch_train(self, data, target):
        ''' train with a batch of training data points '''
        # forward
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(data)
        
        # backward
        self.local_optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.local_optimizer.step()

        # proximal operator
        p1 = self.args.p1 / self.args.l_comm
        p2 = self.args.p2 / self.args.g_comm
        if not self.args.no_prox:
            unravel_model_params(self.model, self.dist_params_tensor, is_grad=False, operation='mix', model_weight=1-p2)            
            if self.args.is_lcomm:
                unravel_model_params(self.model, self.dist_group_params_tensor, is_grad=False, operation='mix', model_weight=1-p1)
        
        # ============== WORK IN PROGRESS (* and **) ================
        if self.args.weight_averaging:
            unravel_model_params(self.model, self.local_x_tensor, is_grad=False, operation='mix', model_weight=self.args.gamma)
            copy_model_params(self.local_params_tensor, self.model)
            self.local_mu_tensor.mul_(1-self.args.alpha).add_(self.args.alpha, self.local_params_tensor)
        # ============== WORK IN PROGRESS (* and **) ================

        # update loss and error
        _, predicted = torch.max(output.data, 1)
        correct = (predicted.to(self.device) == target).sum().item()
        error = (1 - correct/ self.args.batch_size)* 100
        self.avg_loss_meter.update(loss.item())
        self.avg_err_meter.update(error)
        return self.avg_loss_meter.running_avg, self.avg_err_meter.running_avg

    def local_center_test(self):
        ''' test center variable (model) with testing dataset '''
        with torch.no_grad():
            # re-run train loader to update statistics information in batch norm layers
            if 'resnet' in self.args.model or 'vgg' in self.args.model:
                self.model_center.train()
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data = data.to(self.device, non_blocking=True)
                    output = self.model_center(data)
            
            self.model_center.eval()
            loss, correct = 0, 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model_center(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted.to(self.device) == target).sum().item()
        loss = loss/ len(self.test_loader.dataset)
        error = (1 - correct/ self.test_dataset_size)* 100
        
        # For the model associated with batch norm layers we print all test results of each worker
        if 'resnet' in self.args.model or self.my_rank % self.args.num_gpus == 0:
            print("\n({t:.3f}s) TEST [LOSS IS {loss:.3f}] and [ERROR IS {err:.2f}%]".format(t=self.raw_time, loss=loss, err=error))
        return loss, error