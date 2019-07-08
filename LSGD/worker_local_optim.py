import torch
import torch.optim as optim
# import optimizers

import torch.distributed as dist
from local_tools import *
from dist_tools import *
from worker_base import WorkerBase

class WorkerLocalOptim(WorkerBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        '''class for local optimization'''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)
        self.model.train()
        # define optimizer with learning rate augmentation
        self.cur_lr = args.lr
        
        self.local_optimizer = optim.SGD(self.model.parameters(), lr = self.cur_lr, 
                                         weight_decay = args.wd, momentum = args.mom, 
                                         nesterov = (args.mom>0) )

        avg_size = self.args.avg_size
        self.avg_loss, self.avg_err = AverageMeter(running_size=avg_size), AverageMeter(running_size=avg_size)
        self.local_train_loss, self.local_train_error = float('inf'), float('inf')

    def local_batch_train(self, data, target):
        # forward
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(data)
        
        # backward
        self.local_optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.local_optimizer.step()
        
        # update loss and error
        _, predicted = torch.max(output.data, 1)
        correct = (predicted.to(self.device) == target).sum().item()
        error = (1 - correct/ self.args.batch_size)* 100
        self.avg_loss.update(loss.item())
        self.avg_err.update(error)
        return self.avg_loss.running_avg, self.avg_err.running_avg

    def local_center_test(self):
        loss, correct = 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model_center(data)
                loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted.to(self.device) == target).sum().item()
        loss = loss/ self.test_loader_size
        error = (1 - correct/ self.test_dataset_size)* 100
        if self.my_rank % self.args.num_gpus == 0:
            print("{rank:d} [CENTER] TEST LOSS IS {loss:.3f} and ERROR IS {err:.2f}%".format(rank=self.my_rank, loss=loss, err=error))
        return loss, error