import sys
import time
import threading
from queue import Queue

import torch
import torch.distributed as dist

from local_tools import *
from dist_tools import *
from master_base import MasterBase

def listen(lst_rank, p_comm_queue, g_comm, world_size, world_best_worker, world_least_loss, begin_time, period, t_signal):
    request = torch.FloatTensor([0])
    answer  = torch.CharTensor([0])
    is_terminiating = False
    while not is_terminiating:
        # read message
        dist.recv(request, src = lst_rank)
        
        # answer message 
        p_comm = p_comm_queue.get()
        if t_signal[0].item() == 1:
            answer[0] = 2
            is_terminiating = True

        # consider workers=4 and l_comm=1
        # then 8-L->1-L->2-L->3-L->4-D->5-D->6-D->7-D->8
        # where L and D represents "local training" and "distributed training" respectively
        elif p_comm >= g_comm + world_size:
            if request.item() > period and p_comm == g_comm + world_size:
                t_signal[0] = 1
                answer[0] = 2
                is_terminiating = True
            else:
                p_comm = 0
                answer[0] = 0

        elif p_comm >= g_comm:
            answer[0] = 1
        else:
            answer[0] = 0
        p_comm_queue.put(p_comm+1)
        dist.send(answer, dst = lst_rank)

    time.sleep(3) # wait 3 seconds for GPU processors to ternimate
    

class Master(MasterBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This class defined the behavior of Master '''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)

    def run(self):
        print("== Running MAIN FUCTION as [SWARM %d: Master] =="%(self.args.cur_swarm))
        request = torch.FloatTensor([0])
        answer  = torch.CharTensor([0])
        begin_time = time.time()
        is_terminiating = False
        while not is_terminiating:            
            train_time, rank = self.shared_queue_r.get()
            request[0] = train_time
            dist.send(request, dst = 0)
            dist.recv(answer,  src = 0)
            self.shared_queue_a[rank].put(answer.item())
            if answer.item() == 2:
                for i in range(0, self.args.num_gpus):
                    self.shared_queue_a[i].put(answer.item())
                is_terminiating = True
        
        time.sleep(3) # wait 3 seconds for GPU processors to ternimate
        dist.barrier()

class Center(MasterBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This class defined the behavior of Center '''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)

    def run(self):
        print("== Running MAIN FUCTION as [SWARM %d: Center] =="%(self.args.cur_swarm))
        p_comm_queue = Queue()
        p_comm_queue.put(0)
        g_comm = self.args.num_swarms* self.args.num_gpus* self.args.l_comm
        world_best_worker = torch.CharTensor([0])
        world_least_loss = torch.FloatTensor([float('inf')])
        t_signal = torch.CharTensor([0])
        
        world_size = self.args.num_swarms* self.args.num_gpus
        begin_time = time.time()
        period = self.args.minutes* 60 + self.args.hours* 3600
        listener_threads = []
        for i in range(1, dist.get_world_size()):
            p = threading.Thread(target=listen, \
                args=(i, p_comm_queue, g_comm, world_size, world_best_worker, world_least_loss, begin_time, period, t_signal,))
            listener_threads.append(p)
        for p in listener_threads:
            p.start()
        for p in listener_threads:
            p.join()
        dist.barrier()