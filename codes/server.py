import sys
import time
import threading
from queue import Queue

import torch
import torch.distributed as dist

from local_tools import *
from dist_tools import *
from server_base import ServerBase

class LocalServer(ServerBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This class defines the behavior of local server '''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)

    def run(self):
        print("== [Local Server %d:] is running =="%(self.args.cur_group))
        request = torch.FloatTensor([0])
        answer  = torch.CharTensor([0])
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

def listen(lst_rank, iter_counter_queue, g_comm, world_size, begin_time, period, t_signal):
    ''' This function defines how the global server receives and answers requests from local servers
        # case 1: already set to termination.
        # case 2: current total number of iterations is greater or equal to g_comm + world_size
        #         i.e. all the workers just have done distributed training once
        #   - case 2.1: need to terminate training process
        #   - case 2.2: reset total number of iterations to 0
        # case 3: distributed training signal
        # case 4: local training signal
        e.g. consider workers=4 and l_comm=1
        then 8-L->1-L->2-L->3-L->4-D->5-D->6-D->7-D->8
        where L and D represent "local training" and "distributed training" respectively    
    '''
    request = torch.FloatTensor([0])
    answer  = torch.CharTensor([0])
    is_terminiating = False
    while not is_terminiating:
        # read message
        dist.recv(request, src = lst_rank)
        iter_counter = iter_counter_queue.get()
        if t_signal[0].item() == 1: # case 1
            answer[0] = 2
            is_terminiating = True
        elif iter_counter >= g_comm + world_size: # case 2
            if request.item() > period and iter_counter == g_comm + world_size: # case 2.1
                t_signal[0] = 1
                answer[0] = 2
                is_terminiating = True
            else: # case 2.2
                iter_counter = 0
                answer[0] = 0
        elif iter_counter >= g_comm: # case 3
            answer[0] = 1
        else: # case 4
            answer[0] = 0
        
        # answer message
        iter_counter_queue.put(iter_counter+1)
        dist.send(answer, dst = lst_rank)

    time.sleep(3) # wait 3 seconds for GPU processors to ternimate

class GlobalServer(ServerBase):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This class defines the behavior of global server '''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)

    def run(self):
        print("== [Global Server] is running ==")
        iter_counter_queue = Queue()
        iter_counter_queue.put(0)
        g_comm = self.args.num_groups* self.args.num_gpus* self.args.l_comm
        t_signal = torch.CharTensor([0]) # terminating signal (will be shared by all threads)
        
        world_size = self.args.num_groups* self.args.num_gpus
        begin_time = time.time()
        period = self.args.minutes* 60 + self.args.hours* 3600
        
        # each thread listens to a corresponding group
        listener_threads = []
        for i in range(1, dist.get_world_size()):
            p = threading.Thread(target=listen, args=(i, iter_counter_queue, g_comm, world_size, begin_time, period, t_signal,))
            listener_threads.append(p)
        for p in listener_threads:
            p.start()
        for p in listener_threads:
            p.join()
        dist.barrier()