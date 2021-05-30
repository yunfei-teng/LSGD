import datetime
import torch
import torch.distributed as dist

class ServerBase():
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' the base class for the server in distributed training ''' 
        # assign class variables
        self.args = args
        self.cur_worker = cur_worker

        self.shared_tensor = shared_tensor
        self.shared_lock = shared_lock
        self.shared_queue_r = shared_queue_r
        self.shared_queue_a = shared_queue_a

        # create global server and local server 
        timeout = datetime.timedelta(seconds=7200)
        if args.cur_group == 0 and cur_worker == args.num_gpus + 1: # global server 
            dist.init_process_group(rank=0, world_size=args.num_groups+1, backend='gloo', timeout=timeout, init_method=args.dist_url_msr)
        else: # local server
            dist.init_process_group(rank=args.cur_group+1, world_size=args.num_groups+1, backend='gloo', timeout=timeout, init_method=args.dist_url_msr)

        # test initialization
        self.g_tensor = torch.zeros(1)
        dist.all_reduce(self.g_tensor)

        # define variables
        self.my_rank = dist.get_rank()
        self.all_iters = args.iters