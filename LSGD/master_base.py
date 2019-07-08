import torch
import torch.distributed as dist

class MasterBase():
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This is the base class for master and worker in distributed training ''' 
        # assign class variables
        self.args = args
        self.cur_worker = cur_worker

        self.shared_tensor = shared_tensor
        self.shared_lock = shared_lock
        self.shared_queue_r = shared_queue_r
        self.shared_queue_a = shared_queue_a

        # create center and masters  
        if args.cur_swarm == 0 and cur_worker == args.num_gpus + 1:
            # center
            dist.init_process_group(rank=0, world_size=args.num_swarms+1, 
                                    backend='gloo', init_method=args.dist_url_msr)
        else:
            # master
            dist.init_process_group(rank=args.cur_swarm+1, world_size=args.num_swarms+1, 
                                    backend='gloo', init_method=args.dist_url_msr)

        self.g_tensor = torch.zeros(1)
        dist.all_reduce(self.g_tensor)
        print('Gloo backend is successfully initialized')

        self.my_rank = dist.get_rank()
        self.all_iters = args.iters