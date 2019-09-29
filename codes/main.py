import time
from termcolor import colored

import torch
import torch.multiprocessing as mp

from options import Options
from worker import Worker
from server import LocalServer, GlobalServer

# run distributed training for server or workers
def dist_run(cur_worker, args, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
    ''' dist_run: the role of current process is based on the rank '''
    if cur_worker == args.num_gpus + 1: # global server
        dist_obj = GlobalServer(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)
    elif cur_worker == args.num_gpus:   # local server
        dist_obj = LocalServer(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)
    else: # worker
        dist_obj = Worker(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)
    dist_obj.run()

# MAIN FUNCTION
if __name__ == "__main__": 
    # set up environment to 'spawn' as suggested in PyTorch document
    # https://pytorch.org/docs/master/notes/multiprocessing.html
    # https://pytorch.org/docs/stable/distributed.html
    args = Options().parse()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # set up distribution arguments group indices = (0, 1, ..., s-1)
    assert args.cur_group < args.num_groups, "group index is out of limitation"
    args.world_size = args.num_groups* args.num_gpus
    
    # hierarchy: Global Server --> Local Server --> Worker
    # TCP/IP legal port range from 1024 to 65535
    # a dumb way to use the same ip address but different ports for CPU and GPU
    args.dist_url_msr = "tcp://{ip}:{port}0".format(ip=args.dist_ip, port=args.dist_port)
    args.dist_url_wrk = "tcp://{ip}:{port}5".format(ip=args.dist_ip, port=args.dist_port) 

    # a context must be set properly to use threading/process Lock()/Queue() 
    mp_ctx = mp.get_context('spawn')

    # TODO: this is designed for the extension of LSGD in the future
    # shared resourses (Intra-Devbox) --- shared_tensor
    # |0:current time|1:number of iterations of current devbox|
    # |2+num_gpus*0+curretn_worker:training loss|2+num_gpus*1+curretn_worker:training error|
    # |2+num_gpus*2+curretn_worker:testing loss |2+num_gpus*3+curretn_worker:testing error |
    shared_tensor = torch.zeros(2 + 4* args.num_gpus)
    shared_tensor.share_memory_()

    # shared resourses (Inter-Devbox) --- shared_queue
    # shared_queue_r: worker puts requests for local server
    # shared_queue_a: local server answers to worker
    shared_lock = mp_ctx.Lock()
    shared_queue_a = [mp_ctx.Queue() for i in range(args.num_gpus)]
    shared_queue_r = mp_ctx.Queue()

    # spawn sub-processes for distributed training
    main_program_start = time.time()
    num_processes = args.num_gpus + 1
    if args.cur_group == 0:
        num_processes = args.num_gpus+2
    mp.spawn(fn=dist_run, args=(args, shared_tensor, shared_lock, shared_queue_r, shared_queue_a), nprocs=num_processes)
    
    # Tik Tok: https://www.tutorialspoint.com/python/time_clock.htm
    main_program_finish = time.time()
    main_program_time = main_program_finish-main_program_start
    print(colored("The program: {timer:.3f} seconds".format(timer=main_program_time), 'red'))