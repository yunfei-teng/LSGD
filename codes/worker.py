import time
import torch
import torch.distributed as dist

from local_tools import *
from dist_tools import *
from worker_dist_optim import WorkerDistOptim
from worker_local_optim import WorkerLocalOptim

class Worker(WorkerDistOptim, WorkerLocalOptim):
    def __init__(self, args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a):
        ''' This class defined the behavior of Sub-worker '''
        super().__init__(args, cur_worker, shared_tensor, shared_lock, shared_queue_r, shared_queue_a)
        self.train_time = 0
        self.test_time = 0
        self.raw_time = 0 # raw_time = total_time - test_time

        self.wait_time, self.wait_counter   = 0, 1 # time for sending request and waiting answer 
        self.read_time, self.read_counter   = 0, 1 # time for reading data
        self.local_time, self.local_counter = 0, 1 # time for local training
        self.dist_time, self.dist_counter   = 0, 1 # time for distributed training

        self.local_train_num_iters = 0
        self.dist_train_num_iter = 0

        self.local_cur_lr = self.cur_lr
        self.epoch = 0
        self.last_saved_epoch = 0

    def run(self): 
        print("== Running MAIN FUCTION as [Group %d: GPU %d] =="%(self.args.cur_group, self.cur_worker))
        exp_dir = os.path.join(self.args.checkpoints_dir, self.args.exp_name)
        file_name = os.path.join(exp_dir, 'messages.csv')
        csv_fields =  "epoch,rank,best_worker,local_iters,world_iters,raw_time,train_time,"
        csv_fields += "train_loss,train_error,test_loss,test_error,LLR,DLR"
        message_file = open(file_name, "w")
        message_file.write(csv_fields)
        message_file.close()

        train_begin = time.time()
        train_stop = False
        comm_iters = self.args.num_groups* self.args.num_gpus* self.args.l_comm
        
        while not train_stop:
            # set up traininig sampler for current epoch
            self.train_sampler.set_epoch(self.epoch)
            train_iterator = self.train_loader.__iter__()
            train_iterator_len = len(train_iterator)

            # reset batch index
            batch_idx = 0
            while batch_idx < train_iterator_len:
                batch_begin = time.time()
                self.shared_queue_r.put((self.raw_time, self.my_gpu))
                message = self.shared_queue_a[self.my_gpu].get()
                self.wait_time += time.time() - batch_begin
                self.wait_counter += 1

                # the model will either do local trainig or distributed training
                if message == 0: # local training
                    batch_idx += 1
                    self.local_train_num_iters += 1
                    
                    data, target = next(train_iterator)
                    self.read_time += time.time() - batch_begin 
                    self.read_counter += 1

                    self.local_train_loss, self.local_train_error = self.local_batch_train(data, target)
                    self.local_time += time.time() - batch_begin
                    self.local_counter += 1

                    self.train_time += time.time() - batch_begin

                elif message == 1: # distributed training
                    self.dist_batch_train() 
                    if self.my_rank == self.world_best_worker:
                        ten_iteration_time1 = 10* self.raw_time/ self.local_train_num_iters
                        ten_iteration_time2 = 10* self.train_time/ self.local_train_num_iters
                        info_to_print = '\n[worker: {}] Epoch: {} {}/{} ({:.1f}%) Loss: {:.6f} Error: {:.2f}%'.format(
                                        self.my_rank, self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                        100. * batch_idx * len(data) /len(self.train_loader.dataset), 
                                        self.local_train_loss, self.local_train_error)
                        info_to_print += '\n -- Current passes time: %.8f/%.8f'%(self.raw_time, self.train_time)
                        info_to_print += '\n -- Average 10 iterations time: %.8f/%.8f)'%(ten_iteration_time1, ten_iteration_time2)
                        print(info_to_print)
                    dist_dumb_barrier(self.args, self.device) 

                    self.dist_train_num_iter += 1
                    self.dist_counter += 1
                    self.dist_time += time.time() - batch_begin
                    self.train_time += time.time() - batch_begin  
                  
                    # only testing the model after the workers communicate with each other
                    if self.dist_train_num_iter % 16 == 0: # HARDCODED for now
                        test_begin = time.time()

                        # test with the averaged parameters
                        copy_model_params(self.dist_params_tensor, self.model)
                        self.dist_params_req = dist.all_reduce_multigpu([self.dist_params_tensor], async_op=False)
                        self.dist_params_tensor.div_(self.args.world_size) 
                        unravel_model_params(self.model_center, self.dist_params_tensor, is_grad=False, operation='copy')
                        # unravel_model_buffers(self.model_center, self.dist_buffers_tensor, is_grad=False, operation='copy')
                        test_loss, test_error = self.local_center_test()
                        self.dist_params_tensor.zero_() # not necessary but for safety we'd better zero out the distributed tensors

                        # write current status to a csv file
                        # the recorded training time is slightly shorter than its actual value 
                        # since the time of last distributed training is not counted in
                        message_file = open(file_name, "a")
                        _text = "\n%d,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f"\
                        %(self.epoch, self.my_rank, self.world_best_worker, self.local_train_num_iters, comm_iters* self.dist_train_num_iter, self.raw_time, \
                        self.train_time, self.local_train_loss, self.local_train_error, test_loss, test_error, self.local_cur_lr, self.dist_cur_lr)
                        message_file.write(_text)
                        message_file.close()

                        # save checkpoint
                        if self.epoch % self.args.check_point_epochs == 0 and self.last_saved_epoch != self.epoch:
                            self.last_saved_epoch = self.epoch
                            expr_dir = os.path.join(self.args.checkpoints_dir, self.args.exp_name)
                            state = {
                                'epoch': self.epoch,
                                'exp_name': self.args.exp_name,
                                'worker': '%d/%d'%(self.my_rank, self.args.world_size),
                                'iters': self.dist_train_num_iter* comm_iters,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.local_optimizer.state_dict()
                            }
                            torch.save(state, os.path.join(expr_dir,'checkpoint-w%d-e%d.pt'%(self.my_rank, self.epoch)))

                        # statistics for local and distributed training
                        if self.my_rank % self.args.num_gpus == 0:
                            print('waiting takes [%.3f/%.3f]'%(self.wait_time, self.wait_time/self.wait_counter))
                            print('reading data takes [%.3f/%.3f]'%(self.read_time, self.read_time/self.read_counter))
                            print('local training takes [%.3f/%.3f]'%(self.local_time, self.local_time/self.local_counter))
                            print('distributed training takes [%.3f/%.3f]'%(self.dist_time, self.dist_time/self.dist_counter))
                            print('The ratio is: [%.3f %%]'%(self.dist_time/(self.dist_time+self.local_time)* 100))

                        # test time
                        dist_dumb_barrier(self.args, self.device)
                        self.test_time += time.time() - test_begin

                else:  # stop training
                    train_stop = True
                    break

                # update total training time
                self.raw_time = time.time() - train_begin - self.test_time

            # update number of epochs
            self.epoch = self.epoch + 1

        # summarization
        summary_text  = '\n[Distributed Training Summarization] WORKER: %d'%self.my_rank
        summary_text += "\nlocal training time is %.8f"%(self.raw_time)
        summary_text += "\nlocal optimization time is %.8f"%(self.train_time)
        summary_text += "\nnumber of iterations is %d"%self.local_train_num_iters
        summary_text += "\ntotal number of distributed optimization is %d"%self.dist_train_num_iter
        print(summary_text)

        expr_dir = os.path.join(self.args.checkpoints_dir, self.args.exp_name)
        file_name = os.path.join(expr_dir, 'args.txt')
        with open(file_name, 'a') as args_file:
            args_file.write(summary_text)
            args_file.write('\n')