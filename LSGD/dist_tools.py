import time
import torch
import torch.distributed as dist

def set_global_rank(args, cur_worker):
    '''return global rank base on the index of current swarm 
    and the index of current work inside the swarm'''
    global_rank = cur_worker + args.cur_swarm* args.num_gpus
    return global_rank

def dist_print(args, text2print):
    '''only one of the workers (processes) in the same swarm will print out information'''
    if dist.get_rank() % args.num_gpus == 0:
        print(text2print)

def dist_dumb_barrier(args, device):
    '''alternative version of barrier to replace PyTorch's impletation of barrier'''
    dumb_tensor = torch.FloatTensor([1]).to(device)
    dist.all_reduce_multigpu([dumb_tensor], dist.ReduceOp.SUM)
    assert dumb_tensor.item() == args.world_size, 'NCCL was not intialized properly'

def ravel_model_params(model, is_grad, device):
    ''' squash model parameters or gradients into a flat tensor (https://github.com/ucla-labx/distbelief)'''
    numel = 0
    for parameter in model.parameters():
        numel += parameter.data.numel()
    flat_tensor = torch.zeros(numel).to(device)
    current_index = 0
    for parameter in model.parameters():
        if is_grad:
            numel = parameter.grad.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.grad.data.view(-1))
        else:
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.data.view(-1))
        current_index += numel 
    return flat_tensor

def ravel_model_buffers(model, is_grad, device):
    ''' squash model buffers or gradients into a flat tensor '''
    numel = 0
    for parameter in model.buffers():
        numel += parameter.data.numel()
    flat_tensor = torch.zeros(numel).to(device)
    current_index = 0
    for parameter in model.buffers():
        if is_grad:
            numel = parameter.grad.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.grad.data.view(-1))
        else:
            numel = parameter.data.numel()
            flat_tensor[current_index:current_index+numel].copy_(parameter.data.view(-1))
        current_index += numel 
    return flat_tensor

def mix_model_params(model, flat_tensor, tensor_weight=1):
    ''' squash model parameters or gradients into a flat tensor '''
    current_index = 0
    for parameter in model.parameters():
        numel = parameter.data.numel()
        flat_tensor[current_index:current_index+numel].mul_(tensor_weight).add_(parameter.data.mul(1-tensor_weight).view(-1))
        current_index += numel 

def copy_model_params(flat_tensor, model):
    ''' copy model parameters into flat tensor '''
    current_index = 0 
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        flat_tensor[current_index:current_index+numel].data.copy_(parameter.data.view(-1))
        current_index += numel

def copy_model_buffers(flat_tensor, model):
    ''' copy model parameters into flat tensor '''
    current_index = 0
    for parameter in model.buffers():
        numel = parameter.data.numel()
        size = parameter.data.size()
        flat_tensor[current_index:current_index+numel].data.copy_(parameter.data.view(-1))
        current_index += numel

def add_model_grads(flat_tensor, model):
    ''' add model parameters into flat tensor '''
    current_index = 0 
    for parameter in model.parameters():
        numel = parameter.grad.data.numel()
        size = parameter.grad.data.size()
        flat_tensor[current_index:current_index+numel].add_(parameter.grad.data.view(-1))
        current_index += numel

def unravel_model_params(model, flat_tensor, is_grad, operation, model_weight=1):
    ''' assign flat_tensor to model's parameters '''
    assert model_weight <=1 and model_weight >=0, 'weight coefficient should be in the range of [0,1]'
    current_index = 0
    if is_grad:   
        for parameter in model.parameters():
            numel = parameter.grad.data.numel()
            size = parameter.grad.data.size()
            if operation == 'add':
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.grad.data.mul_(model_weight)
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.grad.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 
    else:
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if operation == 'add':
                parameter.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.data.mul_(model_weight)
                parameter.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 

def unravel_model_buffers(model, flat_tensor, is_grad, operation, model_weight=1):
    ''' assign flat_tensor to model's buffers '''
    assert model_weight <=1 and model_weight >=0, 'weight coefficient should be in the range of [0,1]'
    current_index = 0
    if is_grad:  
        for parameter in model.buffers():
            numel = parameter.grad.data.numel()
            size = parameter.grad.data.size()
            if operation == 'add':
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.grad.data.mul_(model_weight)
                parameter.grad.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.grad.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 
    else:
        for parameter in model.buffers():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if operation == 'add':
                parameter.data.add_(flat_tensor[current_index:current_index+numel].view(size))    
            elif operation == 'mix':
                parameter.data.mul_(model_weight)
                parameter.data.add_(flat_tensor[current_index:current_index+numel].data.mul(1-model_weight).view(size))
            elif operation == 'copy':
                parameter.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            else:
                raise ValueError('No such stupic operation')
            current_index += numel 