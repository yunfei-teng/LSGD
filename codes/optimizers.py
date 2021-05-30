import torch
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    def __init__(self, params, lr=required, eta=0.001, momentum=0, weight_decay=0):
        ''' Honest implementation of LARS '''
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, eta = eta, momentum=momentum, weight_decay=weight_decay, )
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    def step(self, cur_steps=0, max_steps=1, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr  = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # lambda_l: LARS paper(https://arxiv.org/pdf/1708.03888.pdf)
                # https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
                # https://erickguan.me/2019/pytorch-parallel-model
                #  -- (1) momentum is diffrent (PyTorch has different implementatio of momentum SGD)
                #  -- (2) no global LR decay is employed here
                local_lr  = eta* p.data.norm(2)/(p.grad.data.norm(2)+weight_decay*p.data.norm(2))
                global_lr = lr* (1 - cur_steps / max_steps) ** 2
                actual_lr = global_lr* local_lr
                d_p.mul_(actual_lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                p.data.add_(-buf)

        return loss