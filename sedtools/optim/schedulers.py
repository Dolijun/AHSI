import math


class LrSchedulerWithWarmup:
    def __init__(self, optimizer, max_iters, base_lr, warmup_steps=0, eta_min=0, anneal_strategy='cos'):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.anneal_strategy = anneal_strategy
        if self.warmup_steps > 0:
            self.warmup_lr_lambda = lambda cur_iter: min(1.0, cur_iter / self.warmup_steps)
        if self.max_iters > self.warmup_steps and self.max_iters > 0:
            assert anneal_strategy in ['cos', 'linear', 'poly'], 'unknown anneal strategy!!'
            if anneal_strategy == 'cos':
                self.lr_lambda = lambda cur_iter: \
                    (1 + math.cos(math.pi * (cur_iter - self.warmup_steps) / (self.max_iters - self.warmup_steps))) / 2
            elif anneal_strategy == 'linear':
                self.lr_lambda = lambda cur_iter: \
                    1 - (1.0 * cur_iter - self.warmup_steps) / (self.max_iters - self.warmup_steps)
            elif anneal_strategy == 'poly':
                self.lr_lambda = lambda cur_iter: \
                    pow(1 - (cur_iter - self.warmup_steps) / (1.0 * self.max_iters - self.warmup_steps), 0.9)

    def step(self, cur_iter):
        if cur_iter <= self.warmup_steps:
            cur_lr = self.warmup_lr_lambda(cur_iter)
        else:
            cur_lr = self.lr_lambda(cur_iter)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr * (
                        self.base_lr * param_group['lr_scale'] - self.eta_min) + self.eta_min

        return cur_lr * (self.base_lr - self.eta_min) + self.eta_min
