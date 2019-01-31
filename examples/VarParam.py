import torch
from torch.distributions import Normal

class VarParam():
    def __init__(self, size, init_m=None, init_log_s=None,
                 dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device

        if init_m is None:
            m = torch.randn(size, dtype=dtype, device=device)
        else:
            m = torch.ones(size, dtype=dtype, device=device) * init_m
        m.requires_grad=True
        self.m = m

        if init_log_s is None:
            log_s = torch.randn(size, dtype=dtype, device=device)
        else:
            log_s = torch.ones(size, dtype=dtype, device=device) * init_log_s
        log_s.requires_grad=True
        self.log_s= log_s

        self.size = size

    def sample(self):
        return torch.randn(self.size, dtype=self.dtype) * torch.exp(self.log_s) + self.m

    def log_prob(self, real_x):
        return Normal(self.m, torch.exp(self.log_s)).log_prob(real_x).sum()
