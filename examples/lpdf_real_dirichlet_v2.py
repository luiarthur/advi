import torch
from torch.nn.functional import pad
from torch.distributions import Dirichlet

def logsumexp_plus_0(logx, dim):
    return logx.exp().sum(dim, keepdim=True).log1p()

def softmax_fullrank(r):
    lse = logsumexp_plus_0(r, -1)
    p_reduced = (r - lse).exp()
    return p_reduced

def complete_dim(p):
    dim0 = 1 - p.sum(-1, keepdim=True)
    return torch.cat((p, dim0), dim=-1)


def log_abs_det_jacobian(r, p_short):
    logdetJ = ((1 - p_short).log() + p_short.log()).sum(-1)
    return logdetJ

def lpdf_real_dirichlet(r, conc):
    p_short = softmax_fullrank(r)
    return Dirichlet(conc).log_prob(complete_dim(p_short)) + log_abs_det_jacobian(r, p_short)

r1 = torch.randn((3,5,2), requires_grad=True)
lpdf_real_dirichlet(r1, torch.ones((3,5,3)))

r2 = torch.randn(3, requires_grad=True)
lpdf_real_dirichlet(r2, torch.ones(4))

p1 = complete_dim(softmax_fullrank(r1))
p1.sum(-1)

p2 = complete_dim(softmax_fullrank(r2))
p2.sum(-1)
