import torch

# Transforms from support space to Real space
def logit(p, a=torch.tensor(0.0), b=torch.tensor(1.0)):
    """
    for scalar parameters with bounded support (no gaps)
    basically a logit transform
    """
    return torch.log(p - a) - torch.log(b - p)

def invsoftmax(p):
    """
    Basically for transforming a Dirichlet to real space
    """
    return torch.log(p) - torch.log(p.max())

# Transforms from to Real space to support space
def invlogit(x, a=torch.tensor(0.0), b=torch.tensor(1.0)):
    """
    sigmoid
    """
    u = torch.sigmoid(x) 
    return (b - a) * u + a

### Density transformations
def lpdf_logx(logx, lpdf_x):
    x = torch.exp(logx)
    return lpdf_x(x) + logx

def lpdf_logitx(logitx, lpdf_x, a=torch.tensor(0.0), b=torch.tensor(1.0)):
    x = invlogit(logitx, a, b) 
    p = (x - a) / (b - a)
    return lpdf_x(x) + torch.log(b - a) + torch.log(p) + torch.log(1 - p)

def lpdf_real_dirichlet(r, lpdf_p):
    """
    Remember to perform: r -= r.max() when before zeroing out the gradient
    """

    # The rank is the dim(r) - 1, since the remaining parameter
    # must be one minus the sum of the other parameters.
    rank = r.size().numel() - 1
    J = torch.empty([rank, rank], dtype=torch.float64)
    p = torch.softmax(r, 0) 

    for i in range(rank):
        for j in range(i + 1):
            if i == j:
                # J[i, j] = torch.exp(x[i]) * (sum_x - torch.exp(x[j])) / (sum_x ** 2)
                J[i, i] = p[i] * (1 - p[i])
            else:
                # tmp = torch.exp(x[i] + x[j]) / (sum_x ** 2)
                tmp = -p[i] * p[j]
                J[i, j] = tmp
                J[j, i] = tmp

    return lpdf_p(p) + torch.logdet(J)
