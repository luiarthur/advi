import time

import torch
from torch.distributions import Dirichlet
from torch.distributions import Normal

torch.manual_seed(0)

# Data dimensions
N = [3, 1, 2]
I = len(N)
for i in range(I):
    N[i] *= 10000
J = 5
K = 4
L1 = 3
L0 = 6

# Create parameters of specified dimensions
y = [torch.randn(N[i], J, 1, 1) for i in range(I)]

mu1 = torch.randn(1, 1, L1, 1); mu1.requires_grad=True
eta1 = [torch.stack([Dirichlet(torch.ones(L1)).sample()
        for j in range(J)]).reshape(1, J, L1, 1) for i in range(I)]

mu0 = torch.randn(1, 1, L0, 1); mu0.requires_grad=True
eta0 = [torch.stack([Dirichlet(torch.ones(L0)).sample()
        for j in range(J)]).reshape(1, J, L0, 1) for i in range(I)]

for i in range(I):
    eta0[i].requires_grad=True
    eta1[i].requires_grad=True

sig = torch.rand(I); sig.requires_grad=True


v = torch.rand(K).reshape(1, 1, K) # Ni x J x K (no L)
v.requires_grad=True
# Z = (torch.rand(J, K) > .5).reshape(1, J, K) # Ni x J x K (no L)

w = [Dirichlet(torch.ones(K)).sample().reshape(1, K) for i in range(I)]
for i in range(I):
    w[i].requires_grad=True

# Time the execution
tic = time.time()
i = 0
d0 = Normal(mu0, sig[i]).log_prob(y[i]) + torch.log(eta0[i])
d1 = Normal(mu1, sig[i]).log_prob(y[i]) + torch.log(eta1[i])

a0 = torch.logsumexp(d1, 2) # Ni x J x K
a1 = torch.logsumexp(d0, 2) # Ni x J x K

c = (a1 * torch.log(v) + a0 * torch.log(1-v)).sum(1) # Ni x K

f = c + torch.log(w[i])
g = torch.logsumexp(f, 1)
ll = g.sum()
print(ll)
tic = time.time(); ll.backward(); print('backward time: {}'.format(time.time() - tic));

toc = time.time() - tic
print('Elapsed time for computation: {}'.format(toc))

# Imputing missing values.
# https://discuss.pytorch.org/t/requires-gradient-on-only-part-of-a-matrix/5680/2

tic = time.time(); bla = y[0] > 0; print(time.time() - tic);

# For variational parameters of y, just do
mi = (torch.rand(N[i], J) > .5).float()
yi_imp = torch.randn(N[i], J)
(mi * Normal(0, 1).log_prob(yi_imp)).sum()

A = torch.zeros(3, 5)
B = torch.ones(3, 5)
m = torch.randn(3, 5) > 0
A[m] = B[m]
