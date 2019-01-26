import torch
import math
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt

import advi

class LogisticReg(advi.Model):
    def __init__(self, priors=None, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.priors = priors

    def init_v(self):
        v = {'b0': None, 'b1': None}
        for k in v:
            v[k] = torch.randn(2, device=self.device, dtype=self.dtype)
            v[k].requires_grad = True
        return v

    def subsample_data(self, data, minibatch_info=None):
        if minibatch_info is None:
            mini_data = data
        else:
            n = minibatch_info['n']
            N = minibatch_info['N']
            idx = np.random.choice(N, n, replace=False)
            mini_data = {'x': data['x'][idx], 'y': data['y'][idx]}
        return mini_data

    def sample_real_params(self, v):
        eta = [torch.distributions.Normal(0, 1).sample() for v in v]
        return {'b0': eta[0] * torch.exp(v['b0'][1]) + v['b0'][0],
                'b1': eta[1] * torch.exp(v['b1'][1]) + v['b1'][0]}

    def log_q(self, real_params, v):
        def engine(vj, rj):
            m, s = vj[0], torch.exp(vj[1])
            return torch.distributions.Normal(m, s).log_prob(rj)

        return sum([engine(v[key], real_params[key]) for key in v])


    def log_prior(self, real_params):
        if self.priors is None:
            return 0.0
        else:
            return NotImplemented

    def loglike(self, real_params, data, minibatch_info=None):
        params = self.to_param_space(real_params)
        p = torch.sigmoid(params['b0'] + params['b1'] * data['x'])
        ll = torch.distributions.Bernoulli(p).log_prob(data['y']).sum()
        if minibatch_info is None:
            out = ll
        else:
            out = minibatch_info['N'] * ll / minibatch_info['n']
        return out

    def to_real_space(self, params):
        return params

    def to_param_space(self, real_params):
        return real_params

if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)

    N = 500
    x = np.random.randn(N)
    b0 = .5
    b1 = 2.
    p = 1 / (1 + np.exp(-(b0 + b1 * x)))
    y = (p > np.random.rand(N)) * 1.0

    x = torch.tensor(x)
    y = torch.tensor(y)
    data = {'x': x, 'y': y}
    mod = LogisticReg(priors=None)
    out = mod.fit(data, lr=1e-2,
                  # minibatch_info={'N': N, 'n': 100},
                  # nmc = 1 is good enough in practice
                  niters=5000, nmc=1, seed=2, eps=0, init=None,
                  print_freq=50)

    # ELBO
    elbo = np.array(out['elbo'])
    plt.plot(elbo); plt.show()
    plt.plot(np.abs(elbo[101:] / elbo[100:-1] - 1)); plt.show()

    # Posterior Distributions
    vp = out['v']
    print('b0 mu: {}, sd: {}'.format(vp['b0'][0], torch.exp(vp['b0'][1])))
    print('b1 mu: {}, sd: {}'.format(vp['b1'][0], torch.exp(vp['b1'][1])))

    # R:
    # Coefficients:
    # Estimate Std. Error z value Pr(>|z|)
    # (Intercept)   0.5010     0.1214   4.126 3.69e-05 ***
    # x             2.1769     0.1993  10.923  < 2e-16 ***
