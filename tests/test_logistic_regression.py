import unittest

import torch
import math
import copy
import datetime

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

class Test_LogisticReg(unittest.TestCase):
    torch.manual_seed(1)

    N = 50
    x = torch.randn(N, dtype=torch.float64)
    b0 = .5
    b1 = 2.
    p = 1 / (1 + torch.exp(-(b0 + b1 * x)))
    y = (p > torch.rand(N, dtype=torch.float64)) * 1.0

    data = {'x': x, 'y': y.double()}

    mod = LogisticReg(priors=None)
    out = mod.fit(data, lr=1e-2,
                  # minibatch_info={'N': N, 'n': 100},
                  niters=100, nmc=10, seed=2, eps=1e-6, init=None,
                  print_freq=50)

    def test_logistic_compiled(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
