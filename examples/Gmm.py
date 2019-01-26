import torch
import math
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt

import advi


class Gmm(advi.Model):
    """
    y[i] ~ sum_{k=1}^K Normal(y[i] | mu_k, sig_k)
    """
    def __init__(self, K:int, priors=None, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.priors = priors
        self.K = K

    def init_v(self):
        v = {'mu': None, 'sig': None, 'w': None}
        for key in v:
            v[key] = torch.randn((self.K, 2), device=self.device, dtype=self.dtype)
            v[key].requires_grad = True

        return v

    def subsample_data(self, data, minibatch_info=None):
        if minibatch_info is None:
            mini_data = data
        else:
            n = minibatch_info['n']
            N = minibatch_info['N']
            idx = np.random.choice(N, n, replace=False)
            mini_data = {'y': data['y'][idx]}
        return mini_data

    def sample_real_params(self, v):
        eta = [torch.randn(1, self.K).double() for vj in v]
        return {'mu': eta[0] * torch.exp(v['mu'][:, 1]) + v['mu'][:, 0],
                'sig': eta[1] * torch.exp(v['sig'][:, 1]) + v['sig'][:, 0],
                'w': eta[2] * torch.exp(v['w'][:, 1]) + v['w'][:, 0]}

    def log_q(self, real_params, v):
        def engine(vj, rj):
            m, s = vj[:, 0], torch.exp(vj[:, 1])
            return torch.distributions.Normal(m, s).log_prob(rj).sum()

        return sum([engine(v[key], real_params[key]) for key in v])


    def log_prior(self, real_params):
        if self.priors is None:
            return 0.0
        else:
            return NotImplemented

    def loglike(self, params, data, minibatch_info=None):
        logw = torch.log(params['w'])

        # Broadcasting: https://pytorch.org/docs/stable/notes/broadcasting.html
        # mu: 1 x K |  sig: 1 x K | w: 1 x K | y: N x 1
        lpdf = torch.distributions.Normal(params['mu'], params['sig']).log_prob(y)
        ll = torch.logsumexp(logw + lpdf, 1).sum()

        # print('here: ', ll2, ll)

        if minibatch_info is None:
            out = ll
        else:
            out = minibatch_info['N'] * ll / minibatch_info['n']
        return out

    def to_real_space(self, params):
        r = dict()
        r['mu'] = params['mu']
        r['sig'] = torch.log(params['sig'])
        r['w'] = advi.trans.invsoftmax(params['w'])
        return r

    def to_param_space(self, real_params):
        p = dict()
        p['mu'] = real_params['mu']
        p['sig'] = torch.exp(real_params['sig'])
        p['w'] = torch.softmax(real_params['w'], 1)
        return p

    def msg(self, t, v):
        if (t + 1) % 100 == 0:
            d = {'mu': v['mu'][:, 0],
                 'sig': torch.exp(v['sig'][:, 0]),
                 'w': torch.softmax(v['w'], 0)[:, 0]}

            for k in d:
                print('{}: {}'.format(k, d[k].tolist()))
    
if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)

    N = 1000

    mu = np.array([3., 1., 2.])
    sig = np.array([.1, .05, .15])
    w = np.array([.3, .6, .1])
    w /= w.sum()
    y = []
    for i in range(N):
        k = np.random.choice(3, p=w)
        y.append(np.random.randn() * sig[k] + mu[k])

    y = torch.tensor(y, dtype=torch.float64).reshape(N, 1)
    data = {'y': y}
    mod = Gmm(K=3)
    out = mod.fit(data, lr=1e-1,
                  minibatch_info={'N': N, 'n': 300},
                  niters=1000, nmc=10, seed=0, eps=1e-6, init=None,
                  print_freq=100, verbose=1)

    # ELBO
    elbo = np.array(out['elbo'])
    plt.plot(elbo); plt.show()
    plt.plot(np.abs(elbo[101:] / elbo[100:-1] - 1)); plt.show()


    # Posterior Distributions
    # vp = out['v']
    # print('b0 mu: {}, sd: {}'.format(vp['b0'][0], torch.exp(vp['b0'][1])))
    # print('b1 mu: {}, sd: {}'.format(vp['b1'][0], torch.exp(vp['b1'][1])))

