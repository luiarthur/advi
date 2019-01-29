import torch
import math
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt

import advi
from torch.distributions.transforms import StickBreakingTransform

class Gmm(advi.Model):
    """
    y[i] ~ sum_{k=1}^K Normal(y[i] | mu_k, sig_k)
    """
    def __init__(self, K:int, priors=None, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.priors = priors
        self.K = K
        self.sbt = StickBreakingTransform(1)

    def init_vp(self):
        v = {'mu': None, 'sig': None, 'w': None}
        for key in v:
            if key is not 'w':
                v[key] = torch.randn((self.K, 2), device=self.device, dtype=self.dtype)
            else:
                v[key] = torch.randn((self.K - 1, 2), device=self.device, dtype=self.dtype)
            v[key].requires_grad = True

        return v

    def subsample_data(self, data, minibatch_info=None):
        if minibatch_info is None:
            mini_data = data
        else:
            n = minibatch_info['n']
            N = minibatch_info['N']
            # Sampling with replacement is much faster for large N,
            # and doesn't make a practical difference.
            idx = np.random.choice(N, n)
            mini_data = {'y': data['y'][idx]}
        return mini_data

    def sample_real_params(self, v):
        eta = [torch.randn(1, self.K).double() for vj in v]
        return {'mu': eta[0] * torch.exp(v['mu'][:, 1]) + v['mu'][:, 0],
                'sig': eta[1] * torch.exp(v['sig'][:, 1]) + v['sig'][:, 0],
                'w': eta[2][:, 1:] * torch.exp(v['w'][:, 1]) + v['w'][:, 0]}

    def log_q(self, real_params, v):
        def engine(vj, rj):
            m, s = vj[:, 0], torch.exp(vj[:, 1])
            return torch.distributions.Normal(m, s).log_prob(rj).sum()

        return sum([engine(v[key], real_params[key]) for key in v])

    def log_prior(self, real_params):
        if self.priors is None:
            return 0.0
        else:
            lpdfw = torch.distributions.Dirichlet(self.priors['w']).log_prob
            real_w = real_params['w'].squeeze()
            w = self.sbt(real_w)
            lpw = lpdfw(w) + self.sbt.log_abs_det_jacobian(real_w, w)

            lpdfs = torch.distributions.Gamma(self.priors['sig'][0],
                    self.priors['sig'][1]).log_prob
            real_s = real_params['sig'].squeeze()
            lps = advi.transformations.lpdf_logx(real_s, lpdfs).sum()

            lpm = torch.distributions.Normal(self.priors['mu'][0],
                    self.priors['mu'][1]).log_prob(real_params['mu']).sum()

            return lpw + lps + lpm

    def loglike(self, real_params, data, minibatch_info=None):
        sig = torch.exp(real_params['sig'])
        mu = real_params['mu']
        logw = torch.log(self.sbt(real_params['w']))

        # Broadcasting: https://pytorch.org/docs/stable/notes/broadcasting.html
        # mu: 1 x K |  sig: 1 x K | w: 1 x K | y: N x 1
        lpdf = torch.distributions.Normal(mu, sig).log_prob(data['y'])
        ll = torch.logsumexp(logw + lpdf, 1).sum()

        if minibatch_info is not None:
            ll *= minibatch_info['N'] / minibatch_info['n']

        return ll

    def to_real_space(self, params):
        r = dict()
        r['mu'] = params['mu']
        r['sig'] = torch.log(params['sig'])
        r['w'] = self.sbt.inv(params['w'])
        return r

    def to_param_space(self, real_params):
        p = dict()
        p['mu'] = real_params['mu']
        p['sig'] = torch.exp(real_params['sig'])
        p['w'] = self.sbt(real_params['w'])
        return p

    def msg(self, t, v):
        if (t + 1) % 100 == 0:
            d = {'mu': v['mu'][:, 0],
                 'sig': torch.exp(v['sig'][:, 0]),
                 'w': self.sbt(v['w'][:, 0])}
                 #'w': softmax(v['w'], 0)[:, 0]}

            for k in d:
                print('{}: {}'.format(k, d[k].tolist()))
    
if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)

    N = 1000 # works
    # N = 200 # works, if priors are set (for regularizing)

    # NOTE: When N is small, priors are required so that the variational
    # parameters do not go to infinity. Priors are (numerically) powerful in
    # small data settings.

    mu = np.array([3., 1., 2.])
    sig = np.array([.1, .05, .15])
    w = np.array([.5, .3, .2])
    w /= w.sum()
    print("generating data")
    y = []
    for i in range(N):
        k = np.random.choice(3, p=w)
        y.append(np.random.randn() * sig[k] + mu[k])

    y = torch.tensor(y, dtype=torch.float64).reshape(N, 1)
    data = {'y': y}
    K = 3

    # These are good priors. Remember in Gmm.
    # you must provide some prior information
    # on how many groups you believe there are.
    # Otherwise, you might get one big group.
    priors={'w': torch.zeros(K).double() + N*.01/K,
            'sig': torch.tensor([1, 10]).double(),
            'mu': torch.tensor([1.85, 5]).double()}
    mod = Gmm(K=K, priors=priors)
    out = mod.fit(data, lr=1e-1,
                  minibatch_info={'N': N, 'n': 500},
                  niters=10000, nmc=1, seed=1, eps=1e-6, init=None,
                  print_freq=100, verbose=1)

    # ELBO
    elbo = np.array(out['elbo'])
    plt.plot(elbo); plt.show()
    plt.plot((elbo[1:] / elbo[:-1] - 1)); plt.show()

    # Posterior Distributions
    samps = [mod.sample_params(out['vp']) for b in range(1000)]
    mu_samps = torch.stack([s['mu'].squeeze() for s in samps])
    sig_samps = torch.stack([s['sig'].squeeze() for s in samps])
    w_samps = torch.stack([s['w'].squeeze() for s in samps])

    print("Posterior Summary:")
    print('mu mean: {}\nmu sd: {}'.format(mu_samps.mean(0).tolist(),
                                          mu_samps.std(0).tolist()))
    print('sig mean: {}\nsig sd: {}'.format(sig_samps.mean(0).tolist(),
                                            sig_samps.std(0).tolist()))
    print('w mean: {}\nw sd: {}'.format(w_samps.mean(0).tolist(),
                                        w_samps.std(0).tolist()))

