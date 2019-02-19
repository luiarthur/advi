import torch
import math
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt

import advi
from torch.distributions.transforms import StickBreakingTransform
from VarParam import VarParam

class Gmm(advi.Model):
    """
    y[i] ~ sum_{k=1}^K Normal(y[i] | mu_k, sig_k)
    """
    def __init__(self, K:int, N:int, priors=None, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.priors = priors
        self.K = K
        self.N = N
        self.sbt = StickBreakingTransform(0)

    def init_vp(self):
        return {'mu': VarParam((1, self.K)),
                'sig': VarParam((1, self.K), init_m=0.0, init_log_s=-2),
                'w': VarParam((1, self.K - 1), init_m=0.0, init_log_s=-2)}

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

    def sample_real_params(self, vp):
        real_params = {}
        for key in vp:
            real_params[key] = vp[key].sample()
        return real_params

    def log_q(self, real_params, vp):
        out = 0.0
        for key in vp:
            out += vp[key].log_prob(real_params[key]).sum()
        return out / self.N

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

            return (lpw + lps + lpm) / self.N

    def loglike(self, real_params, data, minibatch_info=None):
        sig = torch.exp(real_params['sig'])
        mu = real_params['mu']
        logw = torch.log(self.sbt(real_params['w']))

        # Broadcasting: https://pytorch.org/docs/stable/notes/broadcasting.html
        # mu: 1 x K |  sig: 1 x K | w: 1 x K | y: N x 1
        lpdf = torch.distributions.Normal(mu, sig).log_prob(data['y'])
        return torch.logsumexp(logw + lpdf, 1).mean()

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

    def vp_as_list(self, vp):
        return [v.m for v in vp.values()] + [v.log_s for v in vp.values()]

    def msg(self, t, v):
        # if (t + 1) % 100 == 0:
        if False:
            d = {'mu': v['mu'][:, 0],
                 'sig': torch.exp(v['sig'][:, 0]),
                 'w': self.sbt(v['w'][:, 0])}
                 #'w': softmax(v['w'], 0)[:, 0]}

            for k in d:
                print('{}: {}'.format(k, d[k].tolist()))
    
if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)

    # N = 20000 # works
    N = 10000 # works
    # N = 200 # works

    mu = np.array([3., 1., 2.])
    sig = np.array([.1, .05, .15])
    w = np.array([.5, .3, .2])
    w /= w.sum()
    print("generating data")
    k = np.random.choice(3, p=w, size=N)
    y = np.random.randn(N) * sig[k] + mu[k]

    y = torch.tensor(y, dtype=torch.float64).reshape(N, 1)
    data = {'y': y}
    K = 3

    priors={'w': torch.ones(K).double() / K,
            'sig': torch.tensor([1, 10]).double(),
            'mu': torch.tensor([1.85, 5]).double()}
    mod = Gmm(K=K, N=N, priors=priors)
    out = mod.fit(data, lr=1e-1,
                  minibatch_info={'N': N, 'n': 500},
                  niters=1000, nmc=1, seed=1, eps=1e-6, init=None,
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

