import abc
import torch
import copy
import datetime

class Model(abc.ABC):
    def __init__(self, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device

    @abc.abstractmethod
    def to_param_space(self, real_params):
        """
        transform parameters from real space to original parameter space
        """
        pass

    @abc.abstractmethod
    def to_real_space(self, params):
        """
        transform parameters from original parameter space to real space
        """

    @abc.abstractmethod
    def loglike(self, params, data, minibatch_info=None):
        """
        log likelihood of parameters in original scale

        params: parameters in original parameter space
        data: data as a dictionary
        minibatch_info (dict): information related to minibatch. Used to
                               compute loglike if a minibatch is used.
        """
        pass

    @abc.abstractmethod
    def log_prior(self, real_params):
        """
        log prior of parameters transformed to real space
        this should be the log prior of the orginial parameters + log jacobian
        """
        pass

    @abc.abstractmethod
    def log_q(self, real_params, v):
        """
        log of variational density in real space

        real_params: parameters in real space
        v: variational parameters, in real space
        """
        pass


    @abc.abstractmethod
    def sample_real_params(self, v):
        """
        samples parameters in real space from the variational distributions
        given the variational parameters in real space.
        """
        pass

    @abc.abstractmethod
    def subsample_data(self, data, minibatch_info=None):
        """
        subsample data
        """
        pass

    @abc.abstractmethod
    def init_v(self):
        """
        initialize variational parameters in real space
        """
        pass

    def compute_elbo(self, v, data, minibatch_info=None):
        """
        compute elbo

        v: variational parameters
        data: data (may be a minibatch)
        minibatch_info: Information about minibatch
        """
        real_params = self.sample_real_params(v)
        params = self.to_param_space(real_params)
        ll = self.loglike(params, data, minibatch_info)
        lprior = self.log_prior(real_params)
        lq = self.log_q(real_params, v)
        return ll + lprior - lq

    def compute_elbo_mean(self, data, v, nmc, minibatch_info):
        mini_data = self.subsample_data(data, minibatch_info)
        return torch.stack([self.compute_elbo(v, mini_data, minibatch_info)
                            for i in range(nmc)]).mean()
        
    def fit(self, data, niters:int=1000, nmc:int=10, lr=1e-3, minibatch_info=None, seed:int=1,
            eps=1e-8, init=None, print_freq:int=10, verbose:int=1):

        if init is not None:
            v = copy.deepcopy(init)
        else:
            v = self.init_v()

        optimizer = torch.optim.Adam(v.values(), lr=lr)
        elbo = []

        for t in range(niters):
            elbo_mean = self.compute_elbo_mean(data, v, nmc, minibatch_info)
            loss = -elbo_mean
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            elbo.append(elbo_mean.item())

            if print_freq > 0 and (t + 1) % print_freq == 0:
                now = datetime.datetime.now().replace(microsecond=0)
                if verbose >= 1:
                    print('{} | iteration: {}/{} | elbo mean: {}'.format(now,
                          t + 1, niters, elbo[-1]))
                    
                if verbose >= 2:
                    print('state: {}'.format(v))

            if t > 0 and abs(elbo[-1] / elbo[-2] - 1) < eps:
                print("Convergence suspected. Ending optimizer early.")
                break

        return {'v': v, 'elbo': elbo}

