import abc
import torch
import copy
import datetime
import math

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
    def loglike(self, real_params, data, minibatch_info=None):
        """
        log likelihood of parameters in original space

        params: parameters in original parameter space
        data: data as a dictionary
        minibatch_info (dict): information related to minibatch. Used to
                               compute loglike if a minibatch is used.
        real_params: parameteris in real space. defaults to None.
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
    def log_q(self, real_params, vp):
        """
        log of variational density in real space

        real_params: parameters in real space
        vp: variational parameters, in real space
        """
        pass


    @abc.abstractmethod
    def sample_real_params(self, vp):
        """
        samples parameters in real space from the variational distributions
        given the variational parameters in real space.
        """
        pass

    def sample_params(self, vp):
        """
        Samples parameters in parameter space given the variational parameters
        in real space. While this should not be costly, this should not be done
        until the end of the ADVI, when one desires to obtain posterior
        samples.
        """
        return self.to_param_space(self.sample_real_params(vp))

    @abc.abstractmethod
    def subsample_data(self, data, minibatch_info=None):
        """
        subsample data
        """
        pass

    @abc.abstractmethod
    def init_vp(self):
        """
        initialize variational parameters in real space
        """
        pass

    def compute_elbo(self, vp, data, minibatch_info=None):
        """
        compute elbo

        vp: variational parameters
        data: data (may be a minibatch)
        minibatch_info: Information about minibatch
        """
        real_params = self.sample_real_params(vp)
        params = self.to_param_space(real_params)
        ll = self.loglike(real_params=real_params, data=data,
                          minibatch_info=minibatch_info)
        lprior = self.log_prior(real_params)
        lq = self.log_q(real_params, vp)
        return ll + lprior - lq

    def compute_elbo_mean(self, data, vp, nmc, minibatch_info):
        """
        Compute the mean of the elbo via Monte Carlo integration
        
        The number of MC samples (nmc) can be as little as 1 in practice.
        But for some models, nmc=2 may be necessary. nmc >= 10 could be
        overkill for most if not all problems.

        data: data
        vp: variational parameters (real space)
        nmc: number of MC samples
        minibatch_info: information about minibatch
        """
        mini_data = self.subsample_data(data, minibatch_info)
        return torch.stack([self.compute_elbo(vp, mini_data, minibatch_info)
                            for i in range(nmc)]).mean()

    def msg(self, t, vp):
        """
        an optional message to print at the end of each iteration.

        t: iteration number
        vp: variational parameters (real space)
        """
        pass

    def vp_as_list(self, vp):
        return vp.values()
        
    def fit(self, data, niters:int=1000, nmc:int=2, lr:float=1e-2,
            minibatch_info=None, seed:int=1, eps:float=1e-6, init=None,
            print_freq:int=10, verbose:int=1):
        """
        fir the model.

        data: data
        niter: max number of iterations for ADVI
        nmc: number of MC samples for estimating ELBO mean (default=2). nmc=1
             is usually sufficient. nmc >= 2 may be required for noisy gradients.
             nmc >= 10 is overkill in most situations.
        lr: learning rate (> 0)
        minibatch_info: information on minibatches
        seed: random seed for torch (for reproducibility)
        eps: threshold for determining convergence. If `abs((elbo_curr /
             elbo_prev) -1) < eps`, then ADVI exits before `niter` iterations.
        init: initial values for variational parameters (in real space). This has
              the same for as the output.
        print_freq: how often to print ELBO value during algorithm. For monitoring
                    status of ADVI. (default=10, i.e. print every 10 iterations.)
        verbose: an integer indicating how much output to show. defaults to 1, 
                 which prints the ELBO. Setting verbose=0 will turn off all outputs.

        returns: dictionary with keys 'v' and 'elbo', where 'v' is the
                 variational parameters in real space, and 'elbo' is the 
                 ELBO history.
        """

        assert(nmc >= 1)
        assert(lr > 0)
        assert(eps >= 0)


        if init is not None:
            vp = copy.deepcopy(init)
        else:
            vp = self.init_vp()
        
        vp_list = self.vp_as_list(vp)

        optimizer = torch.optim.Adam(vp_list, lr=lr)
        elbo = []

        for t in range(niters):
            elbo_mean = self.compute_elbo_mean(data, vp, nmc, minibatch_info)
            loss = -elbo_mean
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            elbo.append(elbo_mean.item())

            if print_freq > 0 and (t + 1) % print_freq == 0:
                now = datetime.datetime.now().replace(microsecond=0)
                if verbose >= 1:
                    print('{} | iteration: {}/{} | elbo: {}'.format(now,
                          t + 1, niters, elbo[-1]))
                    
                if verbose >= 2:
                    print('state: {}'.format(vp))

                self.msg(t, vp)

            if t > 0 and abs(elbo[-1] / elbo[-2] - 1) < eps:
                print("Convergence suspected. Ending optimizer early.")
                break

            if math.isnan(elbo[-1]):
                print("nan detected! Exiting optimizer early.")
                break

        return {'vp': vp, 'elbo': elbo}

