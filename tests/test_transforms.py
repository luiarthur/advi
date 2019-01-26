import unittest

import torch
import advi.transformations as trans


class Test_Transformations(unittest.TestCase):
    def approx(self, a, b, eps=1E-8):
        self.assertTrue(abs(a - b) < eps)

    def test_bounded(self):
        p = torch.tensor(.5)
        self.assertTrue(trans.logit(p) == 0)

        q = torch.tensor(10.0)
        self.approx(trans.logit(q, 5, 16), -.1823216, 1e-5)
        self.approx(trans.invlogit(trans.logit(q, 5, 16), 5, 16), q, 1e-5)
        
    def test_simplex(self):
        p_orig = [.1, .3, .6]
        p = torch.tensor(p_orig)
        x = trans.invsoftmax(p)
        p_tti = torch.softmax(x, 0)
        self.assertTrue(p_tti.sum() == 1)
        for j in range(len(p_orig)):
            self.approx(p_tti[j].item(), p_orig[j], 1e-6)

    def test_lpdf_logx(self):
        gam = torch.distributions.gamma.Gamma(2, 3)
        x = torch.tensor(3.)
        z = trans.lpdf_logx(torch.log(x), gam.log_prob)
        print(z)
        self.approx(z, -4.6055508, eps=1e-6)
        
    def test_lpdf_logitx(self):
        beta = torch.distributions.beta.Beta(2, 3)
        x = torch.tensor(.6)
        z = trans.lpdf_logitx(trans.logit(x), beta.log_prob,
                              a=torch.tensor(0.), b=torch.tensor(1.))
        print(z)
        self.approx(z, -1.285616793366446, eps=1e-6)

    def test_lpdf_real_dirichlet(self):
        # This tests if the dirichlet in the two-dimensional case
        # (which is essentially a beta) works properly.
        # TODO: Higher dimensional cases are harder to check, but should
        # be done eventually.
        alpha = torch.tensor([2., 3.])
        dirichlet = torch.distributions.dirichlet.Dirichlet(alpha)
        p = torch.tensor([.6, .4])
        r = trans.invsoftmax(p)
        z = trans.lpdf_real_dirichlet(r, dirichlet.log_prob)
        print(z)
        self.approx(z, -1.285616793366446, eps=1e-6)
 

if __name__ == '__main__':
    unittest.main()
