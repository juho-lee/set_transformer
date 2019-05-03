from mixture_of_mvns import MultivariateNormal
import torch
import torch.nn.functional as F
import math

class MultivariateNormalDiag(MultivariateNormal):
    def __init__(self, dim):
        super(MultivariateNormalDiag, self).__init__(dim)

    def sample(self, B, K, labels):
        N = labels.shape[-1]
        device = labels.device
        mu = -4 + 8*torch.rand(B, K, self.dim).to(device)
        sigma = 0.3*torch.ones(B, K, self.dim).to(device)
        eps = torch.randn(B, N, self.dim).to(device)

        rlabels = labels.unsqueeze(-1).repeat(1, 1, self.dim)
        X = torch.gather(mu, 1, rlabels) + \
                eps * torch.gather(sigma, 1, rlabels)
        return X, (mu, sigma)

    def log_prob(self, X, params):
        mu, sigma = params
        dim = self.dim
        X = X.unsqueeze(2)
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        diff = X - mu
        ll = -0.5*math.log(2*math.pi) - sigma.log() - 0.5*(diff.pow(2)/sigma.pow(2))
        return ll.sum(-1)

    def stats(self, params):
        mu, sigma = params
        I = torch.eye(self.dim)[(None,)*(len(sigma.shape)-1)].to(sigma.device)
        cov = sigma.pow(2).unsqueeze(-1) * I
        return mu, cov

    def parse(self, raw):
        pi = torch.softmax(raw[...,0], -1)
        mu = raw[...,1:1+self.dim]
        sigma = F.softplus(raw[...,1+self.dim:])
        return pi, (mu, sigma)
