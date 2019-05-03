import torch
from torch.distributions import (Dirichlet, Categorical)
from plots import scatter_mog
import matplotlib.pyplot as plt

class MultivariateNormal(object):
    def __init__(self, dim):
        self.dim = dim

    def sample(self, B, K, labels):
        raise NotImplementedError

    def log_prob(self, X, params):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError

    def parse(self, raw):
        raise NotImplementedError

class MixtureOfMVNs(object):
    def __init__(self, mvn):
        self.mvn = mvn

    def sample(self, B, N, K, return_gt=False):
        device = 'cpu' if not torch.cuda.is_available() \
                else torch.cuda.current_device()
        pi = Dirichlet(torch.ones(K)).sample(torch.Size([B])).to(device)
        labels = Categorical(probs=pi).sample(torch.Size([N])).to(device)
        labels = labels.transpose(0,1).contiguous()

        X, params = self.mvn.sample(B, K, labels)
        if return_gt:
            return X, labels, pi, params
        else:
            return X

    def log_prob(self, X, pi, params, return_labels=False):
        ll = self.mvn.log_prob(X, params)
        ll = ll + (pi + 1e-10).log().unsqueeze(-2)
        if return_labels:
            labels = ll.argmax(-1)
            return ll.logsumexp(-1).mean(), labels
        else:
            return ll.logsumexp(-1).mean()

    def plot(self, X, labels, params, axes):
        mu, cov = self.mvn.stats(params)
        for i, ax in enumerate(axes.flatten()):
            scatter_mog(X[i].cpu().data.numpy(),
                    labels[i].cpu().data.numpy(),
                    mu[i].cpu().data.numpy(),
                    cov[i].cpu().data.numpy(),
                    ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(hspace=0.1, wspace=0.1)

    def parse(self, raw):
        return self.mvn.parse(raw)
