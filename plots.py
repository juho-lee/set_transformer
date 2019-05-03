import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

def scatter(X, labels=None, ax=None, colors=None, **kwargs):
    ax = ax or plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    if labels is None:
        ax.scatter(X[:,0], X[:,1], facecolor='k',
                edgecolor=[0.2, 0.2, 0.2], **kwargs)
        return None
    else:
        ulabels = np.sort(np.unique(labels))
        colors = cm.rainbow(np.linspace(0, 1, len(ulabels))) \
                if colors is None else colors
        for (l, c) in zip(ulabels, colors):
            ax.scatter(X[labels==l,0], X[labels==l,1], color=c,
                    edgecolor=c*0.6, **kwargs)
        return ulabels, colors

def draw_ellipse(pos, cov, ax=None, **kwargs):
    if type(pos) != np.ndarray:
        pos = to_numpy(pos)
    if type(cov) != np.ndarray:
        cov = to_numpy(cov)
    ax = ax or plt.gca()
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 6):
        ax.add_patch(Ellipse(pos, nsig*width, nsig*height, angle,
            alpha=0.5/nsig, **kwargs))

def scatter_mog(X, labels, mu, cov, ax=None, colors=None):
    ax = ax or plt.gca()
    ulabels, colors = scatter(X, labels=labels, ax=ax, colors=colors, zorder=10)
    for i, l in enumerate(ulabels):
        draw_ellipse(mu[l], cov[l], ax=ax, fc=colors[i])
