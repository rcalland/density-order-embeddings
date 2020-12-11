import random
import numpy as np
from scipy import stats

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch import nn

"""
class Hierarchy(object):

    def __init__(self):
        self.n_leaf_classes = 4
        self.classes = [(0, 'person', [4, 5, 6]), (1, 'rider', [6, 7, 4]), (2, 'car', [3, 7, 7]), (3, 'truck', [8, 9, 4]),
                        (-1, 'ignored', [6])]

        self.ancestor_classes = ['human', 'vehicle']
        #is_thing = [True, True, True, True]
        dataset_hierarchy = []
        dataset_hierarchy.append([((0, 1), 'human'), ((2, 3), 'vehicle')])
        hierarchy_height = len(dataset_hierarchy)

        self.parent_dict = {i: None for i in range(self.n_leaf_classes + len(self.ancestor_classes))}
        for c, ps in hierarchy.create_parent_dict(self.n_leaf_classes, dataset_hierarchy, hierarchy_height).items():
            self.parent_dict[c] = ps[0]

    def get_positive_pair(self):
            print(self.parent_dict)
"""


def kltorch(mu1, mu2, sig1, sig2, epsilon=1e-12, batch=False):
    # This is KL(1 || 2)
    if batch:
      mu1_ = mu1[:,None,:]
      sig1_ = sig1[:,None,:]
      mu2_ = mu2[None,:,:]
      sig2_ = sig2[None,:,:]
    else:
      mu1_, mu2_, sig1_, sig2_ = mu1, mu2, sig1, sig2
    """orig_numpy = False
    if type(mu1_).__module__ == np.__name__:
      mu1_ = torch.from_numpy(mu1_)
      mu2_ = torch.from_numpy(mu2_)
      sig1_ = torch.from_numpy(sig1_)
      sig2_ = torch.from_numpy(sig2_)
      orig_numpy = True
    """
    diff = mu1_/mu1_.norm(p=2, dim=-1, keepdim=True) - mu2_/mu2_.norm(p=2, dim=-1, keepdim=True)
    sig2_inv = 1./(epsilon + sig2_)
    res = torch.sum(diff*sig2_inv*diff, dim=-1)
    res += torch.sum(sig1_*sig2_inv, dim=-1)
    res -= torch.sum(torch.log(sig1_) - torch.log(sig2_), dim=-1)
    res -= mu1.shape[-1]
    """if orig_numpy:
      return res.cpu().numpy()
    else:"""
    
    return res


class Embeddings(nn.Module):

    def __init__(self, batchsize=1, dim=1, classes=[0, 1, 2], parents=[None, 0, 1], scale=1., margin=10., gamma=3):
        super().__init__()
        assert len(classes) == len(parents)

        self.dim = dim
        self.n_classes = len(classes)
        self.margin = margin
        self.gamma = gamma
        self.classes = classes
        self.parents = parents
        mean_init = torch.rand(self.n_classes, dim)
        mean_init = mean_init / torch.sqrt(torch.sum(mean_init**2, dim=(0,1), keepdim=True))
        self._means = nn.parameter.Parameter(mean_init) # / torch.sqrt(torch.sum(mean_init**2, dim=-1, keepdim=True)))
        stds_init = torch.ones(self.n_classes, dim) * np.log(scale)
        self.stds = nn.parameter.Parameter(stds_init)
    
        
    @property
    def means(self):
        return self._means / torch.sqrt(torch.sum(self._means**2, dim=0, keepdim=True))

    def loss(self, i, j, gamma):
        sig1 = torch.exp(self.stds[i])
        sig2 = torch.exp(self.stds[j])
        kl = kltorch(self.means[i], self.means[j], sig1, sig2)
        #print(kl)
        
        """kl = torch.log(sig2/sig1) \
            + (sig1**2 + (self.means[i]-self.means[j])**2) / (2*sig2**2) \
            - 0.5
        kl = torch.sum(kl)"""
        return torch.relu(kl - gamma)

    def is_encapsulated(self, a, b):
        # return true if a is encapsulated by b
        # e.g. b is the parent of a
        # e.g. b an ancestor of a

        p = a
        while True:
            parent = self.parents[p]
            current = self.classes[p]
            if parent == b:
                return True
            if parent == None:
                return False
            p = parent
            
    def get_positive_pair(self):
        # get a class, and get its parent
        # ignore the root class
        i = np.random.randint(1, len(self.classes))
        cls = self.classes[i]
        parent = self.parents[i]
        return cls, parent

    def get_negative_pair(self):
        methods = ("random", "swap")
        neg_sample_method = random.choices(methods, weights=[1., 0])
        
        if "swap" in neg_sample_method:
            i = np.random.randint(1, len(self.classes))
            j = self.parents[i]
            return j, i
        if "random" in neg_sample_method:
            # randomly sample a class
            i = np.random.randint(0, len(self.classes))
            j = np.random.randint(0, len(self.classes))

            while self.is_encapsulated(i, j) or i==j:
                j = np.random.randint(0, len(self.classes))
            
            assert i!=j, "cannot be equal"
            return i, j

    def __call__(self):
        # positive pair
        pi, pj = self.get_positive_pair()

        term1 = self.loss(pi, pj, gamma=self.gamma)

        ni, nj = self.get_negative_pair()
        #if ni in (1,6) and nj in (1,6):
        #print("pos pair {} {} neg pair {} {}".format(pi, pj, ni, nj))

        term2 = self.loss(ni, nj, gamma=self.gamma)
        loss = term1 + torch.relu(self.margin - term2)

        #if ni in (1,2) and nj in (1,2):
        #    print("middle layer", term1, term2)
        
        return loss, [loss.item(), term1.item(), term2.item()]


def plot(means, stds, clear_axis=True):
    colors = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    ax = plt.gca()
    if clear_axis:
        ax.clear()

    #if ax is None:
    #    fig, ax = plt.subplots()
    #ax.clear()
    for i, (m, s) in enumerate(zip(means, stds)):
        #print(i, m, s)
        if len(m) == 1:
            x = np.arange(0., 1, 0.001)
            p = stats.norm(m, np.sqrt(np.exp(s)))
            ax.plot(x, p.pdf(x), label=i, linewidth=7)

        else:
            s1, s2 = 5*np.sqrt(np.exp(s))
            c = Ellipse(m, s1, s2, facecolor="none", edgecolor=colors[i], linewidth=5, label=i)
            ax.add_patch(c)

            ss = 3
            ax.set_xlim(-ss, ss)
            ax.set_ylim(-ss, ss)

    #ax.clear()
    ax.legend()
    #plt.show()


def recover_hierarchy(emb):
    n = 100000
    xy = (np.random.uniform(size=[n, 2]) - 0.5) * 2 * 3

    means = emb.means.detach().numpy()
    stds = emb.stds.detach().numpy()

    probs = []
    for m, s in zip(means, stds):
        cov = np.eye(2) * np.sqrt(np.exp(s.T))
        pdf = stats.multivariate_normal(mean=m, cov=cov)
        probs.append(pdf.pdf(xy)[None])
        
    probs = np.concatenate(probs, axis=0)
    preds = np.argmax(probs, axis=0)
    print(probs.shape, preds.shape)

    #fig2 = plt.figure()
    ax = plt.gca()
    scattered = ax.scatter(xy[:,0], xy[:,1], c=preds)
    plt.colorbar(scattered)
    #plt.show()


def train():

    """
       0
      |  \
      1   2
     /|\  |\
    3 4 5 6 7
    """
    classes = [0, 1, 2, 3, 4, 5, 6, 7]
    parents = [None, 0, 0, 1, 1, 1, 2, 2]

    """
     0
     |
     1
     |
     2
    """
    #classes = [0, 1, 2]
    #parents = [None, 0, 1]

    """
    0
    | \
    1  2
    """
    #classes = [0, 1, 2]
    #parents = [None, 0, 0]

    """
    0
    |
    1
    """
    #classes = [0, 1]
    #parents = [None, 0]
    
    
    epochs = 1000
    emb = Embeddings(dim=2, scale=1.e-2, margin=1000., gamma=40, classes=classes, parents=parents)
    #emb = Embeddings(dim=1, margin=0.1, gamma=0., scale=2.e-1)
    optimizer = torch.optim.Adam([emb._means, emb.stds], lr=0.1)
    def poly_shift(e):
        if e > epochs:
            print('Poly shift got epoch {}, but max epoch is {}, clipping ratio to 1'.format(e, epochs))
        return (1 - min(1, e / epochs)) ** 0.9
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[poly_shift])

    n = 32

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def train_and_plot(i, draw_plot=True, clear_axis=True):
        #for ii in range(epochs):
        if clear_axis:
            ax.clear()
        if 1==1:
            optimizer.zero_grad()
            loss = 0.
            for j in range(n):
                _loss, prints = emb()
                loss += _loss

            loss /= n
            if i % 100 == 0:
                print(i, epochs, prints)
            loss.backward()
            optimizer.step()
            #scheduler.step(i)

            if draw_plot:
                plot(emb.means.detach().numpy(), emb.stds.detach().numpy(), clear_axis=clear_axis)
            """for iii, (m, s) in enumerate(zip(emb.means.detach().numpy(), emb.stds.detach().numpy())):
                print(i, m, s)
                if len(m) == 1:
                    x = np.arange(-5, 5, 0.1)
                    p = stats.norm(m, np.sqrt(np.exp(s)))
                    #print("HLOE")
                    #ax = plt.gca()
                    #ax.clear()
                    ax.plot(x, p.pdf(x), label=iii, linewidth=7)
            """

    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, train_and_plot, frames=epochs, interval=1)
    plt.show()

    """
    for i in range(epochs):
        train_and_plot(i, draw_plot=False)

    recover_hierarchy(emb)
    train_and_plot(i, draw_plot=True, clear_axis=False)
    plt.show()
    """
    
    
if __name__ == "__main__":
    train()
