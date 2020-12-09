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
        self._means = nn.parameter.Parameter(mean_init) # / torch.sqrt(torch.sum(mean_init**2, dim=-1, keepdim=True)))
        stds_init = torch.ones(self.n_classes, dim) * np.log(scale)
        self.stds = nn.parameter.Parameter(stds_init)
        
    @property
    def means(self):
        return self._means #/ torch.sqrt(torch.sum(self._means**2, dim=-1, keepdim=True))

    def loss(self, i, j):
        sig1 = torch.exp(self.stds[i])
        sig2 = torch.exp(self.stds[j])
        kl = torch.log(sig2/sig1) \
            + (sig1**2 + (self.means[i]-self.means[j])**2) / (2*sig2**2) \
            - 0.5

        return torch.relu(torch.mean(kl) - self.gamma)

    def get_positive_pair(self):
        # get a class, and get its parent
        # ignore the root class
        i = np.random.randint(1, len(self.classes))
        cls = self.classes[i]
        parent = self.parents[i]
        return cls, parent

    def get_negative_pair(self):
        # randomly sample a class
        i = np.random.randint(0, len(self.classes))

        # randomly sample another class, that is not the parent
        j = self.parents[i]
        p = self.parents[i]
        while j == p or j == i:
            j = np.random.randint(0, len(self.classes))
        return i, j

    def __call__(self):
        # positive pair
        i, j = self.get_positive_pair()

        """pos_pdf_a = torch.distributions.Normal(self.means[i], torch.exp(self.stds[i]))
        pos_pdf_b = torch.distributions.Normal(self.means[j], torch.exp(self.stds[j]))
        term1 = self.div(pos_pdf_a, pos_pdf_b)
        """
        term1 = self.loss(i, j)

        i, j = self.get_negative_pair()
        """neg_pdf_a = torch.distributions.Normal(self.means[i], torch.exp(self.stds[i]))
        neg_pdf_b = torch.distributions.Normal(self.means[j], torch.exp(self.stds[j]))

        term2 = torch.relu(self.margin - self.div(neg_pdf_a, neg_pdf_b))
        """
        term2 = torch.relu(self.margin - self.loss(i, j))
        loss = torch.mean(term1 + term2)

        #print(loss, term1, term2)
        return loss, [loss, term1, term2]


def plot(means, stds): #, ax=None):
    colors = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    ax = plt.gca()
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
            s1, s2 = np.sqrt(np.exp(s))
            c = Ellipse(m, s1, s2, facecolor="none", edgecolor=colors[i], linewidth=5, label=i)
            ax.add_patch(c)

            ss = 1
            ax.set_xlim(0, ss)
            ax.set_ylim(0, ss)

    #ax.clear()
    ax.legend()
    #plt.show()


def train():

    """
       0
      |  \
      1   2
     /|\  |\
    3 4 5 6 7
    """

    #pdf = torch.distributions.Normal(torch.tensor([0., 1]), torch.tensor([0.5, 0.25]))
    #xx = pdf.log_prob(torch.tensor([0.2, 0.6]))
    #print(xx)
    #exit()

    classes = [0, 1, 2, 3, 4, 5, 6, 7]
    parents = [None, 0, 0, 1, 1, 1, 2, 2]
    
    #classes = [0, 1, 2]
    #parents = [None, 0, 1]

    epochs = 10000
    emb = Embeddings(dim=2, scale=1.e-3, margin=10, gamma=1., classes=classes, parents=parents)
    #emb = Embeddings(dim=1, margin=0.1, gamma=0., scale=2.e-1)
    optimizer = torch.optim.Adam([emb._means, emb.stds], lr=0.2)
    def poly_shift(e):
        if e > epochs:
            print('Poly shift got epoch {}, but max epoch is {}, clipping ratio to 1'.format(e, epochs))
        return (1 - min(1, e / epochs)) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[poly_shift])
    
    n = 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def train_and_plot(i):
        #for ii in range(epochs):
        ax.clear()
        if 1==1:
            optimizer.zero_grad()
            loss = 0.
            for j in range(n):
                _loss, prints = emb()
                loss += _loss
            if i % 100 == 0:
                print(i, epochs, prints)
            loss.backward()
            optimizer.step()
            scheduler.step(i)
            
            plot(emb.means.detach().numpy(), emb.stds.detach().numpy()) #, ax=ax)
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

if __name__ == "__main__":
    train()
