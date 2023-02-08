import torch 
import pdb

import torch
import torch.nn as nn
import torch.distributions as dists
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as tr
from torch.nn.functional import one_hot
import tensorflow_datasets as tfds


def load_mnist(data_dir ='./', batch_size = 1):
    transform = tr.Compose([tr.Resize(28), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
    train_data = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, 1, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, 1, shuffle=True, drop_last=True)
    return train_loader, test_loader


def load_fmnist(n, data_dir ='./', batch_size = 1 ):
    transform = tr.Compose([tr.Resize(28), tr.ToTensor(),
                            lambda x: one_hot(torch.div((n - 1e-6) * x, 1, rounding_mode='trunc').long().view(-1), n).float()])
    train_data = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.FashionMNIST(data_dir, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader

class binRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, data_mean=None, device=torch.device("cpu")):
        super().__init__()
        self.num_categories = 2
        linear = nn.Linear(n_visible, n_hidden)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device
        self.W = nn.Parameter(linear.weight.data)
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.b_v = nn.Parameter(torch.zeros(n_visible))
        self.data_mean = data_mean
        if data_mean is not None:
            init_val = (data_mean / (1. - data_mean)).log()
            self.b_v.data = init_val
            self.init_dist = dists.Bernoulli(probs=data_mean)
        else:
            self.init_dist = dists.Bernoulli(probs=torch.ones((n_visible,)) * .5)
        self.x0 = self.init_dist.sample((1,)).to(device)

    def logp_v_unnorm(self, v):
        sp = torch.nn.Softplus()(v @ self.W.t() + self.b_h).sum(-1)
        vt = (v * self.b_v).sum(-1)
        return sp + vt

    def forward(self, x):
        return self.logp_v_unnorm(x)

    def trace(self, x):
        return (x - self.x0).abs().sum(-1)


def get_data_mean(dataset):
  data_mean = 0
  num_samples = 0
  for x in dataset:
    img = x['image']._numpy().astype(np.float32)  # pylint: disable=protected-access
    data_mean = data_mean + img
    num_samples += 1
  data_mean = np.array(
      np.reshape(data_mean, [-1]), dtype=np.float32) / num_samples / 255.0
  data_mean = np.clip(data_mean, a_min=0.01, a_max=0.99)
  return data_mean.tolist()

class catRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, n, data_mean=None, device=torch.device("cpu")):
        super().__init__()
        self.num_categories = n
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible, n) / math.sqrt(n_visible * n + 2))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.b_v = nn.Parameter(torch.zeros(n_visible, n))
        self.data_mean = data_mean
        if data_mean is not None:
            init_val = data_mean.log()
            self.b_v.data = init_val
            self.init_dist = dists.Multinomial(probs=data_mean)
        else:
            self.init_dist = dists.Multinomial(probs=torch.ones((n_visible, n)))
        self.x0 = self.init_dist.sample().to(device)

    def forward(self, v):
        sp = nn.Softplus()((v.unsqueeze(-3) * self.W).sum(dim=[-1, -2]) + self.b_h).sum(-1)
        vt = (v * self.b_v).sum([-1, -2])
        return sp + vt

    def trace(self, x):
        return (x - self.x0).abs().sum([-1, -2]) / 2



import tarfile
import numpy as np
import os
import pickle

tar = tarfile.open("./ckpts.ckpt", "r:gz")
hidden_dims = [200, 25]
num_categs = [4, 8]
data_set = ['mnist', 'fashion_mnist']
for i, member in enumerate(tar.getmembers()):
    f = tar.extractfile(member)

    if i in [0, 1]:
        train_loader, test_loader = load_mnist()
    else:
        train_loader, test_loader = load_fmnist(n = num_categs[i%2])
    init_data = []
    for x, _ in train_loader:
        init_data.append(x)
    init_data = torch.cat(init_data, 0)
    data_mean = init_data.mean(0).clamp(.01, .99)

    if i in [0, 1]:
        model = binRBM(n_visible=784, n_hidden=hidden_dims[i], data_mean=data_mean)
    else:
        model = catRBM(n_visible=784, n_hidden=50, n=num_categs[i%2], data_mean=data_mean)
    model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    results = {}

    results['params'] = {'w':model.W.detach().numpy(), 'b_h':model.b_h.detach().numpy(), 'b_v':model.b_v.detach().numpy()}
    results['data_mean'] = model.data_mean.detach().numpy()
    results['num_visible'] = model.n_visible
    results['num_hidden'] = model.n_hidden
    results['num_categories'] = model.num_categories

    if i in [0, 1]:
        path = f'./mnist-2-{hidden_dims[i]}' 
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'rbm.pkl'), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    else:
        path = f'./fashion_mnist-{model.num_categories}-50'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'rbm.pkl'), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    print("Loaded the model")

#import pickle
#from functools import  partial 

#pickle.load = partial(pickle.load, encoding="latin1")
#pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#model = torch.load('./ckpts.ckpt', map_location=lambda storage, loc: storage, pickle_module=pickle)
