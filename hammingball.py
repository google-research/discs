from itertools import product
import math
from math import comb
import sys
import torch
import torch.distributions as dists
import pdb
import numpy as np

class BaseSampler(object):

  def __init__(self, args):
    self.is_binary = args['is_binary']
    self.ess_ratio = args['ess_ratio']
    self.n_steps = args['n_steps']
    self._steps = 0
    self._lens = []
    self._accs = []
    self._hops = []

  def step(self, x, model):
    if self.is_binary:
      return self._step_bin(x, model)
    else:
      return self._step_cat(x, model)

  def _step_bin(self, x, model):
    raise NotImplementedError

  def _step_cat(self, x, model):
    raise NotImplementedError

  def update_stats(self, length, acc, hop):
    self._steps += 1
    self._lens.append(length)
    self._accs.append(acc)
    self._hops.append(hop)

  @property
  def accs(self):
    return self._accs[-1]

  @property
  def hops(self):
    return self._hops[-1]

  @property
  def lens(self):
    return self._lens[-1]

  @property
  def avg_lens(self):
    ratio = self.ess_ratio
    return sum(self._lens[int(self._steps * (1 - ratio)) :]) / (
        int(self._steps * ratio) + 1e-10
    )

  @property
  def avg_accs(self):
    ratio = self.ess_ratio
    return sum(self._accs[int(self._steps * (1 - ratio)) :]) / (
        int(self._steps * ratio) + 1e-10
    )

  @property
  def avg_hops(self):
    ratio = self.ess_ratio
    return sum(self._hops[int(self._steps * (1 - ratio)) :]) / (
        int(self._steps * ratio) + 1e-10
    )


class HammingBallSampler(BaseSampler):
  """current implementation only support R = d(x, u) are shared over all chains in parallel in one M-H step"""

  def __init__(self, args, block_size, hamming):
    super().__init__(args)
    self.block_size = block_size
    self.hamming = hamming
    self.R_init = None
    self._dim = None
    self._pos = None

  def step(self, x, model):
    pdb.set_trace()
    if self._dim is None:
      self._dim = x.shape[1]
      self._pos = 0
      if self.is_binary:
        self.R_init = dists.Categorical(
            torch.tensor(
                [1]
                + [comb(self.block_size, j + 1) for j in range(self.hamming)]
            )
        )
      else:
        self.R_init = dists.Categorical(torch.tensor([1] + [comb(self.block_size, j + 1) * (x.shape[-1] - 1) ** (j + 1) for j in range(self.hamming)]))

    if self.is_binary:
      return self._step_bin(x, model)
    else:
      return self._step_cat(x, model)

  def _step_bin(self, x, model):
    pdb.set_trace()
    
    bsize, d, n = x.shape[0], x.shape[-2], x.shape[-1]
    b_idx = torch.arange(bsize).unsqueeze(-1)
    block = self.get_block()
    R = self.R_init.sample()
    u = x.clone()
    if R:
      choice = torch.multinomial(
          torch.ones_like(x[:, block]), R, replacement=False
      )
      u[b_idx, block[choice]] = 1 - u[b_idx, block[choice]]

    Y = u.unsqueeze(1)
    for j in range(self.hamming):
      y = torch.stack(
          [u.clone() for _ in range(comb(self.block_size, j + 1))], dim=1
      )
      idx_pos = torch.combinations(block, j + 1)
      idx_p = torch.arange(idx_pos.shape[0]).unsqueeze(-1)
      y[:, idx_p, idx_pos] = 1 - y[:, idx_p, idx_pos]
      Y = torch.cat([Y, y], dim=1)

    energy = model(Y)
    selected = torch.multinomial(torch.softmax(energy, dim=-1), 1)
    new_x = Y[b_idx, selected].squeeze()
    hop = torch.abs(new_x - x).mean(0).sum().item()
    self.update_stats(length=2 * self.hamming, acc=1, hop=hop)
    return new_x

  def _step_cat(self, x, model):
    bsize, d, n = x.shape[0], x.shape[-2], x.shape[-1]
    b_idx = torch.arange(bsize).unsqueeze(-1)
    block = self.get_block()
    R = self.R_init.sample()
    u = x.clone()
    if R:
      choice = torch.multinomial(
          torch.ones_like(x[:, block, 0]), R, replacement=False
      )
      posterior = (
          torch.ones_like(u[b_idx, block[choice]]) - u[b_idx, block[choice]]
      )
      dist = dists.Multinomial(probs=posterior)
      u[b_idx, block[choice]] = dist.sample()

    Y = u.unsqueeze(1)
    for j in range(self.hamming):
      y = torch.stack(
          [u.clone() for _ in range(comb(self.block_size, j + 1))], dim=1
      )
      idx_pos = torch.combinations(block, j + 1)
      idx_p = torch.arange(idx_pos.shape[0]).unsqueeze(-1)
      val = y[:, idx_p, idx_pos, :]
      y = torch.stack([y.clone() for _ in range((n - 1) ** (j + 1))], dim=1)
      y[:, :, idx_p, idx_pos, :] = self.get_complementary(val)
      y = y.view(bsize, idx_pos.shape[0] * (n - 1) ** (j + 1), d, n)
      Y = torch.cat([Y, y], dim=1)

    energy = model(Y)
    selected = torch.multinomial(torch.softmax(energy, dim=1), 1)
    new_x = Y[b_idx, selected].squeeze()
    hop = torch.abs(new_x - x).mean(0).sum().item() / 2
    self.update_stats(length=2 * self.hamming, acc=1, hop=hop)
    return new_x

  def get_block(self):
    block = torch.arange(self._pos, self._pos + self.block_size) % self._dim
    self._pos = (self._pos + self.block_size) % self._dim
    return block

  @staticmethod
  def get_complementary(val):
    batch, block, r, n = val.shape
    new_y = torch.zeros((n**r * r, n), device=val.device)
    idx = torch.tensor(list(product(range(n), repeat=r)))
    new_y[torch.arange(n**r * r), idx.view(-1)] = 1
    new_y = new_y.view(1, 1, n**r, r, n).repeat(batch, block, 1, 1, 1)
    diff = torch.abs(new_y - val.unsqueeze(2)).sum(dim=[-1, -2])
    mask = diff == 2 * r
    new_y = new_y[mask].view(batch, block, (n - 1) ** r, r, n)
    return new_y.permute(0, 2, 1, 3, 4)


args = {}
args['is_binary'] = True
args['ess_ratio'] = 0.5
args['n_steps'] = 10000
hb = HammingBallSampler(args, block_size=10, hamming=1)
model = None
x = torch.tensor(np.zeros([10, 1000]))
hb.step(x, model)
