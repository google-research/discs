from abstractmodel import AbstractModel
import ml_collections
import numpy as np


class Bernouli(AbstractModel):
  """Bernouli Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

  def make_init_params(self, rnd):
    np.random.seed(rnd)
    return np.random.uniform(0, 1)

  def get_init_samples(self, params, sz):
    assert (params is not None)
    assert (sz >= 1)
    return np.random.binomial(size=sz, p=params, n=1)

  def forward(self, params, x):
    return np.array(x == 1) * params + np.array(x == 0) * (1-params)