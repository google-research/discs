from abstractmodel import AbstractModel
import ml_collections
import numpy as np


class Bernouli(AbstractModel):
  """Bernouli Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.p = None
    self.sz = None

  def make_init_params(self, rnd):
    np.random.seed(rnd)
    self.p = np.random.uniform(0, 1)
    return self.p

  def get_init_samples(self, params, sz):
    assert (params is not None)
    self.p = params
    self.sz = sz
    print('sz=', sz)
    return np.random.binomial(size=self.sz, p=self.p, n=1)

  def forward(self, params, x):
    return np.random.binomial(size=self.sz, p=self.p, n=1)


def main() -> None:
  # Simple Script to quickly test outputs.
  cfg = ml_collections.config_dict.ConfigDict()
  b = Bernouli(cfg)
  params = 0.9
  sz = 20
  print('Manual set params=', params)
  x0 = b.get_init_samples(params, sz)
  print(x0)
  assert (x0.shape[0] == sz)

  seed = 50
  params = b.make_init_params(seed)
  print('Returned model init params=', params)
  x0 = b.get_init_samples(params, sz)
  print(x0)
  assert (x0.shape[0] == sz)


if __name__ == '__main__':
  main()
