import abc
import ml_collections


class AbstractSampler(abs.ABC):
  """Base Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def step(self, x, model, model_param, s, rnd):
    n_x = None
    n_s = None
    return n_x, n_s
