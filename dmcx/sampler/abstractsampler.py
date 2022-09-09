"""Abstract Sampler Class."""

import abc
import ml_collections


class AbstractSampler(abc.ABC):
  """Base Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def step(self, model, rnd, x, model_param, state):
    pass

  @abc.abstractmethod
  def make_init_state(self, rnd):
    pass
