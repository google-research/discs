import abc
import ml_collections


class AbstractModel(abs.ABC):
  """Based Model Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def make_init_params(self, rnd):
    pass

  @abc.abstractmethod
  def get_init_samples(self, param, sz):
    pass

  @abc.abstractmethod
  def forward(self, params, x):
    pass

  def get_value_grad(self, params, x):
    pass
