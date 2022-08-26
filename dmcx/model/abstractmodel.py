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
    """Get initial samples.
    
    Args:
      param:model parameters.
      sz:sample size.

    Returns:
      Initial samples with batch size of sz
    """
    pass

  @abc.abstractmethods
  def forward(self, params, x):
    pass

  @abc.classmethod
  def get_value_grad(self, params, x):
    pass
