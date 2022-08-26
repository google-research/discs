from abc import ABC, abstractmethod
import ml_collections


class AbstractModel(ABC):
  """Based Model Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abstractmethod
  def make_init_params(self, rnd):
    pass

  @abstractmethod
  def get_init_samples(self, params, sz):
    """Get initial samples.
    
    Args:
      param:model parameters.
      sz:sample size.

    Returns:
      Initial samples with batch size of sz
    """
    pass

  @abstractmethod
  def forward(self, params, x):
    pass

  @classmethod
  def get_value_grad(self, params, x):
    """Get Model Val and Grad."""
    pass
