"""Abstract Model Class."""

import abc
import ml_collections


class AbstractModel(abc.ABC):
  """Based Model Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def make_init_params(self, rnd):
    pass

  @abc.abstractmethod
  def get_init_samples(self, rnd, num_samples):
    """Get initial samples.

    Args:
      rnd: jax random seed
      num_samples: sample size.

    Returns:
      Initial samples with batch size of num_samples
    """
    pass

  @abc.abstractmethod
  def forward(self, params, x):
    pass
  
  @abc.abstractmethod
  def get_num_loglike_calls(self):
    pass

  def get_value_and_grad(self, params, x):
    """Get Model Val and Grad."""
    raise NotImplementedError
