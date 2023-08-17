"""Abstract Model Class."""

import abc
import ml_collections


class AbstractModel(abc.ABC):
  """Based Model Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def make_init_params(self, rng):
    """Loads or randomly samples the model parameters."""
    pass

  @abc.abstractmethod
  def get_init_samples(self, rng, num_samples):
    """Get initial samples.

    Args:
      rng: jax random seed
      num_samples: sample size.

    Returns:
      Initial samples with batch size of num_samples
    """
    pass

  @abc.abstractmethod
  def forward(self, params, x):
    """Get the energy of batch of samples."""
    pass

  def get_value_and_grad(self, params, x):
    """Get Model Val and Grad."""
    raise NotImplementedError
