"""Abstract Sampler Class."""

import abc
import jax.numpy as jnp
import ml_collections


class AbstractSampler(abc.ABC):
  """Base Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def step(self, model, rng, x, model_param, state, x_mask=None):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rng: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler.
      x_mask: (optional) broadcast to x, masking out certain dimensions.

    Returns:
      new_x: new sample.
      new_state: new sampler state.
      acc: acceptance ratio.
    """
    pass

  def make_init_state(self, rng):
    """Init sampler state."""
    _ = rng
    return {'num_ll_calls': jnp.zeros(shape=(), dtype=jnp.int32),
            'steps': jnp.zeros(shape=(), dtype=jnp.int32)}

  def update_sampler_state(self, sampler_state):
    sampler_state['steps'] += 1
    return sampler_state
