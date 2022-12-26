"""Abstract Sampler Class."""

import abc
import jax.numpy as jnp
import ml_collections


class AbstractSampler(abc.ABC):
  """Base Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    pass

  @abc.abstractmethod
  def step(self, model, rng, x, model_param, state):
    pass

  def make_init_state(self, rng):
    """Init sampler state."""
    _ = rng
    return {'num_ll_calls': jnp.zeros(shape=(), dtype=jnp.int32)}
