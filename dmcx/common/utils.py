"""Utilities."""

import jax


class RngGen(object):
  """Random number generator state utility for Jax."""

  def __init__(self, init_rng):
    self._base_rng = init_rng
    self._counter = 0

  def __iter__(self):
    return self

  def __next__(self):
    return self.advance(1)

  def advance(self, count: int):
    self._counter += count
    return jax.random.fold_in(self._base_rng, self._counter)
