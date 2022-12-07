"""Utilities."""

import jax
import jax.numpy as jnp


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

def log1mexp(x):
  # Computes log(1-exp(-|x|))
  # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  x = -jnp.abs(x)
  return jnp.where(x > -0.693, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))
