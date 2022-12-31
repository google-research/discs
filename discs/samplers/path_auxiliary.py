"""Path auxiliary sampler."""

from discs.common import math
from discs.samplers import locallybalanced
import jax
import jax.numpy as jnp
import ml_collections


class PAS(locallybalanced.LocallyBalancedSampler):
  
  
