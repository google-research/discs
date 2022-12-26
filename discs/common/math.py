"""Common math func impl in Jax."""

import functools
import jax
import jax.numpy as jnp


def gumbel(rng, loc):
  uniform_sample = jax.random.uniform(
      rng, shape=loc.shape, minval=0.0, maxval=1.0
  )
  return loc -jnp.log(-jnp.log(uniform_sample))


def categorical_without_replacement(rng, log_prob, k):
  """Categorical sampling without replacement.

  Args:
    rng: random generator.
    log_prob: log probability in the shape of [..., num_categories];
    k: number of samples
  Returns:
    pass
  """
  num_categories = log_prob.shape[-1]
  perturbed_prob = gumbel(rng, loc=log_prob)
  idx_sampled = jnp.argsort(-perturbed_prob)
  indices = jnp.expand_dims(jnp.arange(num_categories, dtype=jnp.int32),
                            axis=list(range(k.ndim)))
  mask = indices < k
  return idx_sampled, mask


def multinomial(rng, log_prob, num_samples,
                replacement=True, is_nsample_const=True):
  """Multinomial sample.

  Args:
    rng: random generator;
    log_prob: log probability in the shape of [..., num_categories];
    num_samples: number of samples;
    replacement: sampling with/without replacement
    is_nsample_const: is num_samples a constant int or a tensor?
  Returns:
    sampled indices, in the form of either a binary indicator matrix (
    is_nsample_const=False), or an index matrix (is_nsample_const=True)
  """
  if is_nsample_const:
    if replacement:
      keys = jax.random.split(rng, num_samples)
      idx = jax.vmap(
          functools.partial(jax.random.categorical, logits=log_prob),
          out_axes=-1)(keys)
    else:
      def fn_sample(chosen_mask, i):
        key = jax.random.fold_in(rng, i)
        logits = log_prob * (1.0 - chosen_mask) + chosen_mask * -1e9
        x = jax.random.categorical(key, logits=logits)
        new_mask = chosen_mask + jax.nn.one_hot(
            x, num_classes=logits.shape[-1], dtype=chosen_mask.dtype)
        return new_mask, x
      _, idx = jax.lax.scan(fn_sample, jnp.zeros_like(log_prob),
                            jnp.arange(num_samples))
      idx = jnp.transpose(idx, axes=list(range(1, idx.ndim)) + [0])
    return idx
  else:
    raise NotImplementedError


def bernoulli_logp(rng, log_prob):
  noise = jax.random.uniform(rng, shape=log_prob.shape, minval=0.0, maxval=1.0)
  return jnp.log(noise + 1e-24) < log_prob


def mh_step(rng, log_prob, current_sample, new_sample):
  use_new_sample = bernoulli_logp(rng, log_prob)
  return jnp.where(jnp.expand_dims(use_new_sample, range(1, new_sample.ndim)),
                   new_sample, current_sample), use_new_sample
