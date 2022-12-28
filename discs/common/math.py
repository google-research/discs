"""Common math func impl in Jax."""

import math
import jax
import jax.numpy as jnp


def prod(*args, **kwargs):
  return math.prod(*args, **kwargs)


def gumbel(rng, loc):
  uniform_sample = jax.random.uniform(
      rng, shape=loc.shape, minval=0.0, maxval=1.0
  )
  return loc -jnp.log(-jnp.log(uniform_sample))


def multinomial(rng, log_prob, num_samples,
                replacement=True, is_nsample_const=True, batch_size=1):
  """Multinomial sample.

  Args:
    rng: random generator;
    log_prob: log probability in the shape of [..., num_categories];
    num_samples: number of samples;
    replacement: sampling with/without replacement
    is_nsample_const: is num_samples a constant int or a tensor?
    batch_size: only used when replacement=True and is_nsample_const=False
  Returns:
    sampled indices, in the form of either a counting matrix (
    is_nsample_const=False), or an index matrix (is_nsample_const=True)
  """
  num_classes = log_prob.shape[-1]
  pbatch_shape = log_prob.shape[:-1]
  if is_nsample_const:
    if replacement:
      idx = jax.random.categorical(key=rng, logits=log_prob,
                                   shape=(num_samples,) + pbatch_shape)
      idx = jnp.transpose(idx, axes=tuple(range(1, idx.ndim)) + (0,))
    else:
      perturbed_ll = gumbel(rng, loc=log_prob)
      _, idx = jax.lax.top_k(perturbed_ll, k=num_samples)
    return idx
  else:
    if replacement:
      def body_fun(val):
        cnt, key, total_count = val
        key, next_key = jax.random.split(key)
        idx = jax.random.categorical(
            key=key, logits=log_prob, shape=(batch_size,) + pbatch_shape)
        onehot = jax.nn.one_hot(idx, num_classes=num_classes, dtype=jnp.int32)
        mask = jnp.arange(batch_size, dtype=cnt.dtype) + cnt < num_samples
        mask = jnp.expand_dims(mask, range(1, onehot.ndim)).astype(jnp.int32)
        onehot = onehot * mask
        total_count = total_count + jnp.sum(onehot, axis=0)
        return (cnt + batch_size, next_key, total_count)
      init_val = (jnp.zeros(shape=(), dtype=num_samples.dtype),
                  rng,
                  jnp.zeros(shape=log_prob.shape, dtype=jnp.int32))

      _, _, selected = jax.lax.while_loop(
          cond_fun=lambda val: jnp.less(val[0], num_samples),
          body_fun=body_fun,
          init_val=init_val
      )
    else:
      perturbed_ll = gumbel(rng, loc=log_prob)
      sorted_ll = jnp.sort(perturbed_ll)
      threshold = jnp.expand_dims(
          sorted_ll[..., num_classes - num_samples], axis=-1)
      selected = (perturbed_ll >= threshold).astype(jnp.int32)
    return selected


def bernoulli_logp(rng, log_prob):
  noise = jax.random.uniform(rng, shape=log_prob.shape, minval=0.0, maxval=1.0)
  return jnp.log(noise + 1e-24) < log_prob


def mh_step(rng, log_prob, current_sample, new_sample):
  use_new_sample = bernoulli_logp(rng, log_prob)
  return jnp.where(jnp.expand_dims(use_new_sample, range(1, new_sample.ndim)),
                   new_sample, current_sample), use_new_sample
