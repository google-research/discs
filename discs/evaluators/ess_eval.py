"""ESS Evaluator Class."""

from discs.evaluators import abstractevaluator
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
import tensorflow_probability as tfp


class ESSevaluator(abstractevaluator.AbstractEvaluator):
  """ESS evaluator class."""

  def _get_mapped_samples(self, rnd_ess, samples):
    vec_shape = jnp.array(samples.shape)[2:]
    vec_to_project = jax.random.normal(rnd_ess, shape=vec_shape)
    vec_to_project = jnp.array([[vec_to_project]])
    return jnp.sum(
        samples * vec_to_project, axis=jnp.arange(len(samples.shape))[2:]
    )

  def _get_ess(self, rnd_ess, samples):
    mapped_samples = self._get_mapped_samples(rnd_ess, samples).astype(
        jnp.float32
    )
    tf.config.experimental.enable_tensor_float_32_execution(False)
    ess_of_chains = tfp.mcmc.effective_sample_size(mapped_samples).numpy()
    ess_of_chains[jnp.isnan(ess_of_chains)] = 1.0
    mean_ess = jnp.mean(ess_of_chains)
    return mean_ess

  def evaluate(self, samples, rnd):
    return self._get_ess(rnd, samples)


def build_evaluator(config):
  return ESSevaluator(config.experiment)
