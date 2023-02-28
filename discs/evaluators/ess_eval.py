"""ESS Evaluator Class."""

from discs.evaluators import abstractevaluator
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
import tensorflow_probability as tfp
import pdb

class ESSevaluator(abstractevaluator.AbstractEvaluator):
  """ESS evaluator class."""

  def _get_ess(self, rnd_ess, mapped_samples):
    mapped_samples = mapped_samples.astype(jnp.float32)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    ess_of_chains = tfp.substrates.jax.mcmc.effective_sample_size(mapped_samples)
    ess_of_chains = jnp.nan_to_num(ess_of_chains, nan = 1.0)
    mean_ess = jnp.mean(ess_of_chains)
    return mean_ess

  def evaluate_chain(self, samples, rnd):
    return self._get_ess(rnd, samples)

  def evaluate_step(self, samples, model, params):
    return None


def build_evaluator(config):
  return ESSevaluator(config.experiment)
