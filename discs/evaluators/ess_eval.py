"""ESS Evaluator Class."""

from discs.evaluators import abstractevaluator
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_probability as tfp

class ESSevaluator(abstractevaluator.AbstractEvaluator):
  """ESS evaluator class."""

  def _eval_over_num_ll_calls(self, eval_val, num_loglike_calls):
    """ESS over num of energy evaluation."""
    return eval_val / (num_loglike_calls * self.config.ess_ratio)

  def _eval_over_mh_step(self, eval_val, mh_steps):
    """ESS over mh steps."""
    return eval_val / (mh_steps * self.config.ess_ratio)

  def _eval_over_time(self, eval_val, time):
    """ESS over run time."""
    return eval_val / (time)

  def get_eval_metrics(self, eval_val, running_time, num_ll_calls):
    """Computes objective value over time, M-H step and calls of loglike function."""

    ess_over_mh_steps = self._eval_over_mh_step(
        eval_val, self.config.chain_length
    )
    ess_over_time = self._eval_over_time(eval_val, running_time)
    ess_over_ll_calls = self._eval_over_num_ll_calls(eval_val, num_ll_calls)
    return (eval_val, ess_over_mh_steps, ess_over_time, ess_over_ll_calls)

  def _get_ess(self, rnd_ess, mapped_samples):
    """Computes the mean ESS over the chains."""
    mapped_samples = mapped_samples.astype(jnp.float32)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    ess_of_chains = tfp.substrates.jax.mcmc.effective_sample_size(
        mapped_samples
    )
    ess_of_chains = jnp.nan_to_num(ess_of_chains, nan=1.0)
    mean_ess = jnp.mean(ess_of_chains)
    std_ess = jnp.std(ess_of_chains)
    return jnp.array([mean_ess, std_ess])

  def evaluate(self, samples, rnd):
    return self._get_ess(rnd, samples)


def build_evaluator(config):
  return ESSevaluator(config.experiment)
