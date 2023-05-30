import pdb
from discs.evaluators import abstractevaluator
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections


class Bernoullievaluator(abstractevaluator.AbstractEvaluator):
  """Evaluator class specific to evaluating samplees run on bernoulli model."""

  def _get_ubbiased_sample_variance(self, samples):
    """Computes unbiased variance of each chain."""
    sample_mean = jnp.mean(samples, axis=0, keepdims=True)
    var_over_samples = jnp.sum((samples - sample_mean) ** 2, axis=0) / (
        samples.shape[0] - 1
    )
    return var_over_samples

  def _get_max_mse_error(self, pred, target):
    """Gets the max mse error over the batch of chains."""
    return jnp.max((pred - target) ** 2)

  def _get_mse(self, pred, target):
    return jnp.mean((pred - target) ** 2)

  def _get_sample_mean(self, samples):
    """Computes sample mean of each chain."""
    return jnp.mean(samples, axis=0)

  def _get_population_mean_and_var(self, model, params):
    """Gets the population mean and var from the model(specific to Bernoulli model with closed form solution)."""
    mean_p = model.get_expected_val(params)
    var_p = model.get_var(params)
    return mean_p, var_p

  def _compute_error(self, model, params, samples):
    """Computes the average/max of mean and var error over the batch of chains."""
    mean_p, var_p = self._get_population_mean_and_var(model, params)
    mean_s_batch = self._get_sample_mean(samples)
    avg_mean_error = self._get_mse(mean_s_batch, mean_p)
    max_mean_error = self._get_max_mse_error(mean_s_batch, mean_p)
    var_unbiased_s = self._get_ubbiased_sample_variance(samples)
    avg_var_error = self._get_mse(var_p, var_unbiased_s)
    max_var_error = self._get_max_mse_error(var_p, var_unbiased_s)
    return avg_mean_error, max_mean_error, avg_var_error, max_var_error

  def _compute_error_across_chain_and_batch(self, samples, model, params):
    """Computes the error over chains and the combined last sample of all chains."""
    (
        avg_mean_error,
        max_mean_error,
        avg_var_error,
        max_var_error,
    ) = self._compute_error(model, params, samples)

    last_samples = jnp.expand_dims(samples[-1], axis=1)
    (
        avg_mean_error_last_samples,
        _,
        avg_var_error_last_samples,
        _,
    ) = self._compute_error(model, params, last_samples)

    return (
        avg_mean_error,
        max_mean_error,
        avg_var_error,
        max_var_error,
        avg_mean_error_last_samples,
        avg_var_error_last_samples,
    )

  def evaluate(self, samples, model, params):
    return self._compute_error_across_chain_and_batch(samples, model, params)

  def plot_mixing_time_graph_over_chain(
      self, save_dir, model, all_params, all_samples, all_labels
  ):
    """Plots the error over window of samples of chains over time."""
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(8)
    for i, chain in enumerate(all_samples):
      label = all_labels[i]
      params = all_params[i]
      mean_errors = []
      max_mean_errors = []
      for start in range(0, len(chain), self.config.experiment.window_stride):
        if (len(chain) - start) < self.config.experiment.window_size:
          break
        samples = chain[start : start + self.config.experiment.window_size]
        avg_mean_error, max_mean_error, _, _ = self._compute_error(
            model, params, samples
        )
        mean_errors.append(avg_mean_error)
        max_mean_errors.append(max_mean_error)
      mean_errors = jnp.array(mean_errors)
      plt.plot(
          jnp.arange(1, 1 + len(mean_errors)), mean_errors, '--', label=label
      )
    plt.xlabel('Iteration Step Over Chain')
    plt.ylabel('Avg Mean Error')
    plt.title('Avg Mean Error Over Chains!')
    plt.legend()
    plt.savefig(f'{save_dir}/MixingTimeAvgMean.png')
    '''plt.clf()
    plt.plot(
        jnp.arange(1, 1 + len(max_mean_errors)),
        max_mean_errors,
        '--bo',
        label=label,
    )
    plt.xlabel('Iteration Step Over Chain')
    plt.ylabel('Max Mean Error')
    plt.title('Max Mean Error Over Chains!')
    plt.legend()
    plt.savefig(f'{save_dir}/MixingTimeMaxMean.png')'''


def build_evaluator(config):
  return Bernoullievaluator(config)
