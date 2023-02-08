"""Evaluator Class."""

import ml_collections
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os
import csv
import pdb

from discs.samplers.locallybalanced import LBWeightFn


class Evaluator:

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

  def _get_ess_over_num_loglike_calls(self, ess, num_loglike_calls):
    return ess / (num_loglike_calls * self.config.experiment.ess_ratio)

  def _get_ess_over_mh_step(self, ess, mh_steps):
    return ess / (mh_steps * self.config.experiment.ess_ratio)

  def _get_ess_over_time(self, ess, time):
    return ess / (time * self.config.experiment.ess_ratio)

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
    cv = tfp.mcmc.effective_sample_size(mapped_samples).numpy()
    cv[jnp.isnan(cv)] = 1.0
    return cv

  def get_effective_sample_size_metrics(
      self, samples, running_time, num_loglike_calls
  ):
    """Computes ESS over time, M-H step and calls of loglike function."""
    rnd = jax.random.PRNGKey(0)
    ess_of_chains = self._get_ess(rnd, samples)
    mean_ess = jnp.mean(ess_of_chains)
    ess_over_loglike_calls = self._get_ess_over_num_loglike_calls(
        mean_ess, num_loglike_calls
    )
    ess_over_mh_steps = self._get_ess_over_mh_step(
        mean_ess, self.config.experiment.chain_length
    )
    ess_over_time = self._get_ess_over_time(mean_ess, running_time)
    return mean_ess, ess_over_mh_steps, ess_over_time, ess_over_loglike_calls

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

  def compute_error_across_chain_and_batch(self, model, params, samples):
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

  def plot_mixing_time_graph_over_chain(
      self, model, params, chain, config_main
  ):
    """Plots the error over window of samples of chains over time."""
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
    plt.plot(jnp.arange(1, 1 + len(mean_errors)), mean_errors, '--bo')
    plt.xlabel('Iteration Step Over Chain')
    plt.ylabel('Avg Mean Error')
    plt.title('Avg Mean Error Over Chains for {}!'.format(config_main.sampler))
    plt.savefig('MixingTimeAvgMean_{}'.format(config_main.sampler))
    plt.clf()
    plt.plot(jnp.arange(1, 1 + len(max_mean_errors)), max_mean_errors, '--bo')
    plt.xlabel('Iteration Step Over Chain')
    plt.ylabel('Max Mean Error')
    plt.title('Max Mean Error Over Chains for {}!'.format(config_main.sampler))
    plt.savefig('MixingTimeMaxMean_{}'.format(config_main.sampler))

  def plot_acc_ratio(self, save_dir, acc_ratio):
    plt.plot(jnp.arange(1, 1 + len(acc_ratio)), acc_ratio, '--bo')
    plt.xlabel('Steps')
    plt.ylabel('Acc Ratio')
    plt.title(
        'Acc Ratio for sampler {} on model {}!'.format(
            self.config.sampler.name, self.config.model.name
        )
    )
    
    path = f'{save_dir}/AccRatio_{self.config.sampler.name}_{self.config.model.name}'
    plt.savefig(path)

  def save_results(self, save_dir, ess_metrcis, running_time):
    """Saving the Evaluation Results in txt and CSV file."""
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

    results = {}
    results['sampler'] = self.config.sampler.name
    if 'adaptive' in self.config.sampler.keys():
      results['sampler'] = f'a_{self.config.sampler.name}'

    if 'balancing_fn_type' in self.config.sampler.keys():
      if self.config.sampler.balancing_fn_type == LBWeightFn.RATIO:
        results['sampler'] = results['sampler'] + '(ratio)'
      elif self.config.sampler.balancing_fn_type == LBWeightFn.MAX:
        results['sampler'] = results['sampler'] + '(max)'
      elif self.config.sampler.balancing_fn_type == LBWeightFn.MIN:
        results['sampler'] = results['sampler'] + '(min)'
      else:
        results['sampler'] = results['sampler'] + '(sqrt)'

    ess_metrcis = jnp.array(ess_metrcis)
    results['model'] = self.config.model.name
    results['num_categories'] = self.config.model.num_categories
    results['shape'] = self.config.model.shape
    results['ESS'] = ess_metrcis[0]
    results['ESS_M-H'] = ess_metrcis[1]
    results['ESS_T'] = ess_metrcis[2]
    results['ESS_EE'] = ess_metrcis[3]
    results['Time'] = running_time
    results['batch_size'] = self.config.experiment.batch_size
    results['chain_length'] = self.config.experiment.chain_length
    results['ess_ratio'] = self.config.experiment.ess_ratio

    csv_path = f'{save_dir}/results.csv'
    if not os.path.exists(csv_path):
      with open(f'{save_dir}/results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
        csvfile.close()
    else:
      with open(f'{save_dir}/results.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
        writer.writerow(results)
        csvfile.close()

    with open(
        f'{save_dir}/{self.config.model.name}_{self.config.sampler.name}_{running_time}.txt',
        'w',
    ) as f:
      f.write('Mean ESS: {} \n'.format(ess_metrcis[0]))
      f.write('ESS M-H Steps: {} \n'.format(ess_metrcis[1]))
      f.write('ESS over time: {} \n'.format(ess_metrcis[2]))
      f.write('ESS over loglike calls: {} \n'.format(ess_metrcis[3]))
      f.write('Running time: {} s \n'.format(running_time))
      f.write(str(self.config))


def build_evaluator(config):
  return Evaluator(config)
