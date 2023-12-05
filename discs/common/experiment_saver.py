"""Saver Class."""

import csv
import os
import pdb
import pickle
from discs.evaluators import bernoulli_eval as b_eval
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
import ml_collections
import numpy as np


class Saver:
  """Class used to plot and save the results of the experiments."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config
    self.save_dir = config.experiment.save_root
    if not os.path.isdir(self.save_dir):
      os.makedirs(self.save_dir)

  def _dump_dict(self, params, key_name: str):
    path = os.path.join(self.save_dir, f'{key_name}.pkl')
    if not isinstance(params, dict):
      params = np.array(params)
      params_dict = {}
      params_dict[f'{key_name}'] = params
    else:
      params_dict = params
    with open(path, 'wb') as file:
      pickle.dump(params_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

  def _plot_additional_metrics(self, vals, metric: str):
    """Used to plot hops and acc ratio of the sampling through time."""
    plt.plot(jnp.arange(1, 1 + len(vals)), vals, '--b')
    plt.xlabel('Steps')
    plt.ylabel(f'{metric}')
    if metric == 'Acc Ratio':
      plt.ylim((-0.1, 1.1))
    plt.title(
        '{} for sampler {} on model {}!'.format(
            metric, self.config.sampler.name, self.config.model.name
        )
    )
    path = f'{self.save_dir}/{metric}_{self.config.sampler.name}_{self.config.model.name}'
    plt.savefig(path)
    plt.close()

  def _save_ess_results(self, metrcis, running_time):
    """Saving the Evaluation Results in a CSV file."""

    results = {}
    results['sampler'] = self.config.sampler.name
    if 'adaptive' in self.config.sampler.keys():
      results['sampler'] = f'a_{self.config.sampler.name}'

    if 'solver' in self.config.sampler.keys():
      if self.config.sampler.solver == 'euler_forward':
        results['sampler'] = results['sampler'] + 'f'

    if 'balancing_fn_type' in self.config.sampler.keys():
      if self.config.sampler.balancing_fn_type == 'RATIO':
        results['sampler'] = results['sampler'] + '(ratio)'
      elif self.config.sampler.balancing_fn_type == 'MAX':
        results['sampler'] = results['sampler'] + '(max)'
      elif self.config.sampler.balancing_fn_type == 'MIN':
        results['sampler'] = results['sampler'] + '(min)'
      else:
        results['sampler'] = results['sampler'] + '(sqrt)'

    results['model'] = self.config.model.name
    results['num_categories'] = self.config.model.num_categories
    results['shape'] = self.config.model.shape
    results['batch_size'] = self.config.experiment.batch_size
    results['chain_length'] = self.config.experiment.chain_length
    results['ess_ratio'] = self.config.experiment.ess_ratio
    results['Time'] = running_time
    results['ESS'] = metrcis[0]
    results['ESS_M-H'] = metrcis[1]
    results['ESS_T'] = metrcis[2]
    results['ESS_EE'] = metrcis[3]

    csv_path = f'{self.save_dir}/results.csv'
    with open(f'{self.save_dir}/results.csv', 'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
      writer.writeheader()
      writer.writerow(results)
      csvfile.close()

  def plot_estimation_error(self, model, params, samples):
    ber_eval = b_eval.build_evaluator(self.config)
    ber_eval.plot_mixing_time_graph_over_chain(
        self.save_dir, model, params, samples
    )

  def save_results(self, acc_ratio, hops, metrcis, running_time):
    if self.config.experiment.get_additional_metrics:
      self._plot_additional_metrics(acc_ratio, 'Acc Ratio')
      self._plot_additional_metrics(hops, 'Hops')
    if self.config.experiment.evaluator == 'ess_eval':
      self._save_ess_results(metrcis, running_time)

  def dump_samples(self, samples, visualize=False):
    res = {}
    trajectory = jnp.array(samples)
    res['trajectory'] = trajectory
    self._dump_dict(res, 'samples')
    if visualize:
      for step, samples in enumerate(trajectory):
        size = int(jnp.sqrt(samples[0].shape[0]))
        samples = jnp.reshape(samples, (-1, size, size))
        for chain in range(samples.shape[0]):
          img = samples[chain]
          image_path = os.path.join(
              root_path, f'sample_{step}_of_chain_{chain}.jpeg'
          )
          plt.imsave(image_path, np.array(img), cmap=cm.gray)

  def save_co_resuts(self, trajectory, best_ratio, running_time, best_samples):
    results = {}
    results['trajectory'] = np.array(trajectory)
    results['best_ratio'] = np.array(best_ratio)
    results['running_time'] = running_time
    results['best_ratio_mean'] = np.mean(np.array(best_ratio))
    if len(best_samples) != 0:
      results['best_samples'] = np.array(best_samples)
    self._dump_dict(results, 'results')

  def dump_params(self, params):
    self._dump_dict(params, 'params')

  def save_lm_results(self, results, results_topk):
    self._dump_dict(results, 'results')
    if results_topk:
      self._dump_dict(results_topk, 'results_topk')

  def save_logz(self, logz_finals):
    self._dump_dict(logz_finals, 'logz')


def build_saver(config):
  return Saver(config)
