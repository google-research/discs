"""Saver Class."""

import csv
import os
import pdb
import pickle
from discs.evaluators import bernoulli_eval as b_eval
from matplotlib import cm
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import numpy as np


class Saver:
  """Class used to plot and save the results of the experiments."""

  def __init__(self, save_dir, config: ml_collections.ConfigDict):
    self.config = config
    self.save_dir = save_dir
    if not os.path.isdir(self.save_dir):
      os.makedirs(self.save_dir)

  def _plot_acc_ratio(self, acc_ratio):
    plt.plot(jnp.arange(1, 1 + len(acc_ratio)), acc_ratio, '--b')
    plt.xlabel('Steps')
    plt.ylabel('Acc Ratio')
    plt.ylim((-0.1, 1.1))
    plt.title(
        'Acc Ratio for sampler {} on model {}!'.format(
            self.config.sampler.name, self.config.model.name
        )
    )
    path = f'{self.save_dir}/AccRatio_{self.config.sampler.name}_{self.config.model.name}'
    plt.savefig(path)
    plt.close()

  def _plot_hops(self, hops):
    plt.plot(jnp.arange(1, 1 + len(hops)), hops, '--b')
    plt.xlabel('Steps')
    plt.ylabel('Hops')
    plt.title(
        'Hops for sampler {} on model {}!'.format(
            self.config.sampler.name, self.config.model.name
        )
    )
    path = f'{self.save_dir}/Hops_{self.config.sampler.name}_{self.config.model.name}'
    plt.savefig(path)
    plt.close()

  def _save_ess_results(self, metrcis, running_time):
    """Saving the Evaluation Results in txt and CSV file."""

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
    results['ESS'] = metrcis[0]
    results['ESS_M-H'] = metrcis[1]
    results['ESS_T'] = metrcis[2]
    results['ESS_EE'] = metrcis[3]
    results['Time'] = running_time
    results['batch_size'] = self.config.experiment.batch_size
    results['chain_length'] = self.config.experiment.chain_length
    results['ess_ratio'] = self.config.experiment.ess_ratio

    csv_path = f'{self.save_dir}/results.csv'
    if not os.path.exists(csv_path):
      with open(f'{self.save_dir}/results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
        csvfile.close()
    else:
      with open(f'{self.save_dir}/results.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
        writer.writerow(results)
        csvfile.close()

    with open(
        f'{self.save_dir}/{self.config.model.name}_{self.config.sampler.name}_{running_time}.txt',
        'w',
    ) as f:
      f.write('Mean ESS: {} \n'.format(metrcis[0]))
      f.write('ESS M-H Steps: {} \n'.format(metrcis[1]))
      f.write('ESS over time: {} \n'.format(metrcis[2]))
      f.write('ESS over loglike calls: {} \n'.format(metrcis[3]))
      f.write('Running time: {} s \n'.format(running_time))
      f.write(str(self.config))

  def save_results(self, acc_ratio, hops, metrcis, running_time):
    if self.config.experiment.get_additional_metrics:
      self._plot_acc_ratio(acc_ratio)
      self._plot_hops(hops)
    if self.config.experiment.evaluator == 'ess_eval':
      self._save_ess_results(metrcis, running_time)

  def dump_samples(self, samples, visualize=False):
    root_path = self.save_dir
    if not os.path.isdir(root_path):
      os.makedirs(root_path)
    path = os.path.join(root_path, 'samples.pkl')
    res = {}
    trajectory = jnp.array(samples)
    res['trajectory'] = trajectory
    with open(path, 'wb') as file:
      pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
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
    if not os.path.isdir(self.save_dir):
      os.makedirs(self.save_dir)
    path = os.path.join(self.save_dir, 'results.pkl')
    results = {}
    results['trajectory'] = np.array(trajectory)
    results['best_ratio'] = np.array(best_ratio)
    results['running_time'] = running_time
    if len(best_samples) != 0:
      results['best_samples'] = np.array(best_samples)
    with open(path, 'wb') as file:
      pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    results = {}
    results['best_ratio_mean'] = np.mean(np.array(best_ratio))
    results['running_time'] = running_time
    with open(f'{self.save_dir}/results.csv', 'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
      writer.writeheader()
      writer.writerow(results)
      csvfile.close()

  def dump_params(self, params):
    path = os.path.join(self.save_dir, 'params.pkl')
    params_dict = {}
    params_dict['params'] = params
    with open(path, 'wb') as file:
      pickle.dump(params_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

  def save_lm_results(self, results, results_topk):
    if not os.path.isdir(self.save_dir):
      os.makedirs(self.save_dir)
    path = os.path.join(self.save_dir, 'results.pkl')
    with open(path, 'wb') as file:
      pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    if results_topk:
      path = os.path.join(self.save_dir, 'results_topk.pkl')
      with open(path, 'wb') as file:
        pickle.dump(results_topk, file, protocol=pickle.HIGHEST_PROTOCOL)

  def plot_estimation_error(self, model, params, samples):
    ber_eval = b_eval.build_evaluator(self.config)
    ber_eval.plot_mixing_time_graph_over_chain(
        self.save_dir, model, params, samples
    )

  def save_logz(self, logz_finals):
    path = os.path.join(self.save_dir, 'logz.pkl')
    params_dict = {}
    params_dict['logz'] = np.array(logz_finals)
    with open(path, 'wb') as file:
      pickle.dump(params_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def build_saver(save_dir, config):
  return Saver(save_dir, config)
