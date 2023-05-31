"""Training binary ebm."""

from collections.abc import Sequence
import functools
import os
from absl import app
from absl import flags
from flax import jax_utils
from flax.core.frozen_dict import unfreeze

from discs.common import data_loader
from discs.common import plot
from discs.common import utils
from discs.learning import replay_buffer
from discs.learning import train
from discs.models import resnet
from discs.samplers import dlmc
from ml_collections import config_flags
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import tensorflow as tf

_CONFIG = config_flags.DEFINE_config_file('config')
flags.DEFINE_integer('seed', 1, 'seed')
flags.DEFINE_string('save_root', '', 'root folder for results')

FLAGS = flags.FLAGS


class EBMTrainer(train.Trainer):
  """PCD trainer for EBM."""

  def __init__(self, config, model, sampler, x_train):
    super().__init__(config)
    self.model = model
    self.sampler = sampler
    self.p_control = config.experiment.p_control
    self.energy_l2 = config.experiment.energy_l2
    sampler_step = functools.partial(self.sampler.step, model=self.model)
    self.sampler_step = jax.pmap(sampler_step, axis_name='shard')
    self.batch_forward = jax.pmap(self.model.forward, axis_name='shard')
    self.buffer = replay_buffer.ReplayBuffer(config, x_train)

  def build_loss_func(self, rng, batch):
    del rng
    x_positive, x_negative = batch
    def loss_func(params):
      ll_positive = self.model.forward(params, x_positive)
      ll_negative = self.model.forward(params, x_negative)
      loss = jnp.mean(ll_negative - ll_positive)
      return loss, {'loss': loss,
                    'll_pos': jnp.mean(ll_positive),
                    'll_neg': jnp.mean(ll_negative)}
    return loss_func

  def batch_processing(self, local_state, shared_state, batch_rng_key, batch):
    samples, sampler_state = local_state.samples, local_state.sampler_state
    bsize = samples.shape[0] * samples.shape[1]
    buffer_samples, sample_indices = self.buffer.sample(bsize)
    samples = jnp.reshape(jax.device_put(buffer_samples), samples.shape)
    for micro_step in range(self.config.experiment.pcd_steps):
      batch_rng_key = jax.random.fold_in(batch_rng_key, micro_step)
      batch_rng = utils.shard_prng_key(batch_rng_key)
      samples, sampler_state, _ = self.sampler_step(
          rng=batch_rng, x=samples, model_param=shared_state.params,
          state=sampler_state)
    local_state = utils.SamplerState(
        step=local_state.step + self.config.experiment.pcd_steps,
        samples=samples, sampler_state=sampler_state)
    buffer_samples = np.reshape(jax.device_get(samples),
                                (bsize,) + samples.shape[2:])
    self.buffer.update(buffer_samples, sample_indices)
    batch = (batch, samples)
    return batch, local_state

  def plot_batch(self, step, shared_state, local_state):
    del shared_state
    png_name = '%s/chain-%d.png' % (self.config.experiment.fig_folder, step)
    plot.plot_shareded_image(png_name, local_state.samples, 28, 28, 1,
                             rescale=255.0)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value
  config.experiment.save_root = os.path.join(
      FLAGS.save_root, config.experiment.img_config)
  logger = utils.setup_logging(config)
  data_np = train.load_numpyds(
      config.experiment.dataset, config.experiment.data_dir)
  train_dataset = tf.data.Dataset.from_tensor_slices(data_np)
  with config.unlocked():
    config.model.data_mean = np.mean(data_np, axis=0).tolist()
  with open(os.path.join(config.experiment.save_root, 'config.yaml'), 'w') as f:
    f.write(config.to_yaml())

  global_key = jax.random.PRNGKey(FLAGS.seed)
  model = resnet.build_model(config)
  sampler = dlmc.build_sampler(config)
  trainer = EBMTrainer(config, model, sampler, data_np)

  global_key, init_key = jax.random.split(global_key)
  global_state, local_state = trainer.init_states(init_key)

  train_loader = data_loader.prepare_dataloader(
      train_dataset, config=config.experiment,
      drop_remainder=True, repeat=True)
  train_loader = data_loader.numpy_iter(train_loader)
  final_state = trainer.train_loop(
      logger, global_key, global_state, local_state, train_loader,
      fn_plot=trainer.plot_batch)
  final_state = jax_utils.unreplicate(final_state)
  results = {}
  results['params'] = unfreeze(final_state.params)
  results['params']['data_mean'] = config.model.data_mean
  with open(os.path.join(config.experiment.save_root, 'params.pkl'), 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
