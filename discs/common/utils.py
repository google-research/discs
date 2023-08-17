"""Utilities."""

import functools
import os
from typing import Any
import json
from absl import logging
from clu import metric_writers
from clu.metric_writers.summary_writer import SummaryWriter
from discs.graph_loader import graph_gen
import flax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as random

@flax.struct.dataclass
class SamplerState:
  step: int
  samples: Any
  sampler_state: Any


def apply_ema(decay, avg, new):
  return jax.tree_map(lambda a, b: decay * a + (1.0 - decay) * b, avg, new)


def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)


@functools.partial(jax.pmap, axis_name='shard')
def all_gather(x):
  return jax.lax.all_gather(x, 'shard', tiled=True)


def tree_stack(trees):
  """https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75 ."""
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l) for l in grouped_leaves]
  return treedef_list[0].unflatten(result_leaves)


def setup_logging(config):
  """Setup logging and writer."""
  if jax.process_index() == 0:
    logging.info(config)
    logging.info('process count: %d', jax.process_count())
    logging.info('device count: %d', jax.device_count())
    logging.info('device/host: %d', jax.local_device_count())
  logdir = os.path.join(config.experiment.save_root, 'logs')
  writer = metric_writers.create_default_writer(
      logdir, just_logging=jax.process_index() > 0
  )
  fig_folder = os.path.join(logdir, 'figures')
  if jax.process_index() == 0:
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    if not os.path.exists(fig_folder):
      os.makedirs(fig_folder)
    if config.experiment.use_tqdm:
      writer = SummaryWriter(logdir)
    else:
      writer = metric_writers.create_default_writer(logdir)
  else:
    writer = None
  config.experiment.fig_folder = fig_folder
  with open(os.path.join(config.experiment.save_root, 'config.yaml'), 'w') as f:
    f.write(config.to_yaml())
  return writer


def shard_to_local_devices(pytree):
  n_local = jax.local_device_count()

  def shard_tensor(x):
    assert x.shape[0] % n_local == 0
    return jnp.reshape(x, [n_local, x.shape[0] // n_local] + list(x.shape[1:]))

  return jax.tree_map(shard_tensor, pytree)


def get_per_process_batch_size(batch_size):
  num_devices = jax.device_count()
  assert batch_size // num_devices * num_devices == batch_size, (
      'Batch size %d must be divisible by num_devices %d',
      batch_size,
      num_devices,
  )
  batch_size = batch_size // jax.process_count()
  return batch_size


def shard_prng_key(prng_key):
  # PRNG keys can used at train time to drive stochastic modules
  # e.g. DropOut. We would like a different PRNG key for each local
  # device so that we end up with different random numbers on each one,
  # hence we split our PRNG key and put the resulting keys into the batch
  return jax.random.split(prng_key, num=jax.local_device_count())


class RngGen(object):
  """Random number generator state utility for Jax."""

  def __init__(self, init_rng):
    self._base_rng = init_rng
    self._counter = 0

  def __iter__(self):
    return self

  def __next__(self):
    return self.advance(1)

  def advance(self, count: int):
    self._counter += count
    return jax.random.fold_in(self._base_rng, self._counter)


def create_sharded_sampler_state(sampler_key, model, sampler, num_samples):
  """Create sampler state and shard into local devices."""
  init_rng, sampler_rng = jax.random.split(sampler_key)
  init_samples = model.get_init_samples(init_rng, num_samples=num_samples)
  sampler_rngs = jax.random.split(sampler_rng, jax.local_device_count())
  sampler_states = [
      sampler.make_init_state(sampler_rng) for sampler_rng in sampler_rngs
  ]
  local_state = SamplerState(
      step=jnp.zeros((jax.local_device_count(),), dtype=jnp.int32),
      samples=shard_to_local_devices(init_samples),
      sampler_state=tree_stack(sampler_states),
  )
  return local_state


def graph2edges(
    g,
    build_bidir_edges=False,
    padded_num_edges=0,
    padded_weight=0.0,
    has_edge_weights=True,
):
  """Convert nx graph into param dict."""
  num_edges = len(g.edges())
  if padded_num_edges and padded_num_edges > num_edges:
    num_edges = padded_num_edges
  edge_from = np.zeros((num_edges,), dtype=np.int32)
  edge_to = np.zeros((num_edges,), dtype=np.int32)
  if has_edge_weights:
    edge_weight = np.zeros((num_edges,), dtype=np.float32) + padded_weight
  for i, e in enumerate(g.edges(data=True)):
    x, y = e[0], e[1]
    edge_from[i] = x
    edge_to[i] = y
    if has_edge_weights:
      edge_weight[i] = e[2]['weight']
  edge_mask = [1] * len(g.edges()) + [0] * (num_edges - len(g.edges()))
  ret = {
      'num_edges': jnp.array([len(g.edges())], dtype=jnp.int32),
      'edge_from': jnp.array(edge_from),
      'edge_to': jnp.array(edge_to),
      'edge_mask': jnp.array(edge_mask, dtype=jnp.int32),
  }
  if has_edge_weights:
    ret['edge_weight'] = jnp.array(edge_weight)
  if build_bidir_edges:
    ret['bidir_edge_from'] = jnp.concatenate((ret['edge_from'], ret['edge_to']))
    ret['bidir_edge_to'] = jnp.concatenate((ret['edge_to'], ret['edge_from']))
    if has_edge_weights:
      ret['bidir_edge_weight'] = jnp.concatenate(
          (ret['edge_weight'], ret['edge_weight'])
      )
  ret['mask'] = None
  return ret


def parse_cfg_str(cfg):
  kv_pairs = cfg.split(',')
  cfg_dict = {}
  for kv in kv_pairs:
    args = kv.split('-')
    k = args[0]
    v = '-'.join(args[1:])
    cfg_dict[k] = v
  return cfg_dict


def update_graph_cfg(config, graphs):
  config.model.max_num_nodes = graphs.max_num_nodes
  config.model.max_num_edges = graphs.max_num_edges
  config.model.shape = (graphs.max_num_nodes,)


def get_datagen(config):
  test_graphs = graph_gen.get_graphs(config)
  update_graph_cfg(config, test_graphs)
  datagen = test_graphs.get_iterator('test', config.model.num_models)
  return datagen


def create_infill_dataset(
    data_root,
    tokenizer,
    num_of_masks,
    num_of_sentences=10,
    min_length=10,
    max_length=20,
):
  """data_root: the directory where the datasets are stored (default: './text_infilling_data')

  tokenizer: the tokenizer for the language model
  num_of_sentences: the number of sentences to sample from TBC and Wiki
  min_length: the minimal length of sampled sentence
  max_length: the maximal length of sampled sentence
  num_of_masks: the number of randomly selected masks to infill words
  """
  data = []
  tbc_ref_list = []
  with open(os.path.join(data_root, 'tbc.5k.txt')) as f:
    tbc_lines = f.readlines()
    print('TBC lines:', len(tbc_lines))
    print('Before Shuffle', tbc_lines[0])
    random.shuffle( tbc_lines)
    print('After Shuffle', tbc_lines[0])
    for tbc in tbc_lines:
      if len(data) < num_of_sentences:
        tbc_new = tbc.replace('``', '')
        tbc_new = tbc_new.replace("''", '')
        tbc_new = tbc_new.replace('\n', '')
        tbc_new_list = tbc_new.split(' ')
        if (len(tbc_new_list) <= max_length) and (
            len(tbc_new_list) >= min_length
        ):
          infill_pos = random.choice(
              range(1, len(tbc_new_list) - 1), num_of_masks, replace=False
          )
          print(tbc_new_list)

          for pos in infill_pos:
            tbc_new_list[pos] = '[MASK]'
          tbc_new_masked = ' '.join(tbc_new_list)
          tokens = tokenizer.tokenize(tbc_new_masked)
          infill_pos = []
          for i in range(len(tokens)):
            if tokens[i] == '[MASK]':
              infill_pos.append(i + 1)  ### the starting token 0 will be [CLS]

          data.append({
              'gt_sentence': tbc_new,
              'sentence': tbc_new_masked,
              'infill_pos': infill_pos,
          })
        else:
          tbc_ref_list.append(tbc)
      else:
        tbc_ref_list.append(tbc)

  with open(os.path.join(data_root, 'tbc_remove_infill.5k.txt'), 'w') as f:
    ### NOTE: we remove the sentence to be infilled from the reference dataset  to compute meaningful BLEU score
    f.writelines(tbc_ref_list)

  wiki_ref_list = []
  with open(os.path.join(data_root, 'wiki103.5k.txt')) as f:
    wiki_lines = f.readlines()
    print('WIKI lines:', len(wiki_lines))
    print('Before Shuffle', wiki_lines[0])
    random.shuffle(wiki_lines)
    print('After Shuffle', wiki_lines[0])
    for wiki in wiki_lines:
      if len(data) < (2 * num_of_sentences):
        wiki_new = wiki.replace('@@unknown@@', '[UNK]')
        wiki_new = wiki_new.replace('@@UNKNOWN@@', '[UNK]')
        wiki_new = wiki_new.replace('@-@', '-')
        wiki_new = wiki_new.replace('\n', '')
        wiki_new_list = wiki_new.split(' ')

        if (len(wiki_new_list) <= max_length) and (
            len(wiki_new_list) >= min_length
        ):
          infill_pos = random.choice(
              range(1, len(wiki_new_list) - 1), num_of_masks, replace=False
          )

          for pos in infill_pos:
            wiki_new_list[pos] = '[MASK]'
          wiki_new_masked = ' '.join(wiki_new_list)
          tokens = tokenizer.tokenize(wiki_new_masked)
          infill_pos = []
          for i in range(len(tokens)):
            if tokens[i] == '[MASK]':
              infill_pos.append(i + 1)  ### the starting token 0 will be [CLS]

          data.append({
              'gt_sentence': wiki_new,
              'sentence': wiki_new_masked,
              'infill_pos': infill_pos,
          })
        else:
          wiki_ref_list.append(wiki)
      else:
        wiki_ref_list.append(wiki)

  with open(os.path.join(data_root, 'wiki103_remove_infill.5k.txt'), 'w') as f:
    f.writelines(wiki_ref_list)

  print('Generated Data:')
  print(data)
  with open(os.path.join(data_root, 'infilling_task.json'), 'w') as f_obj:
    json.dump(data, f_obj)
