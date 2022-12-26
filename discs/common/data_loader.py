"""Common sharded data loader based on tf data."""

from absl import logging
from discs.common import utils
import jax
import tensorflow as tf


def prepare_dataloader(dataset, config, fn_preprocess=None,
                       drop_remainder=True, repeat=False):
  """Prepare data loading, including batching, sharding, etc.

  Args:
    dataset: tf dataset
    config: config dict
    fn_preprocess: optional, used to process each single example
    drop_remainder: drop remainder of batch size?
    repeat: repeat dataset?
  Returns:
    processed tf data
  """

  num_shards = jax.process_count()
  shard_id = jax.process_index()
  dataset = dataset.shard(num_shards=num_shards, index=shard_id)
  if config.get('repeat', repeat):
    dataset = dataset.repeat()
  if config.get('shuffle_buffer_size', 0):
    dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size,
                              seed=shard_id)

  if fn_preprocess is not None:
    dataset = dataset.map(fn_preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  proc_batch_size = utils.get_per_process_batch_size(config.batch_size)
  logging.info('Batch size per process: %d', proc_batch_size)

  drop_remainder = config.get('drop_remainder', drop_remainder)
  dataset = dataset.batch(proc_batch_size // jax.local_device_count(),
                          drop_remainder=drop_remainder)
  dataset = dataset.batch(jax.local_device_count(),
                          drop_remainder=drop_remainder)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def tf_to_numpy(tf_batch):
  """TF to NumPy, using ._numpy() to avoid copy."""
  # pylint: disable=protected-access
  return jax.tree_map(
      lambda x: x._numpy() if hasattr(x, '_numpy') else x,
      tf_batch)


def numpy_iter(tf_dataset):
  return map(tf_to_numpy, iter(tf_dataset))
