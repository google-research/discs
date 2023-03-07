from discs.graph_loader import graph_gen
import optax
from absl import flags
import pdb

flags.DEFINE_integer('seed', 1, 'seed')
flags.DEFINE_bool('do_eval', False, 'eval?')
flags.DEFINE_string('save_root', '', 'root folder for results')

FLAGS = flags.FLAGS


def build_temperature_schedule(config):
  """Temperature schedule."""

  if config.t_schedule == 'constant':
    schedule = lambda step: step * 0 + config.init_temperature
  elif config.t_schedule == 'linear':
    schedule = optax.linear_schedule(
        config.init_temperature, config.final_temperature, config.chain_length
    )
  elif config.t_schedule == 'exp_decay':
    schedule = optax.exponential_decay(
        config.init_temperature,
        config.chain_length,
        config.decay_rate,
        end_value=config.final_temperature,
    )
  else:
    raise ValueError('Unknown schedule %s' % config.t_schedule)
  return schedule


def update_graph_config(config, graphs):
  config.experiment.save_root = FLAGS.save_root
  config.model.max_num_nodes = graphs.max_num_nodes
  config.model.max_num_edges = graphs.max_num_edges
  config.model.shape = (graphs.max_num_nodes,)


def get_datagen(config):
  pdb.set_trace()
  test_graphs = graph_gen.get_graphs(config)
  update_graph_config(config, test_graphs)
  datagen = test_graphs.get_iterator('test', config.experiment.batch_size)
  return datagen
