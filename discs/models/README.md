## How to add new models
This directory contains the implementation of different models/tasks.
Each model extends the `AbstractModel` class defined in `abstractmodel.py`.
More specifically, each model overrides the following abstract methods:
* `make_init_params`: loads or randomly samples the model parameters.
* `get_init_samples`: returns a batch of initial samples for the start of chain generation.
* `forward`: computes the energy of batch of samples.

To add a new model, a new python file for the new model should be created, with the name structure of `${model_name}.py` which implements the above methods and extends the `AbstractModel` class.
In the configs folder, you can find the parameters specific to each model.
The config file name for each specific model should follow `${model_name}_config.py` and the below structure.
"""
def get_config():
  model_config = dict(
      shape=,
      num_categories=,
      name='bernoulli',
  )
  return config_dict.ConfigDict(model_config)
"""
Note that for CO and EBM, the shape and number of categories are set based on the loaded information of the graph.

## How to use the models out of this package
The config passed to the model for instantiation should have the following structure:
```
  general_config = dict(
      model=dict(
        name='',
      )
  )
```
We provide the code snippet below as an example of how to instantiate the maxcut model for the purpose of using out of this package pipeline.
Also, to see an example of how the `ml_collections` is used to set up the configs when running a script, you can refer to `discs/experiment/run_sampling_local.sh' and `discs/experiment/main_sampling.py' with `config_flags.DEFINE_config_file` use-case.
```
import importlib
from discs.models.configs import maxcut_config
from ml_collections import config_dict
from ml_collections import config_flags


def get_general_config():
  general_config = dict(
      model=dict(
          name='',
      )
  )
  return config_dict.ConfigDict(general_config)


def get_model_config():
  # getting the base config of maxcut model.
  config = get_general_config()
  config.model.update(maxcut_config.get_config())
  # graph type used in case of CO
  if config.model.get('graph_type', None):
    graph_config = importlib.import_module(
        'discs.models.configs.%s.%s'
        % (config.model.name, config.model.graph_type)
    )
    config.model.update(graph_config.get_model_config(config.model.cfg_str))
  return config


# config
config = get_model_config()
# model
model_mod = importlib.import_module('discs.models.%s' % config.model.name)
model = model_mod.build_model(config)
```

