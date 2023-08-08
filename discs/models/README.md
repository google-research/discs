## Extending the models 
This directory contains the implementation of different models/tasks.
Each model extends the `AbstractModel` class defined in `abstractmodel.py`.
More specifically, each model overrides the following abstract methods:
* `make_init_params`: Loads or randomly samples the model parameters..
* `get_init_samples`: returns the initial samples for the start of chain generation.
* `forward`: computes the energy of batch of samples.

To add a new model, a new python file for the new model should be created, with the name structure of "model_name".py which implements the above methods and extends the `AbstractModel` class.
In the configs folder, you can find the parameters specific to each model.
The config file name for each specific model should follow "model_name"_config.py and the below structure.
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

## Using the models out of this package
The sampling algorithms need to have access to the shape (a tuple) and the number of categories (int) of the sample.
To instantiate a sampler, a general config is needed which contains both the sampler specific parameters and the sample features which is model dependent.
More specifically the config passed to the sampler should have the following structure:
```
  general_config = dict(
      model=dict(
        shape=,
        num_categories=
      ),
      sampler=dict(
          name='',
      ),
  )
```
We provide the code snippet below as an example of how to instantiate the 'bernolli' sampler for the purpose of using out of this package pipeline.
Note that the sample shape and number of categories are manually set in the below example. In `DISCS` pipeline, they are set based on the model (target distribution).
Also, to see an example of how the `ml_collections` is used to set up the configs when running a script, you can refer to `discs/experiment/run_sampling_local.sh' and `discs/experiment/main_sampling.py' with `config_flags.DEFINE_config_file` use-case.
```
import importlib
from ml_collections import config_dict
from ml_collections import config_flags
from discs.models.configs import maxcut_config


def get_model_config():
  # getting the base config of maxcut model.
  config = maxcut_config.get_config()
  if config.get('graph_type', None):
    graph_config = importlib.import_module('discs.models.configs.%s.%s'% (config.name, config.graph_type))
    config.update(graph_config.get_model_config(config.model))
    co_exp_default_config = importlib.import_module(
        'discs.experiment.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
    config.update(_EXPERIMENT_CONFIG.value)
  return config


# config
config = get_model_config()
# model
model_mod = importlib.import_module('discs.models.%s' % config.name)
model = model_mod.build_model(config)
```

