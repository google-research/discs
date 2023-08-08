## Extending the samplers 
This directory contains the implementation of different discrete space samplers.
Each sampler extends the `AbstractSampler` class defined in `abstractsampler.py`.
More specifically, each sampler overrides the following methods:
* `make_init_state`: initialized the state/parameters of the sampler.
* `update_sampler_state`: updated the state/parameters of the sampling algorithm.
* `step`: performs the mh step. It returns the next sample and the updated state of the sampling algorithm. 

To add a new sampler, a new python file for the new sampler should be created, with the name structure of "sampler_name".py which implements the above methods and extends the `AbstractSampler` class.
In the configs folder, you can find the parameters specific to each sampling approach.
The config file name for each specific sampler should follow "sampler_name"_config.py and the below structure.
"""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='gibbs',
      '...'
  )
  return config_dict.ConfigDict(sampler_config)
"""


## Using the samplers out of this package
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
We provide the code snippet below as an example of how to instantiate the 'gwg' sampler for the purpose of using out of this package pipeline.
Note that the sample shape and number of categories are manually set in the below example. In `DISCS` pipeline, they are set based on the model (target distribution).
Also, to see an example of how the `ml_collections` is used to set up the configs when running a script, you can refer to `discs/experiment/run_sampling_local.sh' and `discs/experiment/main_sampling.py' with `config_flags.DEFINE_config_file` use-case.
```
import importlib
from ml_collections import config_dict
from ml_collections import config_flags
from discs.samplers.configs import gwg_config


def get_general_config():
  general_config = dict(
      model=dict(
          shape=(100,),
          num_categories=10,
      ),
      sampler=dict(
          name='',
      ),
  )
  return config_dict.ConfigDict(general_config)


# config
config = get_general_config()
# updating the config with the sampler config values.
config.sampler.update(gwg_config.get_config())
# sampler
sampler_mod = importlib.import_module('discs.samplers.%s' % config.sampler.name)
sampler = sampler_mod.build_sampler(config)
```