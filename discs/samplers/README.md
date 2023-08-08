## Extending Samplers 
This directory contains the implementation of different discrete space samplers.
Each sampler extends the `AbstractSampler` class defined in `abstractsampler.py`.
More specifically, each sampler overrides the abstract methods of:
* `make_init_state`: 
* `update_sampler_state`:
* `step`:
To add a new sampler, the a new python for for the new sampler should be created, "sapler_name".py that overrides the above methods.

In the configs folder, you can find the parameters specific to each sampling approach.
The config file name for each specific sampler should follow "sampler_name"_config.py and the should follow the below structure.
"""Config for Gibbs sampler.
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='gibbs',
      .
      .
      .
  )
  return config_dict.ConfigDict(sampler_config)
"""


## Using the Samplers out of this package
The samplers need to have access to the shape (a tuple file) and the number of categories (int).
To instantiate a sampler, a genaral config is needed which contains both the sampler specific parameters and the sample features which is model dependent.
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
We provide the code snippet below as an example on how to instantiate the gwg sampler for the purpose of using out of this package pipeline.
Note that the sample shape and number of categories are manually set in the below example. In `DISCS` pipeline, they are set based on the model (target distribution).
Also, to see an example on how the `ml_collections` is used to setup the configs when running an scripts, you can refer to `discs/experiment/run_sampling_local.sh' and `discs/experiment/main_sampling.py' with `config_flags.DEFINE_config_file` usecase.
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