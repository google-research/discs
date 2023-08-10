

## Extending the samplers
TLDR; to add your new discrete sampler, extend the `AbstractSampler` class defined in `abstractsampler.py` (${sampler_name}.py) and add its corresponding config file (${sampler_name}_config.py) in the configs folder here. 

To run the sampler locally, as an example on the bernoulli model, you can follow:
```
   model=bernoulli sampler=${sampler_name} ./discs/experiment/run_sampling_local.sh
```
To add your sampler to the predefined xmanager experiments at `discs/run_configs`, simply add its name and any of the sampler configs you want to sweep over following the structure below:
```
    {
        'sampler_config.name': [
            '${sampler_name}',
        ],
        'sampler_config.${sampler config to sweep over}': [
          "..."
        ],
    },
```
Note that, when adding your new sampler to predefined xmanager configs, make sure you follow the same model and experiment setup as the other samplers.

More Details below:
This directory contains the implementation of different discrete space samplers.
Each sampler extends the `AbstractSampler` class defined in `abstractsampler.py`.
More specifically, each sampler overrides the following methods:
* `make_init_state`: initializes the sampler state (sampling algorithm parameters).
* `update_sampler_state`: updates sampler state throughout the chain generation,
* `step`: performs the M-H step. It returns the next sample and the updated state of the sampling algorithm. 

To add a new sampler, a new python file for the new sampler should be created, with the name structure of "sampler_name".py which implements the above methods and extends the `AbstractSampler` class.
In the configs folder, you can find the parameters specific to each sampling approach.
The config file name for each specific sampler should follow "sampler_name"_config.py and the below structure.
"""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='sampler_name',
      # sampling algorithm config values
  )
  return config_dict.ConfigDict(sampler_config)
"""


## Using the samplers out of this package
In addition to the config values defined in the config folder for each sampler, the sampling algorithms need to have access to the sample shape (a tuple) and its number of categories (int).
To instantiate the sampler, a general config value with the below structure should be used.
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
The general config value should contain the sampler config values and the model config value.
The sampler config value is set from the values defined in the sampler's corresponding config file.
In `DISCS` pipeline, the sample shape and number of categories are set based on the model (the target distribution) we are sampling from.


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
