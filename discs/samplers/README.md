

## How to add new samplers
To add your new discrete sampler, extend the `AbstractSampler` class defined in `abstractsampler.py` by adding your `${sampler_name}.py` file and its corresponding config file, `${sampler_name}_config.py`, in the configs folder.
The sampler config file should follow the structure below:
```
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name=${sampler_name},
      # sampling algorithm config values
  )
  return config_dict.ConfigDict(sampler_config)
```

To run the sampler locally, as an example on the Bernoulli model, you can follow:
```
   model=bernoulli sampler=${sampler_name} ./discs/experiment/run_sampling_local.sh
```
To add your sampler to the predefined Xmanager experiments at `discs/run_configs`, simply add to the files the sampler's name and any of the sampler configs you want to sweep over following the structure below:
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
Note that, when adding your new sampler to predefined Xmanager configs, make sure you follow the same model and experiment setup as the other samplers (copy model and experiment related values to the structure above).


Each sampler extends the `AbstractSampler` class defined in `abstractsampler.py`.
More specifically, each sampler overrides the following methods:
* `make_init_state`: initializes the sampler states that change through the chain generation.
* `update_sampler_state`: updates sampler state throughout the chain generation,
* `step`: performs the M-H step. It returns the next sample and the updated state of the sampling algorithm. 

Note that, the package is jax-based and you should follow the functional programming paradigm. 
The class methods should not have any side effects and you should rely on the sampler state to keep track of modified config values.

## How to use the samplers out of this package
In addition to the sampler config values defined in the config folder, the sampling algorithms need to have access to the sample shape (a tuple) and its number of categories (int).
To do so, a general config value with the below structure should be passed when instantiating any of the samplers:
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
In the general config above, the 'model' should contain the sample shape and the number of categories, the 'sampler' should contain the information of the config files defined in the sampler config folder.
In `DISCS` pipeline, the sample shape and number of categories are set based on the model (the target distribution) we are sampling from.


We provide the code snippet below as an example of how to instantiate the `gwg` sampler to use out of this package pipeline.
We show how the setting up the general config happens.
Note that the sample shape and number of categories are manually set in the below example. In `DISCS` pipeline, they are set based on the model (target distribution).
Also, to see an example of how the `ml_collections` is used to set up the configs when running a script, you can check the `config_flags.DEFINE_config_file` use case in `discs/experiment/run_sampling_local.sh` and `discs/experiment/main_sampling.py`.
```
import importlib
from ml_collections import config_dict
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
