# DISCS
DISCS: A Benchmark for Discrete Sampling: [paper](https://openreview.net/pdf?id=oi1MUMk5NF)

## Installation

First, follow the guideline in https://github.com/google/jax#installation to
install Jax.

Then navigate to the root of the project folder `./discs/` and run

    pip install -e .

If you wish to run the experiments on Xmanager, please follow the steps described in https://github.com/google-deepmind/xmanager to setup your Xmanager.

## DISCS Package Structure 
To run a sampling experiment, we need to set up three main components. 1) The model we want to sample from (target distribution), 2) the sampler we want to use and 3) MCMC experimental configuration (number of chains, chain length, etc.). To achieve this, these three main components are being structured in DISCS package as below:
* **Model configs** which are defined under `./discs/models/configs/`. For each model, its corresponding config contains the shape and the number of categories of the sample and also additional model config values to set up the model parameters.
* **Sampler config** which are defined under `./discs/samplers/configs/`. For each sampler, its corresponding config contains the values required to set up the sampler.
* **Experiment config** which are defined under `./discs/experiment/configs/`. Note that the experimental setup varies depending on the type of model we wish to run. For all the experiments, we first load the common general experiment configs defined at `./discs/common/configs.py` and then update the values depending on the task type. `./discs/experiment/configs/lm_experiment.py` contains the experiment setup for the `text_infilling` problem. `./discs/experiment/configs/co_experiment.py` contains the common configs of the combinatorial optimization problem which depending on the problem type additional configs of the graph type are defined under their different folder in `./discs/experiment/configs/`.

**Note**: For adding new samplers, models and experiments the above configs need to get updated. To learn how to add your sampler or model to the packages, you can refer to the explanations provided in `./discs/samplers/` and `./discs/models/`.
  
Under the `./discs/samplers/` directory, you can see the list of all the samplers with their corresponding configuration under `./discs/samplers/configs/`.
List of the samplers:
* randomwalk
* hammingball
* blockgibbs
* gwg
* path_auxiliary
* dlmc (dlmcf: solver=euler_forward)
* dmala

Under the `./discs/models/` directory, you can see the list of all the models with their corresponding configuration under `./discs/models/configs/`.
List of Models
* Classical Models
    * bernoulli
    * categorical
    * ising
    * potts
    * fhmm
* Combinatorial optimization problems
    * maxcut
        * ba, er, optsicom
    * maxclique
        * rb, twitter
    * mis
        * er_density, ertest, satlib
    * normcut
        * nets: INCEPTION, ALEXNET, MNIST, RESNET, VGG
* Energy based models
    * rbm
    * resnet
    * text_infilling (language model)

**Note**: For running energy-based models, `data_path` and for combinotorial optimization problems, `data_root`, in the model config should be set. For the text infilling model, additional path of `bert_model` should be set. Further information on the `data` and how to access it can be found `data` sections below.


## Running sampling experiments
Below we provide an example of how to run a sampling experiment for different tasks by passing the name of sampler and the model. 

### Run an experiment locally 

To run an experiment locally, under the root folder `./discs/`, run:

    model=bernoulli sampler=randomwalk ./discs/experiment/run_sampling_local.sh

For combinatorial optimization problems you further need to set the graph type:

    model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

Note that for the experiments above the default config value of the sampler, model and the experiments are used.
To define your own experiment setup, you can modify the corresponding config values.


### Run an experiment on Xmanager


Under the `./discs/run_configs/` you can find predefined experiment configs for all model types which are used to study the performance of different samplers and the effect of different config values of models, samplers and the experiment. To define your own experiment config please check below section.
To run an experiment on Xmanager, under the root folder `./discs/`, run:

    config=./discs/run_configs/co/maxclique/rb_sampler_sweep.py ./discs/run_xmanager.sh

The provided example above will run all the samplers on all the `maxclique` problems with graph type of `rb`. 


#### Define your own Xmanager experiment

For defining your own Xamanger script to sweep over any of the experiment, sampler or model configs, you should follow the below structure.
```
from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='categorical',
          ## default sampler 
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [100000, 200000],
                  'model_config.num_categories': [4, 8],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                      'RATIO',
                      'MAX',
                      'MIN',
                  ],
              },
          ],
      )
  )
  return config
```
In the above example, `sampler_config.name` is used to sweep over samplers, since all of them are locally balanced function based, `sampler_config.balancing_fn_type` sweeps over the types. `config.experiment.${any experiment config you want to sweep over}` is used to sweep over experiment config, which is the chain length in the above example. `model_config.${any model config you want to sweep over}` is used to sweep over any model related config values.

### Metric, Results and Plotting
Depending on type of the model we are running the sampling on, different metrics are being calculated and the results are being stored in different forms. 
* For the `classical model`, the ESS is computed over the chains after the burn-in phase. The ESS is normalized over running time and number of energy evaluations and stored as a `csv` file.

* For `combinatorial optimization`, the objective function is being evaluated throughout the chain generation and stored in a `pickle` file. Note that for `normcut` problem, the best sample is also being stored in the `pickle` for further post processing to get the `edge cut ratio` and `balanceness`.


* For the `text_infilling` task, the generated sentences and their evaluated metrics including `bleu`, `self-bleu` and `unique-ngrams` are being stored in a pickle file. 
* For energy based models, the image of selected samples are also further saved through the chain generation.
To get the plots and arrange the results two main scripts `run_plot_results.sh` and `run_co_through_time.sh` are used that could be found at `./discs/plot_results/`.


**Note**: For detailed explanation on the metrics used and the way that they are being calculated please refer to DISCS [paper](https://openreview.net/pdf?id=9LoW5l6r4z). For reproducing the tables and figures in the paper, please refer to the explanation provided in `./discs/plot_results/` and `./discs/models/`.

## Data
The data used in this package could be found [here](https://drive.google.com/drive/u/1/folders/1nEppxuUJj8bsV9Prc946LN_buo30AnDx).
The data contains the following components:
* Graph data of combinatorial optimization problems for different graph types under `/DISCS-DATA/sco/`.
* Model parameters for energy-based models found at `DISCS-DATA/BINARY_EBM` and `DISCS-DATA/RBM_DATA`. Binray ebm is trained on MNIST, Omniglot, and Caltech dataset and binary and categorical RBM are trained on MNIST and Fashion-MNIST dataset. For the language model, you could download the model parameters from [here](https://huggingface.co/bert-base-uncased).
* Text infilling data generated from WT103 and TBC found at `/DISCS-DATA/text_infilling_data/`.


## How to add your own model, sampler and evaluator
For more details on how to plug in your sampler, model and evaluator please check the explanations under `./discs/samplers`, `./discs/models` and `./discs/evaluator` folders.

## Test

You can simply run `pytest` under the root folder to test everything.

## Contributing

We welcome pull request, please check CONTRIBUTING.md for more details.


## License
This package is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
