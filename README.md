# discrete_mcmc

## Installation

First follow the guideline in https://github.com/google/jax#installation to
install Jax.

Then navigate to the root of project folder and run

    pip install -e .

## Experiment
To run an experiment, three main components are required:
* model: you need to set the path to the config values of the model.
* sampler: you need to set the path to the config values of the model.
* experiment configs

## Run sampling locally 

To run an expriment locally under the root folder ./discs/, run 

    model=bernoulli sampler=randomwalk ./discs/experiment/run_sampling_local.sh

For combinatorial optimization problems you further need to set the graph type:

    model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

Under the `./discs/samplers/` directory, you can see the list of all the samplers with their corresponding configuration under `./discs/samplers/configs/`.
List of the samplers:
* randomwalk
* hammingball
* blockgibbs
* path_auxiliary
* dlmc
* dmala

Under the `./discs/models/` directory, you can see the list of all the models with their corresponding configuration under `./discs/samplers/configs/`.
List of Models
* Classical Models
  * bernoulli
  * categorical
  * ising
  * potts
  * fhmm
* Combinatorial optimization problems
  * maxcut
  * maxclique
  * mis
  * normcut
* Energy based models
  * rbm
  * resnet
  * text_infilling (language model)



## Test

You can simply run `pytest` under the root folder to test everything.

## Contributing

We welcome pull request, please check CONTRIBUTING.md for more details.


## License
This package is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
