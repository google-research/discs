# discrete_mcmc

## Installation

First follow the guideline in https://github.com/google/jax#installation to
install Jax.

Then navigate to the root of project folder and run

    pip install -e .

## Run sampling locally 

under the root folder, run 

    model=bernoulli sampler=randomwalk ./discs/experiment/run_sampling_local.sh

For combinatorial optimization problems you further need to set the graph type:

    model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

# List of samplers
* randomwalk
* hammingball
* blockgibbs
* path_auxiliary
* dlmc
* dmala

# List of Models
* Classical Models
  * bernoulli
  * categorical
  * ising
  * potts
* Combinatorial Optimization problems
  * maxcut
  * maxclique
  * mis
  * normcut

## Test

You can simply run `pytest` under the root folder to test everything.

## Contributing

We welcome pull request, please check CONTRIBUTING.md for more details.


## License
This package is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
