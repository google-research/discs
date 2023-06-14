# discrete_mcmc

## Installation

First follow the guideline in https://github.com/google/jax#installation to
install Jax.

Then navigate to the root of project folder and run

    pip install -e .

## Sampling Experiment
To run a sampling experiment, we need to setup the model we want to sample from, setup the sampler we want to use and also define the experiment set up. To achieve that three main components are required to run an experiment:
* Model configs which are defined under `./discs/models/configs/`. For each model, its config contains the sample shape, number of categories of the sample and model specific parameters.
* Sampler_config which are defined under `./discs/sampelr/configs/`. For each sampler, its config contains the parameters required to setup the sampler.
* Experiment_config which are defined under `./discs/experiment/configs/`. Note that the experimental setup varries dependent on the type of the model you wish to run. `./discs/experiment/configs/lm_experiment.py` contains the experiment setup for the `text_infilling` problem. `./discs/experiment/configs/co_experiment.py` contains the common values of a combinatorial optimization problem. Note that for the combinotorial optimization experiment, depending on the problem and type of the graph, the experimental setup varries which are defined under their corresponding folder in `./discs/experiment/configs/`
  
    ### Run an experiment locally 

    To run an expriment locally, under the root folder ./discs/, run 

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
            * graph types: ba, er, optsicom
        * maxclique
            * graph types: rb, twitter
        * mis
            * graph types: er_density, ertest, satlib
        * normcut
            * nets: INCEPTION, ALEXNET, MNIST, RESNET, VGG
    * Energy based models
        * rbm
        * resnet
        * text_infilling (language model)

    ### Running an experiment on Xmanager
    To run an experiment on xmanager, under the root folder ./discs/

        config=./discs/run_configs/co/maxclique/rb_sampler_sweep.py ./discs/run_xmanager.sh

    Note that under the `./discs/run_configs/` you can find predifined experiment configs for all model types which are used to sweep over all the samplers. The provided example above will run all the samplers on all the `maxclique` problems with graph type of `rb`. 

By modifying the config values related to models, samplers and experiments, you can define your own set of experiments. 

## Test

You can simply run `pytest` under the root folder to test everything.

## Contributing

We welcome pull request, please check CONTRIBUTING.md for more details.


## License
This package is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
