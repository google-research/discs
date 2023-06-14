# DISCS

## Installation

First follow the guideline in https://github.com/google/jax#installation to
install Jax.

Then navigate to the root of project folder and run

    pip install -e .

## Sampling Experiment
To run a sampling experiment, we need to setup the model we want to sample from, setup the sampler we want to use and also define the experiment setup. To achieve that three main components are required to run an experiment in this package:
* Model configs which are defined under `./discs/models/configs/`. For each model, its corresponding config contains the shape and number of categories of the sample + the parametets specific to that model.
* Sampler config which are defined under `./discs/sampelr/configs/`. For each sampler, its corresponding config contains the parameters required to setup the sampler.
* Experiment config which are defined under `./discs/experiment/configs/`. Note that the experimental setup varries dependent on the type of the model we wish to run. `./discs/experiment/configs/lm_experiment.py` contains the experiment setup for the `text_infilling` problem. `./discs/experiment/configs/co_experiment.py` contains the common configs of the combinatorial optimization problem. Note that for the combinotorial optimization experiment, depending on the problem tpye and its corresponding graph type, the experimental setup varries which are defined under their different folder in `./discs/experiment/configs/`.
  
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

    ### Run an experiment locally 

    To run an expriment locally, under the root folder `./discs/`, run:

        model=bernoulli sampler=randomwalk ./discs/experiment/run_sampling_local.sh

    For combinatorial optimization problems you further need to set the graph type:

        model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

   
    ### Running an experiment on Xmanager
    To run an experiment on xmanager, under the root folder `./discs/`, run:

        config=./discs/run_configs/co/maxclique/rb_sampler_sweep.py ./discs/run_xmanager.sh

    Note that under the `./discs/run_configs/` you can find predifined experiment configs for all model types which are used to study the performance of different samplers and effect of different config values of models, samples and the experiment. The provided example above will run all the samplers on all the `maxclique` problems with graph type of `rb`. 

    ### Metric, Results and Plotting
    Depending on the type of the model we are running the sampling on, different metrics are being calculated and the results are being stored in different forms. 
    * For the `classical model`, the ESS is being computed over the chains after the burn-in phase. The ESS over running time and number of energy evaluations is being stored as a `csv` file. 
    * For `combinatorial optimiation`, the objective function is being evaluated through out the chain generation and stored in a `pickle` file. Note that for `normcut` problem, the best sample is also being stored in the `pickle` for further post processing to get the `edge cut ratio` and `balanceness`. 
    * For the `text_infilling` task, the generated sentences and their evaluated metrics including `bleu`, `self-bleu` and `unique-ngrams` are being stored in a pickle file. 
    * For energy based models, the image of selected samples are also further saved through the chain generation.
    To get the plots and arrange the results two main scripts `run_plot_results.sh` and `run_co_through_time.sh` are used that could be found at `./discs/plot_results/`.

## Data
The data used in this package could be found [here](https://drive.google.com/corp/drive/u/0/folders/1QvRlqRi2-BbDBZvmXStM_v_I4wvLGTJI?resourcekey=0-4D8AT7s80EPgbj6klqR-lA).
The data contains the following components:
* Graph data of combinatorial optimization problems for different graph types.
* Model parameters for resnet and rbm.
* Text infilling data
Note that `bert-base-uncased` could be downloade from [here](https://huggingface.co/bert-base-uncased)

## Test

You can simply run `pytest` under the root folder to test everything.

## Contributing

We welcome pull request, please check CONTRIBUTING.md for more details.


## License
This package is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
