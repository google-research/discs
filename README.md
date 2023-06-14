# DISCS

## Installation

First, follow the guideline in https://github.com/google/jax#installation to
install Jax.

Then navigate to the root of the project folder `./discs/` and run

    pip install -e .

## Sampling Experiment
To run a sampling experiment, we need to set up the model we want to sample from, set up the sampler we want to use and also define the experiment setup. To achieve that three main components are required to run an experiment in this package:
* Model configs which are defined under `./discs/models/configs/`. For each model, its corresponding config contains the shape and the number of categories of the sample + the parameter values specific to that model.
* Sampler config which are defined under `./discs/samplers/configs/`. For each sampler, its corresponding config contains the parameters required to setup the sampler.
* Experiment config which are defined under `./discs/experiment/configs/`. Note that the experimental setup varries dependent on the type of the model we wish to run. `./discs/experiment/configs/lm_experiment.py` contains the experiment setup for the `text_infilling` problem. `./discs/experiment/configs/co_experiment.py` contains the common configs of the combinatorial optimization problem. Note that for the combinatorial optimization experiment, depending on the problem tpye and its corresponding graph type, the experimental setup varries which are defined under their different folder in `./discs/experiment/configs/`.
  
Under the `./discs/samplers/` directory, you can see the list of all the samplers with their corresponding configuration under `./discs/samplers/configs/`.
List of the samplers:
* randomwalk
* hammingball
* blockgibbs
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

Note that, for running energy-based models, `data_path` and for combinotorial optimization problems, `data_root`, in the model config should be set. For the text infilling model, additional path of `bert_model` should be set. Further information on the data will be found in the following sections.

    ### Run an experiment locally 

    To run an experiment locally, under the root folder `./discs/`, run:

        model=bernoulli sampler=randomwalk ./discs/experiment/run_sampling_local.sh

    For combinatorial optimization problems you further need to set the graph type:

        model=maxcut graph_type=ba sampler=path_auxiliary ./discs/experiment/run_sampling_local.sh

   
    ### Running an experiment on Xmanager
    To run an experiment on xmanager, under the root folder `./discs/`, run:

        config=./discs/run_configs/co/maxclique/rb_sampler_sweep.py ./discs/run_xmanager.sh

    Note that under the `./discs/run_configs/` you can find predefined experiment configs for all model types which are used to study the performance of different samplers and effect of different config values of models, samples and the experiment. The provided example above will run all the samplers on all the `maxclique` problems with graph type of `rb`. 



    ### Metric, Results and Plotting
    Depending on the type of the model we are running the sampling on, different metrics are being calculated and the results are being stored in different forms. 
    * For the `classical model`, the ESS is computed over the chains after the burn-in phase. The ESS over running time and number of energy evaluations is being stored as a `csv` file. 
    * For `combinatorial optimization`, the objective function is being evaluated throughout the chain generation and stored in a `pickle` file. Note that for `normcut` problem, the best sample is also being stored in the `pickle` for further post processing to get the `edge cut ratio` and `balanceness`. 
    * For the `text_infilling` task, the generated sentences and their evaluated metrics including `bleu`, `self-bleu` and `unique-ngrams` are being stored in a pickle file. 
    * For energy based models, the image of selected samples are also further saved through the chain generation.
    To get the plots and arrange the results two main scripts `run_plot_results.sh` and `run_co_through_time.sh` are used that could be found at `./discs/plot_results/`.

## Data
The data used in this package could be found [here](https://drive.google.com/drive/u/1/folders/1nEppxuUJj8bsV9Prc946LN_buo30AnDx).
The data contains the following components:
* Graph data of combinatorial optimization problems for different graph types under `/DISCS-DATA/sco/`.
* Model parameters for energy-based models found at `DISCS-DATA/BINARY_EBM` and `DISCS-DATA/RBM_DATA`. Binray ebm is trained on MNIST, Omniglot, and Caltech dataset and binary and categorical RBM are trained on MNIST and Fashion-MNIST dataset. For the launguage model, you could download the model parameters from [here](https://huggingface.co/bert-base-uncased).
* Text infilling data generated from WT103 and TBC found at `/DISCS-DATA/text_infilling_data/`.


## Test

You can simply run `pytest` under the root folder to test everything.

## Contributing

We welcome pull request, please check CONTRIBUTING.md for more details.


## License
This package is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
