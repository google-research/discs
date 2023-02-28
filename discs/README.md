# Energy Based Model Learning and Sampling

Below, we will explain how to set up the training of the energy based models and
how to perform sampling on the trained model. Note that, we have picked RBM
model as an example.

## Learning

Set the required config values in the model's corresponding training run script
and run the script

As an example, in the case of RBM, in './discs/learning/rbm/run_rbm_training.sh'
set the required training paramets then run

```
./discs/Learning/rbm/run_rbm_training.sh
```

Once the training is done, you should be able to locate two files of
'params.pkl' and 'config.yaml' in the provided saving directiry.

## Sampling

Update 'data_path' value in the correspinding model config file. As an example
in the case of RBM, set the 'data_path' in
'./discs/models/configs/rbm_config.py' with the location where the 'params.pkl'
and 'config.yaml' files were dumped from the learning step.

Now, under the root folder, you can run the following to start the sampling from
the learned RBM model.

```
model=rbm sampler=randomwalk ./discs/experiments/run_sampling_local.sh
```
