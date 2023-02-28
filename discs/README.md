# Deep Energy Based Models Learning and Sampling

Below, we will explain how to train the deep energy based models and how to
perform sampling on the learned model. Note that, we have picked RBM model as an
example.

## Learning

You should run the training script of the corresponsing model you want. Before
that, make sure to set the required variables needed in the script. Set the
required config values in the model's corresponding training run script and run
the script

As an example, in the case of RBM, set your desired training parameters in
`./discs/learning/rbm/run_rbm_training.sh` and then run

```
./discs/learning/rbm/run_rbm_training.sh
```

Once the training is done, you should be able to locate two files of
`params.pkl` and `config.yaml` in the `save_root` set in the training script
above.

## Sampling

Update `data_path` value with the trained model saved directory in the
correspinding model config file located at `./discs/models/configs/`. As an
example in the case of RBM, set the `data_path` in
`./discs/models/configs/rbm_config.py` with the location where the `params.pkl`
and `config.yaml` files were dumped from the learning step.

Under the root folder, you can run the following to start the sampling from the
learned RBM model:

```
model=rbm sampler=randomwalk ./discs/experiments/run_sampling_local.sh
```
