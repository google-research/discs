# Reproducing the classical graphical models figures, CO and LM tables:
To reproduce the figures related to the classical graphical models, you need to run the `./discs/plot_results/run_plot_results.sh` script.
You need to set the `experiment_folder`, `evaluation_type`, and `key`. In the case of classical graphical models, the `evaluation_type` would be set as `ess`. `key` is going to be the config value that you want to study the samplers performance on.
As an example, if you want to get the figures related to the effect of number of categories on the potts or categorical distribution, you need to set the `key` as `num_categories`. This means that based on the provided path of `experiment_folder`, all the experiments that are only different based on `num_categories` of the model are going to be plotted in the same figure.
We have provided an example of different cases in the running script. Note that in case that you are studying the effect of the model temperature, you need to set the key as `name`.

For the cases of CO and LM experiments, running the `run_plot_results.sh` would give you a CSV file containing the summary of their experiment results. This is what we refered to when filling out the tables in the paper. We have provided an example on how to set up an CO and LM experiments in the `run_plot_results.sh` script.


# Reproducing CO problems figues:
To reproduce the CO figures, you need to run `run_plor_co_through_time.sh`. You need to only setup the `experiment_folder` and the `graph_type`.


# Adding new sampler:
To add your new sampler, you need to update a few things as below:

Updates to `./discs/plot_results/plot_utils.py`:
* In `process_keys` function, add the name of your sampler as you want it in the plots and tables.
* Update `color_map` dictionary and `get_color` function and assign the color of your new sampler.
* Update `sort_based_on_samplers` and add your sampler to the end of `sampler_list`. Note that in case of locally balanced sampler, you need to follow the similar structure of the provided example of locally balanced samplers.

Updates to `./discs/plot_results/plot_results.py`:
* In `plot_graph_cluster` function, update the `alphas`, `bar_widths`, and `x_poses` to add your new sampler to the plot.

