# Evaluators

The metrics used to evaluate the performance of the samplers on each type of the model are defined here.
For each type of model, its corresponding metrics are defined by extending the `AbstractEvaluator` class.
* For classical models `ESS` is used which is reported over `running time` and `number of energy evaluations`.
* For `combinatorial optimization` problems, the objective function of the optimization problem is the evaluation metric.
* For `language model`, `bleu`, `self-bleu` and `unique ngrams` are being computed.


# How to add new evaluation metrics
To add new evaluation methods, extend the `AbstractEvaluator` class defined in `abstractevaluator.py` by overriding the `evaluate` method (`${method_name}.py`).
To enable the evaluaiton of the generated samples using your metric, you need to update the experiments's `evaluator` config value with `method_name`.
The default value used for the evaluation is `ess_eval` defined in `discs/common/configs.py` which is updated by `co_eval` in the case of combinotorial optimization (defined under `discs/experiment/configs/co_experiment.py`) and by `lm_eval` in the case of text_infilling task (defined under `discs/experiment/configs/lm_experiment.py`).