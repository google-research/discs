# Evaluators

The metrics used to evaluate the performance of the samplers on each type of the model are defined here.
For each problem, its corresponding metrics are defined by extending the 'AbstractEvaluator' class.
* For classical models `ESS` is used which is measured over `running time` and `number of energy evaluations`.
* For `combinatorial optimization` problems, the `objective function of the optimization` problem is used.
* For `language model`, `bleu`, `self-bleu` and `unique ngrams` are being computed.
