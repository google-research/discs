"""Combinotorial Optimization Evaluator Class."""

from discs.evaluators import abstractevaluator


class COevaluator(abstractevaluator.AbstractEvaluator):
  """Combinotorial optimization evaluator class."""

  def evaluate(self, samples, model, params):
    return model.forward(params, samples) * params['temperature']


def build_evaluator(config):
  return COevaluator(config.experiment)
