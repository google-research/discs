"""Combinotorial Optimization Evaluator Class."""

from discs.evaluators import abstractevaluator


class COevaluator(abstractevaluator.AbstractEvaluator):
  """Combinotorial optimization evaluator class."""

  def evaluate_step(self, samples, model, params):
    return model.evaluate(params, samples)

  def evaluate_chain(self, samples, rnd):
    return None


def build_evaluator(config):
  return COevaluator(config.experiment)
