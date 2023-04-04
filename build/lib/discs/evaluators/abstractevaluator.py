"""Abstract Evaluator Class."""

import abc
import ml_collections
import pdb


class AbstractEvaluator(abc.ABC):
  """Abstract evaluator class: needs to be extended by any new objective function."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

  def _eval_over_num_ll_calls(self, eval_val, num_loglike_calls):
    return eval_val / (num_loglike_calls * self.config.ess_ratio)

  def _eval_over_mh_step(self, eval_val, mh_steps):
    return eval_val / (mh_steps * self.config.ess_ratio)

  def _eval_over_time(self, eval_val, time):
    return eval_val / (time)

  def get_eval_metrics(self, eval_val, running_time, num_ll_calls):
    """Computes objective value over time, M-H step and calls of loglike function."""

    ess_over_mh_steps = self._eval_over_mh_step(
        eval_val, self.config.chain_length
    )
    ess_over_time = self._eval_over_time(eval_val, running_time)
    ess_over_ll_calls = self._eval_over_num_ll_calls(eval_val, num_ll_calls)
    return (eval_val, ess_over_mh_steps, ess_over_time, ess_over_ll_calls)

  @abc.abstractmethod
  def evaluate(self, *args, **kwargs):
    pass
