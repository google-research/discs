"""Abstract Evaluator Class."""

import abc
import ml_collections


class AbstractEvaluator(abc.ABC):
  """Abstract evaluator class: needs to be extended by new evaluation metrics."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config

  @abc.abstractmethod
  def evaluate(self, *args, **kwargs):
    pass
