"""Tests for potts."""

from absl.testing import absltest
from absl.testing import parameterized
import discs.models.potts as potts_model
import jax
from ml_collections import config_dict
import numpy as np


class PottsTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.config = config_dict.ConfigDict(
        initial_dictionary=dict(shape=(3, 3), lambdaa=0.4407, num_categories=5))
    self.potts_model = potts_model.Potts(self.config)
    self.rng = jax.random.PRNGKey(0)
    if isinstance(self.config.shape, int):
      self.config.shape = (self.config.shape,)

  def test_make_init_params(self):
    params = self.potts_model.make_init_params(self.rng)
    self.assertEqual(params[0].shape,
                     self.config.shape + (self.config.num_categories,))
    self.assertEqual(params[1].shape,
                     self.config.shape + (self.config.num_categories,))

  @parameterized.named_parameters(('Potts Initial Samples', 10))
  def test_get_samples(self, num_samples):
    x0 = self.potts_model.get_init_samples(self.rng, num_samples)
    self.assertEqual(x0.shape, (num_samples,) + self.config.shape +
                     (self.config.num_categories,))
    self.assertTrue(
        np.array_equal(
            np.sum(x0, axis=-1), np.ones((num_samples,) + self.config.shape)))

  @parameterized.named_parameters(('Potts Forward', 10))
  def test_forward(self, num_samples):
    rng_param, rng_x0 = jax.random.split(self.rng)
    params = self.potts_model.make_init_params(rng_param)
    x0 = self.potts_model.get_init_samples(rng_x0, num_samples)
    loglikelihood = self.potts_model.forward(params, x0)
    self.assertEqual(loglikelihood.shape, (num_samples,))

  @parameterized.named_parameters(('Potts Value and Grad', 10))
  def test_value_grad_one_hot(self, num_samples):
    rng_param, rng_x0 = jax.random.split(self.rng)
    params = self.potts_model.make_init_params(rng_param)
    x0 = self.potts_model.get_init_samples(rng_x0, num_samples)
    loglikelihood, grad = self.potts_model.get_value_and_grad(params, x0)
    self.assertEqual(loglikelihood.shape, (num_samples,))
    self.assertEqual(grad.shape, (num_samples,) + self.config.shape +
                     (self.config.num_categories,))


if __name__ == '__main__':
  absltest.main()
