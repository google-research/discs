"""Tests for bernouli."""

from absl.testing import absltest
from absl.testing import parameterized
import dmcx.model.bernouli as bernouli_model
import jax
from ml_collections import config_dict
import numpy as np


class BernouliTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.config = config_dict.ConfigDict(
        initial_dictionary=dict(dimension=100, init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config)
    self.rng = jax.random.PRNGKey(0)

  def test_make_init_params(self):
    params = self.bernouli_model.make_init_params(self.rng)
    self.assertEqual(params.shape, (self.config.dimension,))

  @parameterized.named_parameters(('Bernouli Initial Samples', 10))
  def test_get_init_samples(self, num_samples):
    x0 = self.bernouli_model.get_init_samples(self.rng, num_samples)
    self.assertEqual(x0.shape, (num_samples, self.config.dimension))

  @parameterized.named_parameters(('Bernouli Forward', 10))
  def test_forward(self, num_samples):
    rng_param, rng_x0 = jax.random.split(self.rng)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    energy = self.bernouli_model.forward(params, x0)
    self.assertEqual(energy.shape, (num_samples,))

  @parameterized.named_parameters(('Bernouli Value and Grad', 10))
  def test_value_grad(self, num_samples):
    rng_param, rng_x0 = jax.random.split(self.rng)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    energy, grad = self.bernouli_model.get_value_and_grad(params, x0)
    self.assertEqual(energy.shape, (num_samples,))
    self.assertEqual(grad.shape, (num_samples, self.config.dimension))
    np_param = jax.device_get(params)
    np_grad = jax.device_get(grad)
    self.assertTrue(np.allclose(np_grad, np.expand_dims(np_param, axis=0)))


if __name__ == '__main__':
  absltest.main()
