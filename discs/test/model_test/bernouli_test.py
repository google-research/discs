"""Tests for bernouli."""

from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import jax
from ml_collections import config_dict
import numpy as np


class BernouliTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.config = config_dict.ConfigDict(
        initial_dictionary=dict(shape=(100, 100), init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config)
    self.rng = jax.random.PRNGKey(0)
    if isinstance(self.config.shape, int):
      self.config.shape = (self.config.shape,)

  def test_make_init_params(self):
    params = self.bernouli_model.make_init_params(self.rng)
    self.assertEqual(params.shape, self.config.shape)

  @parameterized.named_parameters(('Bernouli Initial Samples', 10))
  def test_get_init_samples(self, num_samples):
    x0 = self.bernouli_model.get_init_samples(self.rng, num_samples)
    self.assertEqual(x0.shape, (num_samples,) + self.config.shape)

  @parameterized.named_parameters(('Bernouli Forward', 10))
  def test_forward(self, num_samples):
    rng_param, rng_x0 = jax.random.split(self.rng)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    loglikelihood = self.bernouli_model.forward(params, x0)
    self.assertEqual(loglikelihood.shape, (num_samples,))

  @parameterized.named_parameters(('Bernouli Value and Grad', 10))
  def test_value_grad(self, num_samples):
    rng_param, rng_x0 = jax.random.split(self.rng)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    loglikelihood, grad = self.bernouli_model.get_value_and_grad(params, x0)
    self.assertEqual(loglikelihood.shape, (num_samples,))
    self.assertEqual(grad.shape, (num_samples,) + self.config.shape)
    np_param = jax.device_get(params)
    np_grad = jax.device_get(grad)
    self.assertTrue(np.allclose(np_grad, np.expand_dims(np_param, axis=0)))

  def test_get_expected_val(self):
    rng_param, _ = jax.random.split(self.rng)
    params = self.bernouli_model.make_init_params(rng_param)
    expected_val = self.bernouli_model.get_expected_val(params)
    self.assertEqual(expected_val.shape, self.config.shape)

  def test_get_var(self):
    rng_param, _ = jax.random.split(self.rng)
    params = self.bernouli_model.make_init_params(rng_param)
    variance = self.bernouli_model.get_var(params)
    self.assertEqual(variance.shape, self.config.shape)


if __name__ == '__main__':
  absltest.main()
