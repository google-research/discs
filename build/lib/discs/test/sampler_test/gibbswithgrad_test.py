"""Tests for Random Walk Sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.models.categorical as categorical_model
import discs.samplers.gibbswithgrad as gibbswithgrad_sampler
import jax
from ml_collections import config_dict
import numpy as np


class GibbsWithGradSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.num_categories = 5
    self.sample_shape = (2, 2)
    self.config_model = config_dict.ConfigDict(
        initial_dictionary=dict(
            shape=self.sample_shape,
            init_sigma=1.0,
            num_categories=self.num_categories,
            one_hot_representation=True))
    self.bernouli_model = bernouli_model.Bernouli(self.config_model)
    self.categorical_model = categorical_model.Categorical(self.config_model)

    self.config_sampler = config_dict.ConfigDict(
        initial_dictionary=dict(
            sample_shape=self.sample_shape, num_categories=self.num_categories))
    self.sampler = gibbswithgrad_sampler.GibbsWithGradSampler(
        self.config_sampler)

  @parameterized.named_parameters(('Gibbs With Grad Step Binay', 3))
  def test_step_binary(self, num_samples):
    self.sampler.num_categories = 2
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler.make_init_state(rng_sampler)
    n_x, _ = self.sampler.step(self.bernouli_model, rng_sampler_step, x0,
                               params, state)
    self.assertEqual(n_x.shape, (num_samples,) + self.sample_shape)
    self.sampler.num_categories = self.num_categories

  @parameterized.named_parameters(('Gibbs With Grad Step Categorical', 3))
  def test_step_categorical(self, num_samples):
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.categorical_model.make_init_params(rng_param)
    x0 = self.categorical_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler.make_init_state(rng_sampler)
    n_x, _ = self.sampler.step(self.categorical_model, rng_sampler_step, x0,
                               params, state)
    self.assertEqual(n_x.shape, (num_samples,) + self.sample_shape +
                     (self.num_categories,))


if __name__ == '__main__':
  absltest.main()
