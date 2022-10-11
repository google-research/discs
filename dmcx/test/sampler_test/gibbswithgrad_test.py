"""Tests for Random Walk Sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.gibbswithgrad as gibbswithgrad_sampler
import jax
from ml_collections import config_dict


class GibbsWithGradSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.config_model = config_dict.ConfigDict(
        initial_dictionary=dict(shape=(20), init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config_model)
    # non-adaptive sampler
    self.config_sampler = config_dict.ConfigDict(
        initial_dictionary=dict(
            sample_shape=(2, 2),
            num_categories=2))
    self.sampler = gibbswithgrad_sampler.GibbsWithGradSampler(
        self.config_sampler)

    if isinstance(self.config_model.shape, int):
      self.sample_shape = (self.config_model.shape,)
    else:
      self.sample_shape = self.config_model.shape

  @parameterized.named_parameters(('Gibbs With Grad Step', 3))
  def test_step(self, num_samples):
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler.make_init_state(rng_sampler)
    n_x, n_s = self.sampler.step(self.bernouli_model, rng_sampler_step, x0,
                                 params, state)
    self.assertEqual(n_x.shape, (num_samples,) + self.sample_shape)
    self.assertEqual(n_s, state)

  # @parameterized.named_parameters(('Random Walk Step Adaptive', 5))
  # def test_step_adaptive(self, num_samples):
  #   rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
  #       self.rng, num=4)
  #   params = self.bernouli_model.make_init_params(rng_param)
  #   x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
  #   state = self.sampler_adaptive.make_init_state(rng_sampler)
  #   n_x, n_s = self.sampler_adaptive.step(self.bernouli_model, rng_sampler_step,
  #                                         x0, params, state)
  #   self.assertEqual(n_x.shape, (num_samples,)+ self.sample_shape)
  #   self.assertNotEqual(n_s, state)


if __name__ == '__main__':
  absltest.main()
