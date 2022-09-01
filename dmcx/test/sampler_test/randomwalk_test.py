"""Tests for bernouli."""

from absl.testing import absltest
from absl.testing import parameterized
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import jax
from ml_collections import config_dict


class RandomWalkSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.config_model = config_dict.ConfigDict(
        initial_dictionary=dict(dimension=10, init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config_model)
    # non-adaptive sampler
    self.config_sampler_nonadaprive = config_dict.ConfigDict(
        initial_dictionary=dict(
            adaptive=False, target_accept_ratio=0.234, num_flips=3))
    self.sampler_nonadaptive = randomwalk_sampler.RandomWalkSampler(
        self.config_sampler_nonadaprive)
    # # adaptive sampler
    # self.config_sampler_adaptive = config_dict.ConfigDict(
    #     initial_dictionary=dict(adaptive=True, target_accept_ratio=0.234))
    # self.sampler_adaptive = randomwalk_sampler.RandomWalkSampler(
    #     self.config_sampler_adaptive)

  @parameterized.named_parameters(('Random Walk Step Non Adaptive', 5, 3))
  def test_step_nonadaptive(self, num_samples, state):
    rng_param, rng_x0, rng_sampler = jax.random.split(self.rng, num=3)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    n_x, n_s = self.sampler_nonadaptive.step(rng_sampler, x0,
                                             self.bernouli_model, params, state)
    self.assertEqual(n_x.shape, (num_samples, self.config_model.dimension))
    self.assertEqual(n_s, state)

  # @parameterized.named_parameters(('Random Walk Step Adaptive', 5, 3))
  # def test_step_adaptive(self, num_samples, state):
  #   rng_param, rng_x0, rng_sampler = jax.random.split(self.rng, num=3)
  #   params = self.bernouli_model.make_init_params(rng_param)
  #   x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
  #   n_x, n_s = self.sampler_adaptive.step(rng_sampler, x0, self.bernouli_model,
  #                                         params, state)
  #   self.assertEqual(n_x.shape, (num_samples, self.config_model.dimension))
  #   self.assertNotEqual(n_s, state)


if __name__ == '__main__':
  absltest.main()
