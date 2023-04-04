"""Tests for Random Walk Sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.samplers.randomwalk as randomwalk_sampler
import jax
from ml_collections import config_dict


class RandomWalkSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.config_model = config_dict.ConfigDict(
        initial_dictionary=dict(shape=(10,), init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config_model)
    # non-adaptive sampler
    self.config_sampler_nonadaprive = config_dict.ConfigDict(
        initial_dictionary=dict(
            adaptive=False,
            target_acceptance_rate=0.234,
            sample_shape=(10,),
            num_categories=2))
    self.sampler_nonadaptive = randomwalk_sampler.RandomWalkSampler(
        self.config_sampler_nonadaprive)
    # adaptive sampler
    self.config_sampler_adaptive = config_dict.ConfigDict(
        initial_dictionary=dict(
            adaptive=True,
            target_acceptance_rate=0.234,
            sample_shape=(10,),
            num_categories=2))
    self.sampler_adaptive = randomwalk_sampler.RandomWalkSampler(
        self.config_sampler_adaptive)
    if isinstance(self.config_model.shape, int):
      self.sample_shape = (self.config_model.shape,)
    else:
      self.sample_shape = self.config_model.shape

  @parameterized.named_parameters(('Random Walk Step Non Adaptive', 100))
  def test_step_nonadaptive(self, num_samples):
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler_nonadaptive.make_init_state(rng_sampler)
    n_x, n_s = self.sampler_nonadaptive.step(self.bernouli_model,
                                             rng_sampler_step, x0, params,
                                             state)
    self.assertEqual(n_x.shape, (num_samples,) + self.sample_shape)
    self.assertEqual(n_s[0], state[0])

  @parameterized.named_parameters(('Random Walk Step Adaptive', 5))
  def test_step_adaptive(self, num_samples):
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler_adaptive.make_init_state(rng_sampler)
    n_x, n_s = self.sampler_adaptive.step(self.bernouli_model, rng_sampler_step,
                                          x0, params, state)
    self.assertEqual(n_x.shape, (num_samples,) + self.sample_shape)
    self.assertNotEqual(n_s[0], state[0])


if __name__ == '__main__':
  absltest.main()
