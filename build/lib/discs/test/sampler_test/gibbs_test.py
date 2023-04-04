"""Tests for Gibbs Sampler."""

import copy
from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.samplers.gibbs as gibbs_sampler
import jax
from ml_collections import config_dict


class GibbsSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.num_categories = 2
    self.sample_shape = (2, 3)
    self.config = config_dict.ConfigDict(dict(
        model=dict(
            shape=self.sample_shape, num_categories=self.num_categories,
            init_sigma=0.5, name='bernoulli',
        ),
        sampler=dict(
            name='gibbs',
        )))
    self.model = bernouli_model.build_model(self.config)
    self.num_samples = 1

  def run_single_step(self, config, sampler):  # pylint: disable=g-unreachable-test-method
    num_samples = self.num_samples
    rng, state_rng, init_rng = jax.random.split(self.rng, 3)
    state = sampler.make_init_state(state_rng)
    param_rng, sample_rng = jax.random.split(init_rng)
    params = self.model.make_init_params(param_rng)
    x = self.model.get_init_samples(sample_rng, num_samples)
    x, state = sampler.step(self.model, rng, x, params, state)
    self.assertEqual(x.shape, tuple([num_samples] + list(config.model.shape)))

  def test_binary(self):
    config = copy.deepcopy(self.config)
    sampler = gibbs_sampler.build_sampler(self.config)
    self.run_single_step(config, sampler)


if __name__ == '__main__':
  absltest.main()
