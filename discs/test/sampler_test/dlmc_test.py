"""Tests for Path Auxiliary Sampler."""

import copy
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.models.categorical as categorical_model
from discs.samplers import dlmc
import discs.samplers.locallybalanced as lb_sampler
import jax
from ml_collections import config_dict


class DLMCTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.sample_shape = (5, 4)
    self.config = config_dict.ConfigDict(dict(
        sampler=dict(
            name='dlmc', init_log_tau=0.0, solver='interpolate',
            adaptive=False, target_acceptance_rate=0.574, logz_ema=0.9,
            balancing_fn_type=lb_sampler.LBWeightFn.SQRT,
        )))
    self.num_samples = 3

  def run_single_step(self, model, sampler):  # pylint: disable=g-unreachable-test-method
    rng, state_rng, init_rng = jax.random.split(self.rng, 3)
    state = sampler.make_init_state(state_rng)
    param_rng, sample_rng = jax.random.split(init_rng)
    params = model.make_init_params(param_rng)
    x = model.get_init_samples(sample_rng, self.num_samples)
    x, state = sampler.step(model, rng, x, params, state)
    return x, state

  def test_adaptive_binary(self):
    config = copy.deepcopy(self.config)
    config.sampler.adaptive = True
    config.model = config_dict.ConfigDict(dict(
        shape=self.sample_shape, num_categories=2,
        init_sigma=0.5, name='bernoulli'))
    sampler = dlmc.build_sampler(config)
    model = bernouli_model.build_model(config)
    x, _ = self.run_single_step(model, sampler)
    self.assertEqual(x.shape,
                     tuple([self.num_samples] + list(config.model.shape)))

  def test_adaptive_categorical(self):
    config = copy.deepcopy(self.config)
    config.sampler.adaptive = True
    config.model = config_dict.ConfigDict(dict(
        shape=self.sample_shape, num_categories=7,
        init_sigma=0.5, name='categorical'))
    sampler = dlmc.build_sampler(config)
    model = categorical_model.build_model(config)
    x, _ = self.run_single_step(model, sampler)
    self.assertEqual(x.shape,
                     tuple([self.num_samples] + list(config.model.shape)))
