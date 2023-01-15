"""Tests for Path Auxiliary Sampler."""

import copy
from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.models.categorical as categorical_model
import discs.samplers.locallybalanced as lb_sampler
import discs.samplers.path_auxiliary as pas
import jax
from ml_collections import config_dict


class PASTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.sample_shape = (4,)
    self.config = config_dict.ConfigDict(dict(
        model=dict(
            shape=self.sample_shape, num_categories=2,
            init_sigma=0.5, name='bernoulli',
        ),
        sampler=dict(
            name='path_auxiliary', num_flips=1,
            adaptive=False, target_acceptance_rate=0.574,
            use_fast_path=True,
            balancing_fn_type=lb_sampler.LBWeightFn.SQRT,
        )))
    self.num_samples = 3

  def run_single_step(self, model, config, sampler):  # pylint: disable=g-unreachable-test-method
    num_samples = self.num_samples
    rng, state_rng, init_rng = jax.random.split(self.rng, 3)
    state = sampler.make_init_state(state_rng)
    param_rng, sample_rng = jax.random.split(init_rng)
    params = model.make_init_params(param_rng)
    x = model.get_init_samples(sample_rng, num_samples)
    x, state = sampler.step(model, rng, x, params, state)
    self.assertEqual(x.shape, tuple([num_samples] + list(config.model.shape)))

  def test_static_binary(self):
    config = copy.deepcopy(self.config)
    config.sampler.num_flips = 2
    sampler = pas.build_sampler(config)
    model = bernouli_model.build_model(config)
    self.run_single_step(model, config, sampler)

  def test_adaptive_binary(self):
    config = copy.deepcopy(self.config)
    config.sampler.num_flips = 2
    config.sampler.adaptive = True
    sampler = pas.build_sampler(config)
    model = bernouli_model.build_model(config)
    self.run_single_step(model, config, sampler)

  def test_adaptive_categorical(self):
    config = copy.deepcopy(self.config)
    config.sampler.adaptive = True
    config.model.name = 'categorical'
    config.model.num_categories = 5
    model = categorical_model.build_model(config)
    sampler = pas.build_sampler(config)
    self.run_single_step(model, config, sampler)

  def test_static_categorical(self):
    config = copy.deepcopy(self.config)
    config.sampler.num_flips = 2
    config.sampler.adaptive = False
    config.model.name = 'categorical'
    config.model.num_categories = 5
    model = categorical_model.build_model(config)
    sampler = pas.build_sampler(config)
    self.run_single_step(model, config, sampler)


if __name__ == '__main__':
  absltest.main()
