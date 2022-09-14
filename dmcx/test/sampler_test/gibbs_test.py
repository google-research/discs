"""Tests for Random Walk Sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.blockgibbs as blockgibbs_sampler
import jax
from ml_collections import config_dict


class GibbsSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.config_model = config_dict.ConfigDict(
        initial_dictionary=dict(dimension=10, init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config_model)
    self.config_sampler = config_dict.ConfigDict(
        initial_dictionary=dict(
            sample_dimension=10,
            num_categories=2,
            random_order=False,
            block_size=3))
    self.sampler = blockgibbs_sampler.BlockGibbsSampler(self.config_sampler)

  @parameterized.named_parameters(('Random Walk Step Non Adaptive', 4))
  def test_step(self, num_samples):
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler.make_init_state(rng_sampler)
    n_x, _ = self.sampler.step(self.bernouli_model, rng_sampler_step, x0,
                               params, state)
    self.assertEqual(n_x.shape, (num_samples, self.config_model.dimension))


if __name__ == '__main__':
  absltest.main()
