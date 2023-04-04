"""Tests for Block Gibbs Sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import discs.models.bernoulli as bernouli_model
import discs.samplers.blockgibbs as blockgibbs_sampler
import jax
from ml_collections import config_dict


class GibbsSamplerTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.config_model = config_dict.ConfigDict(
        initial_dictionary=dict(shape=(3, 3), init_sigma=1.0))
    self.bernouli_model = bernouli_model.Bernouli(self.config_model)
    self.config_sampler = config_dict.ConfigDict(
        initial_dictionary=dict(
            sample_shape=(3, 3),
            num_categories=2,
            random_order=True,
            block_size=3))
    self.sampler = blockgibbs_sampler.BlockGibbsSampler(self.config_sampler)
    if isinstance(self.config_model.shape, int):
      self.sample_shape = (self.config_model.shape,)
    else:
      self.sample_shape = self.config_model.shape

  @parameterized.named_parameters(('Block Gibbs Step', 5))
  def test_step(self, num_samples):
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        self.rng, num=4)
    params = self.bernouli_model.make_init_params(rng_param)
    x0 = self.bernouli_model.get_init_samples(rng_x0, num_samples)
    state = self.sampler.make_init_state(rng_sampler)
    n_x, _ = self.sampler.step(self.bernouli_model, rng_sampler_step, x0,
                               params, state)
    self.assertEqual(n_x.shape, (num_samples,)+ self.sample_shape)


if __name__ == '__main__':
  absltest.main()
