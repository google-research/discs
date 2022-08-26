"""Tests for bernouli."""

from absl.testing import absltest
from absl.testing import parameterized
import bernouli
import ml_collections


class BernouliTest(parameterized.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    cfg = ml_collections.config_dict.ConfigDict()
    self.bernouli_model = bernouli.Bernouli(cfg)

  @parameterized.named_parameters(
      ('Bernouli Initial Params Creation', 20, 0, 1))
  def test_make_init_params(self, rnd, expected1, expected2):
    params = self.bernouli_model.make_init_params(rnd)
    self.assertGreaterEqual(params, expected1)
    self.assertLessEqual(params, expected2)

  @parameterized.named_parameters(('Bernouli Initial Samples', 0.5, 10))
  def test_get_init_samples(self, params, sz):
    result = self.bernouli_model.get_init_samples(params, sz)
    self.assertEqual(sz, result.shape[0])

  @parameterized.named_parameters(('Bernouli Forward', 0.5, 10, 0, 1))
  def test_forward(self, params, sz, expected1, expected2):
    x0 = self.bernouli_model.get_init_samples(params, sz)
    likelihood = self.bernouli_model.forward(params, x0)
    self.assertEqual(sz, likelihood.shape[0])
    self.assertGreaterEqual(likelihood.all(), expected1)
    self.assertLessEqual(likelihood.all(), expected2)


if __name__ == '__main__':
  absltest.main()
