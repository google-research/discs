"""Test math common module."""


from absl.testing import absltest
from absl.testing import parameterized
from discs.common import math
import jax
import jax.numpy as jnp


class MathTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    key = jax.random.PRNGKey(1)
    self.log_prob = jax.nn.log_softmax(
        jax.random.normal(key, shape=(2, 3, 4)) * 2.0, axis=-1
    )

  def test_multinomial(self):
    key = jax.random.PRNGKey(2)
    num_samples = 2
    idx = math.multinomial(key, self.log_prob, num_samples=num_samples,
                           replacement=True, is_nsample_const=True)
    self.assertEqual(idx.shape,
                     tuple(list(self.log_prob.shape[:-1]) + [num_samples]))
    num_samples = 4
    idx = math.multinomial(key, self.log_prob, num_samples=num_samples,
                           replacement=False, is_nsample_const=True)
    self.assertEqual(idx.shape,
                     tuple(list(self.log_prob.shape[:-1]) + [num_samples]))


if __name__ == '__main__':
  absltest.main()
