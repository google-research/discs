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
        jax.random.normal(key, shape=(7, 3, 4)) * 2.0, axis=-1
    )

  def test_multinomial(self):
    key = jax.random.PRNGKey(2)
    num_samples = 2
    idx = math.multinomial(key, self.log_prob, num_samples=num_samples,
                           replacement=True, is_nsample_const=True)
    self.assertEqual(idx.shape, self.log_prob.shape[:-1] + (num_samples,))
    num_samples = 4
    idx = math.multinomial(key, self.log_prob, num_samples=num_samples,
                           replacement=False, is_nsample_const=True)
    self.assertEqual(idx.shape, self.log_prob.shape[:-1] + (num_samples,))
    num_samples = jnp.array(2, dtype=jnp.int32)
    idx = math.multinomial(key, self.log_prob, num_samples=num_samples,
                           replacement=False, is_nsample_const=False)
    self.assertEqual(idx.shape, self.log_prob.shape)
    num_samples = jnp.array(5, dtype=jnp.int32)
    idx = math.multinomial(key, self.log_prob, num_samples=num_samples,
                           replacement=True, is_nsample_const=False,
                           batch_size=3)
    self.assertEqual(idx.shape, self.log_prob.shape)

  def test_multinomial_ll(self):
    key = jax.random.PRNGKey(2)
    num_samples = 2
    idx, ll_idx = math.multinomial(
        key, self.log_prob, num_samples=num_samples, replacement=True,
        is_nsample_const=True, return_ll=True)
    self.assertEqual(idx.shape, self.log_prob.shape[:-1] + (num_samples,))
    self.assertEqual(ll_idx.shape, idx.shape)
    idx, ll_idx = math.multinomial(
        key, self.log_prob, num_samples=num_samples, replacement=False,
        is_nsample_const=True, return_ll=True)
    self.assertEqual(idx.shape, self.log_prob.shape[:-1] + (num_samples,))
    self.assertEqual(ll_idx.shape, idx.shape)

    num_samples = jnp.array(2, dtype=jnp.int32)
    idx, ll_idx = math.multinomial(
        key, self.log_prob, num_samples=num_samples, replacement=True,
        is_nsample_const=False, return_ll=True)
    self.assertEqual(idx.shape, self.log_prob.shape)
    self.assertEqual(ll_idx.shape, idx.shape)

    idx, ll_idx = math.multinomial(
        key, self.log_prob, num_samples=num_samples, replacement=False,
        is_nsample_const=False, return_ll=True)

    self.assertEqual(idx.shape, self.log_prob.shape)
    self.assertEqual(ll_idx.shape, self.log_prob.shape)


if __name__ == '__main__':
  absltest.main()
