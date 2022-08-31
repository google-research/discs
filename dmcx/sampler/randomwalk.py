import jax.numpy as jnp
import jax.random as random
import ml_collections
from dmcx.sampler import abstractsampler


class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.adaptive
    self.target_accept_ratio = config.target_accept_ratio

  def step(self, rnd, x, model, model_param, state):

    def get_new_sample(rnd, x, num_flips):

      aranged_index_batch = jnp.array([jnp.arange(x.shape[-1])] * x.shape[0])
      # which indices to flip
      index_axis1 = random.permutation(
          rnd, aranged_index_batch, axis=-1,
          independent=True)[:, 0:num_flips].reshape(1, -1)
      index_axis0 = jnp.array([jnp.arange(x.shape[0])] * num_flips).flatten('F')
      zeros = jnp.zeros(x.shape)
      # 2d array with 1 in flipped indices
      flipped = zeros.at[index_axis0, index_axis1].set(1)
      print("Flipppeeeeeeed")
      print(flipped)
      # flipping
      new_x = jnp.where(x + flipped == 2, 0, x + flipped)
      return new_x

    def get_accept_ratio(model_param, x, new_x):
      e_x = model.forward(model_param, x)
      e_new_x = model.forward(model_param, new_x)
      return jnp.exp(-e_new_x + e_x)

    def is_accepted(rnd, accept_ratio):
      u = random.uniform(rnd, shape=accept_ratio.shape, minval=0.0, maxval=1.0)
      return jnp.expand_dims(jnp.where(accept_ratio <= u, 1, 0), axis= -1)

    def update_state(accept_ratio, u, x):
      return min( max([1, u + ( jnp.mean(accept_ratio) - self.target_accept_ratio)]), x.shape[-1] )

    print("XXXXXXXXXXXXXXXXX")
    print(x)
    rnd_uniform, rnd_new_sample, rnd_acceptance = random.split(rnd, num=3)
    del rnd
    u = state
    
    if random.uniform(rnd_uniform, minval=0.0, maxval=1.0) < u - jnp.floor(u):
      num_flips = int(jnp.floor(u))
    else:
      num_flips = int(jnp.ceil(u))


    new_x = get_new_sample(rnd_new_sample, x, num_flips)
    print("newwwwwwXXXXXXXXXXXXXXXXX")
    print(new_x)
    accept_ratio = get_accept_ratio(model_param, x, new_x)
    print("accept_ratio")
    print(accept_ratio)
    accepted = is_accepted(rnd_acceptance, accept_ratio)
    print("accepted")
    print(accepted)
    
    print(jnp.shape(x), jnp.shape(new_x), jnp.shape(accepted))
    new_x = accepted * new_x + (1 - accepted) * x
    
    print("returned x")
    print(new_x)

    new_state = state
    if self.adaptive:
      new_state = update_state(accepted, u, x)
    
    return new_x, new_state
  
