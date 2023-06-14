"""LM Factorized Energy Function."""

import json
import os
import pdb
from discs.common.customized_huggingface_flax_bert import FlaxBertForMaskedLM_Infilling
from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
from transformers import BertTokenizer


class TextInfilling(abstractmodel.AbstractModel):
  """Language Model for text infilling."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    self.infill_dataset = self.load_dataset(config.data_root)
    self.num_categories = config.num_categories  ### for bert: 30522
    self.model = FlaxBertForMaskedLM_Infilling.from_pretrained(
        config.bert_model
    )
    self.mask_token = 103
    self.random_init_sample = config.random_init_sample
    self.forward_vmap = jax.vmap(self.single_forward, [None, (0)])
    self.get_value_and_grad_vmap = jax.vmap(
        self.single_get_value_and_grad, [None, (0)]
    )

  def load_dataset(self, data_root):
    if not os.path.exists(os.path.join(data_root, 'infilling_task.json')):
      raise ValueError(
          'Dataset not found! Create the data set first using '
          'create_text_infilling_dataset!!'
      )
    with open(
        os.path.join(data_root, 'infilling_task.json'),
        'r',
        encoding='utf-8',
    ) as f:
      infill_dataset = json.load(f)
    print('Length of the DATASET is: ', len(infill_dataset))
    return iter(infill_dataset)

  def decode(self, x, params):
    sampled_infill_tokens = jnp.array(x[0])
    token_ids = params['input_ids']
    token_ids = token_ids.at[:, self.infill_pos].set(sampled_infill_tokens)
    sampled_sentence = self.tokenizer.decode(token_ids[0, 1:-1])
    return sampled_sentence

  def make_init_params(self, rnd):
    try:
      data = next(self.infill_dataset)
    except:
      return None
    self.sentence = data['sentence']
    print(self.sentence)
    self.infill_pos = data['infill_pos']  ### infill positions
    print(self.infill_pos)
    self.shape = (len(self.infill_pos),)
    inputs = self.tokenizer(self.sentence, return_tensors='jax')
    params = {}
    params['input_ids'] = inputs['input_ids']
    params['attention_mask'] = inputs['attention_mask']
    params['token_type_ids'] = inputs['token_type_ids']
    print(params['input_ids'])
    self.input_ids = params['input_ids']
    self.attention_mask = params['attention_mask']
    self.token_type_ids = params['token_type_ids']
    return params

  def get_init_samples(self, rnd, num_samples: int):
    assert (
        num_samples == 1
    )  ### currently only works with one sentence at a time

    if self.random_init_sample:
      ### NOTE: random init
      x0 = jax.random.randint(
          rnd,
          shape=(num_samples,) + self.shape,
          minval=0,
          maxval=self.num_categories,
          dtype=jnp.int32,
      )
    else:
      ### NOTE: categorical init
      mask_dummy_array = jnp.zeros(
          (1, len(self.infill_pos), self.num_categories)
      )
      mask_dummy_array = mask_dummy_array.at[:, :, self.mask_token].set(1.0)
      outputs = self.model(
          input_ids=self.input_ids,
          infill_one_hots=mask_dummy_array,
          infill_pos=self.infill_pos,
          attention_mask=self.attention_mask,
          token_type_ids=self.token_type_ids,
      )
      logits = outputs.logits
      infill_logits = logits[:, self.infill_pos, :]
      x0 = jax.random.categorical(rnd, infill_logits, axis=-1)

    return x0

  def forward(self, params, x):
    bs = x.shape[0]
    res = self.forward_vmap(params, x)
    return res.reshape(bs)

  def get_value_and_grad(self, params, x):
    bs = x.shape[0]
    print(x.shape)
    res = self.get_value_and_grad_vmap(params, x)
    ll = res[0].reshape(bs)
    return (ll, res[1])

  def single_forward(self, params, x):
    if x.shape[-1] != self.num_categories:
      x = jax.nn.one_hot(x, self.num_categories)
    x = jnp.where(x.shape[0] == len(self.infill_pos), jnp.array([x]), x)

    mask_dummy_array = jnp.zeros([1, self.num_categories])
    mask_dummy_array = mask_dummy_array.at[1, self.mask_token].set(1.0)
    loglikelihood = 0.0
    for i in range(0, len(self.infill_pos)):
      x_new = x.at[:, i, :].set(mask_dummy_array)
      outputs = self.model(
          input_ids=params['input_ids'],
          infill_one_hots=x_new,
          infill_pos=self.infill_pos,
          attention_mask=params['attention_mask'],
          token_type_ids=params['token_type_ids'],
      )
      logits = outputs.logits
      loglikelihood = loglikelihood + jnp.sum(
          jax.nn.log_softmax(logits[:, self.infill_pos[i], :], axis=-1)
          * x[:, i, :],
          axis=-1,
      )

    return loglikelihood

  def single_get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.single_forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)

    return loglikelihood, grad


def build_model(config):
  return TextInfilling(config.model)
