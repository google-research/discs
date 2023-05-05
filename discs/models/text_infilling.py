"""LLM Factorized Energy Function."""

import json
import os
import pdb
from discs.common import utils
from discs.common.customized_huggingface_flax_bert import FlaxBertForMaskedLM_Infilling
from discs.models import abstractmodel
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from transformers import BertTokenizer, pipeline
import pdb


class TextInfilling(abstractmodel.AbstractModel):
  """Categorical Distribution."""

  def load_dataset(self, data_root, tokenizer, num_of_masks):
    if not os.path.exists(os.path.join(data_root, 'infilling_task.json')):
      print('Dataset not found! Generating dataset first')
      utils.create_infill_dataset(
          data_root,
          tokenizer,
          num_of_masks,
          num_of_sentences=10,
          min_length=15,
          max_length=25,
      )
    with open(
        os.path.join(data_root, 'infilling_task.json'), 'r', encoding='utf-8'
    ) as f:
      infill_dataset = json.load(f)
    return iter(infill_dataset)

  def __init__(self, config: ml_collections.ConfigDict):
    self.tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    num_of_mask = config.shape[0]
    self.infill_dataset = self.load_dataset(config.data_root, self.tokenizer, num_of_mask)
    self.num_categories = config.num_categories  ### for bert: 30522
    self.model = FlaxBertForMaskedLM_Infilling.from_pretrained(
        config.bert_model
    )

  def tokenizer(self):
      return self.tokenizer

  def make_init_params(self, rnd):
    try:
      data = next(self.infill_dataset)
    except:
      return None
    self.sentence = data['sentence']  ### this is the original sentence before masking all the infill positions
    self.infill_pos = data['infill_pos']  ### infill positions
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

    ### NOTE: random init
    # x0 = jax.random.randint(
    #    rnd,
    #    shape=(num_samples,) + self.shape,
    #    minval=0,
    #    maxval=self.num_categories,
    #    dtype=jnp.int32,
    # )

    pdb.set_trace()
    ### NOTE: max init
    mask_dummy_array = jnp.zeros((1, len(self.infill_pos), self.num_categories))
    mask_dummy_array = mask_dummy_array.at[:, :, 103].set(1.0)
    outputs = self.model(
        input_ids=self.input_ids,
        infill_one_hots=mask_dummy_array,
        infill_pos=self.infill_pos,
        attention_mask=self.attention_mask,
        token_type_ids=self.token_type_ids,
    )
    logits = outputs.logits
    infill_logits = logits[:, self.infill_pos, :]
    x0 = jnp.argmax(infill_logits, axis=-1)

    return x0

  def forward(self, params, x):
    if len(x.shape) - 1 == len(self.shape):
      x = jax.nn.one_hot(x, self.num_categories)
      ### x: 1, len(infill_pos), 30522

    mask_dummy_array = jnp.zeros((1, self.num_categories))
    mask_dummy_array = mask_dummy_array.at[1, 103].set(
        1.0
    )  ### set to [MASK] token
    loglikelihood = 0.0
    for i in range(len(self.infill_pos)):
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
          logits[:, self.infill_pos[i], :] * x[:, i, :], axis=-1
      )

    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)

    return loglikelihood, grad


def build_model(config):
  return TextInfilling(config.model)
