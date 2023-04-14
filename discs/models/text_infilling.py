"""Categorical Factorized Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
import pdb
from .customized_huggingface_flax_bert import FlaxBertForMaskedLM_Infilling
from transformers import BertTokenizer, pipeline
import numpy as np
from jax import grad, jit, vmap
from jax import random


class TextInfilling(abstractmodel.AbstractModel):
  """Categorical Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape ### number of blank spaces
    self.num_categories = config.num_categories ### for bert: 30522
    self.tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    self.model = FlaxBertForMaskedLM_Infilling.from_pretrained(config.bert_model)
    self.sentence = config.sentence ### this is the original sentence before masking all the infill positions
    self.infill_pos = config.infill_pos ### infill positions

  def make_init_params(self, rnd):
    params = {}
    inputs = self.tokenizer(self.sentence, return_tensors='jax')
    params['input_ids'] = inputs['input_ids']
    print(params['input_ids'])
    params['attention_mask'] = inputs['attention_mask']
    params['token_type_ids'] = inputs['token_type_ids']
    self.input_ids = inputs['input_ids']  
    self.attention_mask = inputs['attention_mask']
    self.token_type_ids = inputs['token_type_ids']
    return params

  def get_init_samples(self, rnd, num_samples: int):
    assert num_samples == 1 ### currently only works with one sentence at a time

    ### NOTE: random init
    #x0 = jax.random.randint(
    #    rnd,
    #    shape=(num_samples,) + self.shape,
    #    minval=0,
    #    maxval=self.num_categories,
    #    dtype=jnp.int32,
    #)

    ### NOTE: max init
    mask_dummy_array = jnp.zeros((1, len(self.infill_pos), self.num_categories))
    mask_dummy_array.at[:, :, 103].set(1.)
    outputs = self.model(input_ids=self.input_ids, 
              infill_one_hots=mask_dummy_array,
              infill_pos=self.infill_pos,
              attention_mask=self.attention_mask,
              token_type_ids=self.token_type_ids,)
    logits = outputs.logits
    infill_logits = logits[:, self.infill_pos, :]
    x0 = jnp.argmax(infill_logits, axis=-1)

    return x0

  def forward(self, params, x):
    if len(x.shape) - 1 == len(self.shape):
      x = jax.nn.one_hot(x, self.num_categories)
      ### x: 1, len(infill_pos), 30522 

    mask_dummy_array = jnp.zeros((1, self.num_categories))
    mask_dummy_array.at[1, 103].set(1.) ### set to [MASK] token
    loglikelihood = 0.0
    for i in range(len(self.infill_pos)):
        x_new = x.at[:, i, :].set(mask_dummy_array)
        outputs = self.model(input_ids=params['input_ids'], 
                infill_one_hots=x_new,
                infill_pos=self.infill_pos,
                attention_mask=params['attention_mask'],
                token_type_ids=params['token_type_ids'],)
        logits = outputs.logits
        loglikelihood = loglikelihood + jnp.sum(logits[:, self.infill_pos[i], :] * x[:, i, :], axis=-1)
    
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
