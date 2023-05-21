"""Used to generate LM text infilling data."""
from collections.abc import Sequence
from absl import app
from discs.common import utils
from discs.models.configs import text_infilling_config as lm_config
from transformers import BertTokenizer, pipeline


def main(_):
  config = lm_config.get_config()
  tokenizer = BertTokenizer.from_pretrained(config.bert_model)
  num_of_masks = config.shape[0]
  utils.create_infill_dataset(
      config.data_root,
      tokenizer,
      num_of_masks,
      num_of_sentences=config.num_of_sentences,
      min_length=config.min_sentence_len,
      max_length=config.max_sentence_len,
  )


if __name__ == '__main__':
  app.run(main)
