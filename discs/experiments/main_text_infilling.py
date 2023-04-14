"""Main script for sampling based experiments."""
import importlib
import logging
import discs.common.experiment_saver as saver_mod

from absl import app
from absl import flags
from discs.common import text_infilling_configs as common_configs
from ml_collections import config_flags

import json
import os

from transformers import BertTokenizer
import random
import numpy as np

# from discs.experiments import co_setup
_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')
FLAGS = flags.FLAGS


def get_save_dir(config):
  save_folder = config.model.get('save_dir_name', config.model.name)
  return _SAVE_DIR.value + '_' + save_folder


def get_main_config(model_config, sampler_config, sentence, infill_pos):
  config = common_configs.get_config()
  config.sampler.update(sampler_config)
  model_config['shape'] = (len(infill_pos), )
  model_config['sentence'] = sentence
  model_config['infill_pos'] = infill_pos
  config.model.update(model_config)
  
  return config

def create_infill_dataset(data_root, tokenizer, num_of_sentences=10, min_length=10, max_length=20, num_of_masks=4):
  """
     data_root: the directory where the datasets are stored (default: './text_infilling_data')
     tokenizer: the tokenizer for the language model
     num_of_sentences: the number of sentences to sample from TBC and Wiki
     min_length: the minimal length of sampled sentence
     max_length: the maximal length of sampled sentence
     num_of_masks: the number of randomly selected masks to infill words
  """
  data = []

  tbc_ref_list = []
  with open(os.path.join(data_root, 'tbc.5k.txt')) as f:
    tbc_lines = f.readlines()
    print('TBC lines:', len(tbc_lines))
    print('Before Shuffle', tbc_lines[0])
    random.shuffle(tbc_lines)
    print('After Shuffle', tbc_lines[0])
    for tbc in tbc_lines:
        if len(data) < num_of_sentences:
            tbc_new = tbc.replace("``", "")
            tbc_new = tbc_new.replace("\'\'", "")
            tbc_new = tbc_new.replace("\n", "")
            tbc_new_list = tbc_new.split(' ')
            if (len(tbc_new_list)<=max_length) and (len(tbc_new_list)>=min_length):
                infill_pos = random.sample(range(1, len(tbc_new_list)-1), num_of_masks)  
                print(tbc_new_list)
                
                for pos in infill_pos:
                    tbc_new_list[pos] = '[MASK]'
                tbc_new_masked = " ".join(tbc_new_list)
                tokens = tokenizer.tokenize(tbc_new_masked)
                infill_pos = []
                for i in range(len(tokens)):
                    if tokens[i] == '[MASK]': infill_pos.append(i+1) ### the starting token 0 will be [CLS] 

                data.append({'gt_sentence': tbc_new, 'sentence': tbc_new_masked, 'infill_pos': infill_pos})
            else:
                tbc_ref_list.append(tbc)
        else:
            tbc_ref_list.append(tbc)

  with open(os.path.join(data_root, 'tbc_remove_infill.5k.txt'), "w") as f:
      ### NOTE: we remove the sentence to be infilled from the reference dataset  to compute meaningful BLEU score
      f.writelines(tbc_ref_list)

  wiki_ref_list = []
  with open(os.path.join(data_root, 'wiki103.5k.txt')) as f:
    wiki_lines = f.readlines()
    print('WIKI lines:', len(wiki_lines))
    print('Before Shuffle', wiki_lines[0])
    random.shuffle(wiki_lines)
    print('After Shuffle', wiki_lines[0])
    for wiki in wiki_lines:
        if len(data) < (2*num_of_sentences):
            wiki_new = wiki.replace("@@unknown@@", "[UNK]")
            wiki_new = wiki_new.replace("@@UNKNOWN@@", "[UNK]")
            wiki_new = wiki_new.replace("@-@", "-")
            wiki_new = wiki_new.replace("\n", "")
            wiki_new_list = wiki_new.split(' ')
             
            if (len(wiki_new_list)<=max_length) and (len(wiki_new_list)>=min_length):
                infill_pos = random.sample(range(1, len(wiki_new_list)-1), num_of_masks)  
                
                for pos in infill_pos:
                    wiki_new_list[pos] = '[MASK]'
                wiki_new_masked = " ".join(wiki_new_list)
                tokens = tokenizer.tokenize(wiki_new_masked)
                infill_pos = []
                for i in range(len(tokens)):
                    if tokens[i] == '[MASK]': infill_pos.append(i+1) ### the starting token 0 will be [CLS] 

                data.append({'gt_sentence': wiki_new, 'sentence': wiki_new_masked, 'infill_pos': infill_pos})
            else:
                wiki_ref_list.append(wiki)
        else:
            wiki_ref_list.append(wiki)
        
  with open(os.path.join(data_root, 'wiki103_remove_infill.5k.txt'), "w") as f:
      f.writelines(wiki_ref_list)

  print('Generated Data:')
  print(data)
  with open(os.path.join(data_root, 'infilling_task.json'), 'w') as f_obj:
    json.dump(data, f_obj) 
  
#### Evaluation
from nltk.translate import bleu_score as bleu

def prepare_data(data_file, replacements={}, uncased=True):
    data = [d.strip().split() for d in open(data_file, 'r').readlines()]
    if uncased:
        data = [[t.lower() for t in sent] for sent in data]

    for k, v in replacements.items():
        data = [[t if t != k else v for t in sent] for sent in data]

    return data

def prepare_wiki(data_file, uncased=True):
    replacements = {"@@unknown@@": "[UNK]"}
    return prepare_data(data_file, replacements=replacements, uncased=uncased)

def prepare_tbc(data_file):
    replacements = {"``": "\"", "\'\'": "\""}
    return prepare_data(data_file, replacements=replacements)

def corpus_bleu(generated, references):
    """ Compute similarity between two corpora as measured by
    comparing each sentence of `generated` against all sentences in `references` 
    
    args:
        - generated (List[List[str]]): list of sentences (split into tokens)
        - references (List[List[str]]): list of sentences (split into tokens)
        
    returns:
        - bleu (float)
    """
    #print(len(generated), len(references))
    bleu2, bleu3, bleu4 = bleu.corpus_bleu([references for _ in range(len(generated))], generated, weights=[(1./2., 1./2.), (1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)])
    print('BLEUs:', bleu2, bleu3, bleu4)
    return bleu4

### Self-BLEU
from collections import Counter
from nltk.util import ngrams

def self_bleu(sents):
    return bleu.corpus_bleu([[s for (j, s) in enumerate(sents) if j != i] for i in range(len(sents))], sents)

def get_ngram_counts(sents, max_n=4):
    size2count = {}
    for i in range(1, max_n + 1):
        size2count[i] = Counter([n for sent in sents for n in ngrams(sent, i)])
    return size2count

def ref_unique_ngrams(preds, refs, max_n=4):
    # get # of *distinct* pred ngrams that don't appear in ref
    pct_unique = {}
    pred_ngrams = get_ngram_counts(preds, max_n)
    ref_ngrams = get_ngram_counts(refs, max_n)
    for i in range(1, max_n + 1):
        pred_ngram_counts = set(pred_ngrams[i].keys())
        total = sum(pred_ngrams[i].values())
        ref_ngram_counts = set(ref_ngrams[i].keys())
        pct_unique[i] = len(pred_ngram_counts.difference(ref_ngram_counts)) / total
    return pct_unique

def self_unique_ngrams(preds, max_n=4):
    # get # of pred ngrams with count 1
    pct_unique = {}
    pred_ngrams = get_ngram_counts(preds, max_n)
    for i in range(1, max_n + 1):
        n_unique = len([k for k, v in pred_ngrams[i].items() if v == 1])
        total = sum(pred_ngrams[i].values())
        pct_unique[i] = n_unique / total
    return pct_unique

def main(_):
  data_root = '/home/xcliu/ws/discrete_sampling/discs/discs/experiments/text_infilling_data'
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  if not os.path.exists(os.path.join(data_root, 'infilling_task.json')):
      print('Dataset not found! Generating dataset first')
      create_infill_dataset(data_root, tokenizer, num_of_sentences=10, min_length=15, max_length=25, num_of_masks=4)       
    
  with open(os.path.join(data_root, 'infilling_task.json'), 'r', encoding='utf-8') as f:
      infill_dataset = json.load(f)

  print('Dataset Loaded!!')
  print('Length:', len(infill_dataset))
  
  infill_sents = []
  for data in infill_dataset: 
      print(data)
      config = get_main_config(_MODEL_CONFIG.value, _SAMPLER_CONFIG.value, data['sentence'], data['infill_pos'])

      # model
      model_mod = importlib.import_module('discs.models.%s' % config.model.name)
      model = model_mod.build_model(config)

      # sampler
      sampler_mod = importlib.import_module(
          'discs.samplers.%s' % config.sampler.name
      )
      sampler = sampler_mod.build_sampler(config)

      # experiment
      experiment_mod = getattr(
          importlib.import_module('discs.experiment.experiment'),
          f'{config.experiment.name}',
      )
      experiment = experiment_mod(config)

      # evaluator
      evaluator_mod = importlib.import_module(
          'discs.evaluators.%s' % config.experiment.evaluator
      )
      evaluator = evaluator_mod.build_evaluator(config)

      # saver
      saver = saver_mod.build_saver(get_save_dir(config), config)
      
      ### TODO: dummy evaluator and saver to make the library work

      save_dir = get_save_dir(config)
      print('save dir:', save_dir)

      for seed in range(5):
          # chain generation
          sampled_infill_tokens = experiment.get_results(model, sampler, evaluator, saver, seed)
          sampled_infill_tokens = np.array(sampled_infill_tokens[0,0])

          token_ids = tokenizer(data['sentence'], return_tensors='np')['input_ids'][0]
          real_infill_pos = [pos for pos in data['infill_pos']]
          for i in range(len(real_infill_pos)):
            token_ids[real_infill_pos[i]] = sampled_infill_tokens[i]
          new_sent = tokenizer.decode(token_ids[1:-1])
          infill_sents.append(new_sent)
          print(seed, new_sent)

  ### NOTE: evaluation and save results
  results = {}
  results['infill_sents'] = infill_sents
  wiki103_file = os.path.join(data_root, 'wiki103_remove_infill.5k.txt')
  tbc_file = os.path.join(data_root, 'tbc_remove_infill.5k.txt')
  wiki_data = prepare_wiki(wiki103_file)
  tbc_data = prepare_tbc(tbc_file)

  infill_sents = [sent.strip().split() for sent in results['infill_sents']]
  results['tbc_bleu'] = corpus_bleu(infill_sents, tbc_data)
  print("TBC BLEU: %.2f" % (100 * results['tbc_bleu']))
  results['wiki_bleu'] = corpus_bleu(infill_sents, wiki_data)
  print("Wiki103 BLEU: %.2f" % (100 * results['wiki_bleu']))
  results['tbc_wiki_bleu'] = corpus_bleu(infill_sents, tbc_data[:] + wiki_data[:]) 
  print("{TBC + Wiki103} BLEU: %.2f" % (100 * results['tbc_wiki_bleu']))
  
  results['self_bleu'] = self_bleu(infill_sents)
  print("self-BLEU: %.2f" % (100 * results['self_bleu']))
  
  max_n = 4

  pct_uniques = ref_unique_ngrams(infill_sents, wiki_data, max_n)
  for i in range(1, max_n + 1):
    print("unique %d-grams relative to Wiki: %.2f" % (i, 100 * pct_uniques[i]))
    results['unique_wiki_%d_grams'%i] = pct_uniques[i]

  pct_uniques = ref_unique_ngrams(infill_sents, tbc_data, max_n)
  for i in range(1, max_n + 1):
    print("unique %d-grams relative to TBC: %.2f" % (i, 100 * pct_uniques[i]))
    results['unique_tbc_%d_grams'%i] = pct_uniques[i]

  pct_uniques = self_unique_ngrams(infill_sents, max_n)
  for i in range(1, max_n + 1):
    print("unique %d-grams relative to self: %.2f" % (i, 100 * pct_uniques[i]))
    results['unique_self_%d_grams'%i] = pct_uniques[i]

  with open(os.path.join(save_dir, 'results.json'), 'w') as f_obj:
    json.dump(results, f_obj) 

if __name__ == '__main__':
  app.run(main)
