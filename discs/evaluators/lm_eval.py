"""Language Model Evaluator Class."""

from collections import Counter
import os
from discs.evaluators import abstractevaluator
from nltk.translate import bleu_score as bleu
from nltk.util import ngrams


class LMevaluator(abstractevaluator.AbstractEvaluator):
  """Language model evaluator class."""

  def prepare_data(self, data_file, replacements={}, uncased=True):
    data = [d.strip().split() for d in open(data_file, 'r').readlines()]
    if uncased:
      data = [[t.lower() for t in sent] for sent in data]

    for k, v in replacements.items():
      data = [[t if t != k else v for t in sent] for sent in data]

    return data

  def prepare_wiki(self, data_file, uncased=True):
    replacements = {'@@unknown@@': '[UNK]'}
    return self.prepare_data(
        data_file, replacements=replacements, uncased=uncased
    )

  def prepare_tbc(self, data_file):
    replacements = {'``': '"', "''": '"'}
    return self.prepare_data(data_file, replacements=replacements)

  def corpus_bleu(self, generated, references):
    """Computes similarity between two corpora as measured by comparing 
    each sentence of `generated` against all sentences in `references`."""
    bleu2, bleu3, bleu4 = bleu.corpus_bleu(
        [references for _ in range(len(generated))],
        generated,
        weights=[
            (1.0 / 2.0, 1.0 / 2.0),
            (1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0),
        ],
    )
    print('BLEUs:', bleu2, bleu3, bleu4)
    return bleu4

  def self_bleu(self, sents):
    return bleu.corpus_bleu(
        [
            [s for (j, s) in enumerate(sents) if j != i]
            for i in range(len(sents))
        ],
        sents,
    )

  def get_ngram_counts(self, sents, max_n=4):
    size2count = {}
    for i in range(1, max_n + 1):
      size2count[i] = Counter([n for sent in sents for n in ngrams(sent, i)])
    return size2count

  def ref_unique_ngrams(self, preds, refs, max_n=4):
    # get # of *distinct* pred ngrams that don't appear in ref
    pct_unique = {}
    pred_ngrams = self.get_ngram_counts(preds, max_n)
    ref_ngrams = self.get_ngram_counts(refs, max_n)
    for i in range(1, max_n + 1):
      pred_ngram_counts = set(pred_ngrams[i].keys())
      total = sum(pred_ngrams[i].values())
      ref_ngram_counts = set(ref_ngrams[i].keys())
      pct_unique[i] = (
          len(pred_ngram_counts.difference(ref_ngram_counts)) / total
      )
    return pct_unique

  def self_unique_ngrams(self, preds, max_n=4):
    # get # of pred ngrams with count 1
    pct_unique = {}
    pred_ngrams = self.get_ngram_counts(preds, max_n)
    for i in range(1, max_n + 1):
      n_unique = len([k for k, v in pred_ngrams[i].items() if v == 1])
      total = sum(pred_ngrams[i].values())
      pct_unique[i] = n_unique / total
    return pct_unique

  def evaluate(self, sentences, data_root):
    """Evaluates the generated sentences bleu+self-bleu+ngrams."""
    results = {}
    results['infill_sents'] = sentences
    wiki103_file = os.path.join(data_root, 'wiki103_remove_infill.5k.txt')
    tbc_file = os.path.join(data_root, 'tbc_remove_infill.5k.txt')
    wiki_data = self.prepare_wiki(wiki103_file)
    tbc_data = self.prepare_tbc(tbc_file)

    infill_sents = [sent.strip().split() for sent in results['infill_sents']]
    results['tbc_bleu'] = self.corpus_bleu(infill_sents, tbc_data)
    results['wiki_bleu'] = self.corpus_bleu(infill_sents, wiki_data)
    results['tbc_wiki_bleu'] = self.corpus_bleu(
        infill_sents, tbc_data[:] + wiki_data[:]
    )

    if len(infill_sents) > 1:
      results['self_bleu'] = self.self_bleu(infill_sents)

    max_n = self.config.max_n_grams

    pct_uniques = self.ref_unique_ngrams(infill_sents, wiki_data, max_n)
    for i in range(1, max_n + 1):
      results['unique_wiki_%d_grams' % i] = pct_uniques[i]

    pct_uniques = self.ref_unique_ngrams(infill_sents, tbc_data, max_n)
    for i in range(1, max_n + 1):
      results['unique_tbc_%d_grams' % i] = pct_uniques[i]

    pct_uniques = self.self_unique_ngrams(infill_sents, max_n)
    for i in range(1, max_n + 1):
      results['unique_self_%d_grams' % i] = pct_uniques[i]
    return results


def build_evaluator(config):
  return LMevaluator(config.experiment)
