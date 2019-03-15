import os
import json
from typing import *
from collections import defaultdict


class Pruner:

    def __init__(self, path_out: str, min_count: int = 3) -> None:
        """Initialize the pruner."""
        self.path_out = path_out
        self.path_token_terms = os.path.join(
            path_out, 'processed_corpus/token_term_idxs.txt')
        self.path_lemma_terms = os.path.join(
            path_out, 'processed_corpus/lemma_term_idxs.txt')
        self.path_tf_tokens = os.path.join(
            path_out, 'frequencies/tf_tokens.json')
        self.path_tf_lemmas = os.path.join(
            path_out, 'frequencies/tf_lemmas.json')
        self.min_count = min_count

    def prune_tf(self)-> None:
        """Filter all terms under min_count.

        Replace old tf-file.
        """
        with open(self.path_tf_tokens, 'r', encoding='utf8') as f:
            tf_tokens = json.load(f)
            tf_tokens_pruned, token_terms_pruned = self._prune_tf(tf_tokens)
            pdb.set_trace()
        with open(self.path_tf_tokens, 'w', encoding='utf8') as f:
            json.dump(tf_tokens_pruned, f)
        with open(self.path_token_terms, 'w', encoding='utf8') as f:
            for term_id in token_terms_pruned:
                f.write(term_id + '\n')
        with open(self.path_tf_lemmas, 'r', encoding='utf8') as f:
            tf_lemmas = json.load(f)
            tf_lemmas_pruned, lemma_terms_pruned = self._prune_tf(tf_lemmas)
        with open(self.path_tf_lemmas, 'w', encoding='utf8') as f:
            json.dump(tf_lemmas_pruned, f)
        with open(self.path_lemma_terms, 'w', encoding='utf8') as f:
            for term_id in lemma_terms_pruned:
                f.write(term_id + '\n')

    def _prune_tf(self,
                  tf: Dict[str, Dict[str, int]]
                  ) -> Tuple[Dict[str, Dict[str, int]], Set[str]]:
        """Filter all term under min_count.

        Args:
            tf: Term-frequency dictionary.
        Return:
            A tuple of the filtered dictionary and a set of all term-ids
            with a count above the min_count.
        """
        terms_pruned = set()
        tf_global = defaultdict(int)
        tf_pruned = {}

        for doc_id in tf:
            for word_id in tf[doc_id]:
                tf_global[word_id] += tf[doc_id][word_id]

        for doc_id in tf:
            tf_pruned[doc_id] = {}
            for word_id in tf[doc_id]:
                if tf_global[word_id] >= self.min_count:
                    tf_pruned[doc_id][word_id] = tf[doc_id][word_id]
                    terms_pruned.add(word_id)

        return tf_pruned, terms_pruned
