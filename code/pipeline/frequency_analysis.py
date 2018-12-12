import time
import os
import json
from collections import defaultdict
from typing import *
from utility_functions import get_docs


class FreqAnalyzer:

    def __init__(self, path: str) -> None:
        """Initialize the frequency analyzer.

        Args:
            path: The path to the output directory.
        """
        self.path = path
        self.path_lemma_idx_corpus = os.path.join(
            path, 'processed_corpus/lemma_idx_corpus.txt')
        self.path_df = os.path.join(
            path, 'frequencies/tf_lemmas.json')
        self.path_terms_idxs = os.path.join(
            path, 'processed_corpus/lemma_terms_idxs.txt')

    def calc_tf(self) -> None:
        """Calculate the term frequency of each term for each document.

        Create a file tf_lemmas.json of the form:
        {lemma_idx: {doc_idx: tf}}
        """
        print('start')
        term_idxs = self._load_term_idxs()
        df = defaultdict(lambda: defaultdict(int))
        print('here')
        for i, doc in enumerate(get_docs(self.path_lemma_idx_corpus)):
            for sent in doc:
                for lemma_idx in sent:
                    lemma_idx = int(lemma_idx)
                    if lemma_idx in term_idxs:
                        df[lemma_idx][i] += 1
            print(i, df[i])

        with open(self.path_df, 'w', encoding='utf8') as f:
            json.dump(df, f)

    def _load_term_idxs(self) -> Set[int]:
        term_idxs = set()
        with open(self.path_terms_idxs, 'r', encoding='utf8') as f:
            for line in f:
                term_idxs.add(int(line.strip('\n')))
        return term_idxs

    def calc_df(self):
        pass

    def calc_dl(self):
        pass


def main():
    fa = FreqAnalyzer('./output/dblp/')
    fa.calc_tf()


if __name__ == '__main__':
    main()