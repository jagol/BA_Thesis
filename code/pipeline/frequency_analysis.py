import time
import os
import json
from collections import defaultdict
from typing import *
from utility_functions import get_docs
# from text_processing_unit import TextProcessingUnit


class FreqAnalyzer():

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

        self._file_write_threshhold = 100
        # self._num_docs = 1000
        self._docs_processed = 0
        # super().__init__(path, path, max_docs)

    def calc_tf(self) -> None:
        """Calculate the term frequency of each term for each document.

        Create a file tf_lemmas.txt, which has one dictionary per line:
        {lemma_idx: tf}
        The dictionary of line 1 belongs to document 1 etc.
        """
        term_idxs = self._load_term_idxs()
        dicts = []
        for doc in get_docs(self.path_lemma_idx_corpus):
            df = defaultdict(int)
            for sent in doc:
                for lemma_idx in sent:
                    lemma_idx = int(lemma_idx)
                    if lemma_idx in term_idxs:
                        df[lemma_idx] += 1
            dicts.append(df)
            self._docs_processed += 1
            # self._update_cmd_counter()
            if self._docs_processed % self._file_write_threshhold == 0:
                msg = '{} documents processed, writing to file...'
                print(msg.format(self._docs_processed))
                mode = self._get_write_mode()
                with open(self.path_df, mode, encoding='utf8') as f:
                    for d in dicts:
                        f.write(json.dumps(d) + '\n')
                dicts = []

        self._docs_processed = 0


    def _load_term_idxs(self) -> Set[int]:
        term_idxs = set()
        with open(self.path_terms_idxs, 'r', encoding='utf8') as f:
            for line in f:
                term_idxs.add(int(line.strip('\n')))
        return term_idxs

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_processed == self._file_write_threshhold:
            return 'w'
        return 'a'

    def calc_df(self):
        pass

    def calc_dl(self):
        pass


def main():
    fa = FreqAnalyzer('./output/dblp/')
    fa.calc_tf()


if __name__ == '__main__':
    main()
