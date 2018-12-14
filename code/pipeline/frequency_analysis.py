import os
import json
from collections import defaultdict
from typing import *
from numpy import mean
from utility_functions import get_docs
# from text_processing_unit import TextProcessingUnit


class FreqAnalyzer:

    def __init__(self, path: str) -> None:
        """Initialize the frequency analyzer.

        Args:
            path: The path to the output directory.
        """
        self.path = path
        self.path_lemma_idx_corpus = os.path.join(
            path, 'processed_corpus/lemma_idx_corpus.txt')
        self.path_terms_idxs = os.path.join(
            path, 'processed_corpus/lemma_terms_idxs.txt')
        self.path_tf = os.path.join(
            path, 'frequencies/tf_lemmas.json')
        self.path_df = os.path.join(
            path, 'frequencies/df_lemmas.json')
        self.path_dl = os.path.join(
            path, 'frequencies/dl.json')

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

    def calc_df(self) -> None:
        """Calculate the document frequency for all terms.

        Do plus one smoothing to avoid zero division errors. Write
        output to 'frequency_analysis/df_lemmas.json' in form
        of a dict: {<term_id>: df}
        """
        term_idxs = self._load_term_idxs()
        df = defaultdict(int)
        for doc in get_docs(self.path_lemma_idx_corpus):
            for t in term_idxs:
                if self._is_in_doc(t, doc):
                    df[t] += 1

            self._docs_processed += 1
            self._update_cmd_counter()

        msg = '{} documents processed'
        print(msg.format(self._docs_processed))

        with open(self.path_df, 'w', encoding='utf8') as f:
            json.dump(df, f)

        self._docs_processed = 0

    @staticmethod
    def _is_in_doc(t: int, d: List[List[str]]) -> bool:
        """Check if term t is in document d.

        Args:
            t: index of term t
            d: index representation of document d
        """
        for s in d:
            s_int = [int(l) for l in s]
            if t in s_int:
                return True
        return False

    def calc_dl(self) -> None:
        """Compute the number of tokens for each document (doc length).

        Store the average length at index -1. Write output to
        'frequency_analysis/dl.json' in form of a dict: {doc_id: length}
        """
        dl = {}
        for doc in get_docs(self.path_lemma_idx_corpus, sent_tokenized=False):
            dl[self._docs_processed] = len(doc)
            self._docs_processed += 1
            self._update_cmd_counter()

        msg = '{} documents processed'
        print(msg.format(self._docs_processed))

        print('calculate mean length...')
        dl[-1] = mean([dl[i] for i in dl])

        with open(self.path_dl, 'w', encoding='utf8') as f:
            json.dump(dl, f)

        self._docs_processed = 0

    def _update_cmd_counter(self) -> None:
        """Update the information on the command line."""
        msg = '{} documents processed\r'
        print(msg.format(self._docs_processed), end='\r')


def main():
    fa = FreqAnalyzer('./output/dblp/')
    fa.calc_tf()
    fa.calc_df()
    fa.calc_dl()


if __name__ == '__main__':
    main()
