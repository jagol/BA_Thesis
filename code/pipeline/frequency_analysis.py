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

        self.path_token_idx_corpus = os.path.join(
            path, 'processed_corpus/token_idx_corpus.txt')
        self.path_token_terms_idxs = os.path.join(
            path, 'processed_corpus/token_terms_idxs.txt')
        self.path_tf_tokens = os.path.join(
            path, 'frequencies/tf_tokens.json')
        self.path_df_tokens = os.path.join(
            path, 'frequencies/df_tokens.json')

        self.path_lemma_idx_corpus = os.path.join(
            path, 'processed_corpus/lemma_idx_corpus.txt')
        self.path_lemma_terms_idxs = os.path.join(
            path, 'processed_corpus/lemma_terms_idxs.txt')
        self.path_tf_lemmas = os.path.join(
            path, 'frequencies/tf_lemmas.json')
        self.path_df_lemmas = os.path.join(
            path, 'frequencies/df_lemmas.json')

        self.path_dl = os.path.join(
            path, 'frequencies/dl.json')

        self._file_write_threshhold = 100
        # self._num_docs = 1000
        self._docs_processed = 0
        # super().__init__(path, path, max_docs)

    def calc_tf(self, level) -> None:
        """Calculate the term frequency of each term for each document.

        Create a file tf_lemmas.txt, which has one dictionary per line:
        {lemma_idx: tf}
        The dictionary of line 1 belongs to document 1 etc.
        """
        if level == 't':
            term_idxs = self._load_term_idxs('t')
            path_in = self.path_token_idx_corpus
            path_out = self.path_tf_tokens
        elif level == 'l':
            term_idxs = self._load_term_idxs('l')
            path_in = self.path_lemma_idx_corpus
            path_out = self.path_tf_lemmas

        dicts = []
        for doc in get_docs(path_in):
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
                with open(path_out, mode, encoding='utf8') as f:
                    for d in dicts:
                        f.write(json.dumps(d) + '\n')
                dicts = []

        self._docs_processed = 0

    def _load_term_idxs(self, level: str) -> Set[int]:
        """Load term indices from file.

        Args:
            level: 't' if tokens, 'l' if lemmas.
        """
        term_idxs = set()
        if level == 't':
            path = self.path_token_terms_idxs
        elif level == 'l':
            path = self.path_lemma_terms_idxs

        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                term_idxs.add(int(line.strip('\n')))
        return term_idxs

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_processed == self._file_write_threshhold:
            return 'w'
        return 'a'

    def calc_df(self, level: str) -> None:
        """Calculate the document frequency for all terms.

        Do plus one smoothing to avoid zero division errors. Write
        output to 'frequency_analysis/df_lemmas.json' in form
        of a dict: {<term_id>: df}

        Args:
            level: 't' if tokens, 'l' if lemmas.
        """
        if level == 't':
            path_df = self.path_df_tokens
            path_idx_corpus = self.path_token_idx_corpus
        elif level == 'l':
            path_df = self.path_df_lemmas
            path_idx_corpus = self.path_lemma_idx_corpus

        term_idxs = self._load_term_idxs(level)
        df = defaultdict(int)
        for doc in get_docs(path_idx_corpus):
            for t in term_idxs:
                if self._is_in_doc(t, doc):
                    df[t] += 1

            self._docs_processed += 1
            self._update_cmd_counter()

        msg = '{} documents processed'
        print(msg.format(self._docs_processed))
        with open(path_df, 'w', encoding='utf8') as f:
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
    from utility_functions import get_config, get_cmd_args
    config = get_config()
    args = get_cmd_args()
    path = config['paths'][args.location][args.corpus]['path_out']
    fa = FreqAnalyzer(path)
    fa.calc_tf('t')
    fa.calc_tf('l')
    fa.calc_df('t')
    fa.calc_df('l')
    fa.calc_dl()


if __name__ == '__main__':
    main()
