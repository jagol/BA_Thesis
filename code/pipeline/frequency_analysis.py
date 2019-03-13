import os
import json
from collections import defaultdict
from typing import *
from numpy import mean
from math import log
from utility_functions import get_docs, get_num_docs
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
        self.path_tfidf_tokens = os.path.join(
            path, 'frequencies/tfidf_tokens.json')

        self.path_lemma_idx_corpus = os.path.join(
            path, 'processed_corpus/lemma_idx_corpus.txt')
        self.path_lemma_terms_idxs = os.path.join(
            path, 'processed_corpus/lemma_terms_idxs.txt')
        self.path_tf_lemmas = os.path.join(
            path, 'frequencies/tf_lemmas.json')
        self.path_df_lemmas = os.path.join(
            path, 'frequencies/df_lemmas.json')
        self.path_tfidf_lemmas = os.path.join(
            path, 'frequencies/tfidf_lemmas.json')

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

        tf = {}
        for doc_idx, doc in enumerate(get_docs(path_in)):
            tf[doc_idx] = {}
            tf_doc = tf[doc_idx]
            for sent in doc:
                for lemma_idx in sent:
                    if lemma_idx in term_idxs:
                        if lemma_idx in tf_doc:
                            tf_doc[lemma_idx] += 1
                        else:
                            tf_doc[lemma_idx] = 1
                        # print(doc_idx, lemma_idx)
                        # print(type(doc_idx), type(lemma_idx))
                        # tf[doc_id][lemma_idx] += 1
                        # tf_doc = tf[doc_idx]
                        # tf_doc[lemma_idx]
            # self._docs_processed += 1
            # self._update_cmd_counter()
            # if self._docs_processed % self._file_write_threshhold == 0:
            #     msg = '{} documents processed, writing to file...'
            #     print(msg.format(self._docs_processed), end='\r')
            #     mode = self._get_write_mode()
        with open(path_out, 'w', encoding='utf8') as f:
            json.dump(tf, f)

        self._docs_processed = 0

    def _load_term_idxs(self, level: str) -> Set[str]:
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
                term_idxs.add(line.strip('\n'))
        return term_idxs

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_processed <= self._file_write_threshhold:
            return 'w'
        return 'a'

    def calc_df(self, level: str) -> None:
        """Calculate the document frequency for all terms.

        Do plus one smoothing to avoid zero division errors. Write
        output to 'frequency_analysis/df_lemmas.json' in form
        of a dict: {<term_id>: [doc_id1, ...]}

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
        df = defaultdict(list)
        for i, doc in enumerate(get_docs(path_idx_corpus,
                                         sent_tokenized=False)):
            for word_id in set(doc):
                if word_id in term_idxs:
                    df[word_id].append(i)

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
        t = str(t)
        for s in d:
            if t in s:
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

        print('Calculate mean length...')
        dl[-1] = mean([dl[i] for i in dl])

        with open(self.path_dl, 'w', encoding='utf8') as f:
            json.dump(dl, f)

        self._docs_processed = 0

    def calc_tfidf(self, level: str) -> None:
        """Calculate tfidf values.

        Args:
            level: 't' if tokens, 'l' if lemmas.
        """
        if level == 't':
            path_tf = self.path_tf_tokens
            path_df = self.path_df_tokens
            path_tfidf = self.path_tfidf_tokens
            path_idx_corpus = self.path_token_idx_corpus
        if level == 'l':
            path_tf = self.path_tf_lemmas
            path_df = self.path_df_lemmas
            path_tfidf = self.path_tfidf_lemmas
            path_idx_corpus = self.path_lemma_idx_corpus

        with open(path_tf, 'r', encoding='utf8') as f:
            tf = json.load(f)

        with open(path_df, 'r', encoding='utf8') as f:
            df_raw = json.load(f)  # {word_id: List of docs}
            df = {}
            for term_id in df_raw:
                df[term_id] = len(df_raw[term_id])
        n = len(tf)
        corpus_docs = [str(i) for i in range(get_num_docs(path_idx_corpus))]

        tfidf = {}
        for doc_id in corpus_docs:
            tf_doc = tf[doc_id]
            tfidf[doc_id] = {}
            for word_id in tf_doc:
                df_word = df[word_id]
                tf_word_doc = tf_doc[word_id]
                tfidf[doc_id][word_id] = tf_word_doc*log(n/df_word)

        with open(path_tfidf, 'w', encoding='utf8') as f:
            json.dump(tfidf, f)

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
    print('Calculate token term frequencies...')
    fa.calc_tf('t')
    print('Calculate lemma term frequencies...')
    fa.calc_tf('l')
    print('Calculate token document frequencies...')
    fa.calc_df('t')
    print('Calculate lemma document frequencies...')
    fa.calc_df('l')
    print('Calculate token tfidf-values...')
    fa.calc_tfidf('t')
    print('Calculate lemma tfidf-values...')
    fa.calc_tfidf('l')
    print('Calculate document lengths...')
    fa.calc_dl()
    print('Done')


if __name__ == '__main__':
    main()
