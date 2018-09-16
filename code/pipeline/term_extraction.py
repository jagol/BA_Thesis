import json
import os
import math
from typing import *


class TermExractor:
    """Class to extract relevant terms from a corpus.

    To determine relevancy, use a combination of TFIDF and
    C-Value-Method.

    A tokenized, tagged and lemmatized corpus as json-file is expected
    as input.

    The Extractor splits the corpus into parts (documents) to calculate
    a TFIDF-value. It operates under the assumption that important
    content words appear concentrated in one place of the corpus
    (because the word belongs to a topic talked about) whereas
    non-topic words appear everywhere in the corpus and are thus get a
    low TFIDF-value.

    Nested Terms or multiword terms are not captured by TFIDF but should
    get a high number in the c-value.

    For a term to be 'important' and thus get extracted it is enough to
    have a high TFIDF or a high c-value.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 threshhold_tfidf: float=0.8,
                 threshhold_cvalue: float=0.8,
                 max_files: Union[int, None]=None
                 ) -> None:
        self.path_in = path_in
        self._fnames = [fname for fname in os.listdir(self.path_in)
                        if os.path.isfile(os.path.join(self.path_in, fname))]
        self._fnames.sort()
        self._num_files = len(self._fnames)
        self.path_out = path_out
        self.threshhold_tfidf = threshhold_tfidf
        self.threshhold_cvalue = threshhold_cvalue
        self._max_files = max_files

    def _count_words(self) -> None:
        """Count words occurences per document.

        Count how many times a word appears in each document (tf).

        Output the result in a json file of the form:
        dict[str, List[int]]
        {
            word: [freq_doc1, ..., freq_docn]
            ...
        }
        In the case of europarl n = 8367.
        """
        word_counts = {}
        for i in range(len(self._fnames)):
            fpath = os.path.join(self.path_in, self._fnames[i])
            with open(fpath, 'r', encoding='utf8') as f:
                sent_dict = json.load(f)
                for key in sent_dict:
                    sent = sent_dict[key]
                    for word in sent:
                        if word[1].startswith('N'):
                            lemma = word[2]
                            if lemma not in word_counts:
                                word_counts[lemma] = [0]*self._num_files
                            word_counts[lemma][i] += 1

        with open('word_counts.json', 'w', encoding='utf8') as f:
            json.dump(word_counts, f, ensure_ascii=False)

    def _calc_tfidf(self) -> None:
        tfidf = {}
        with open('word_counts.json', 'r', encoding='utf8') as f:
            word_counts = json.load(f)
        for word in word_counts:
            idf = self._calc_idf(word_counts[word])
            tfidf[word] = [tf*idf for tf in word_counts[word]]
        with open('tfidf.json', 'w', encoding='utf8') as f:
            json.dump(tfidf, f, ensure_ascii=False)

    def _calc_idf(self, frequencies: List[int]) -> float:
        n = self._num_files
        df = len([c for c in frequencies if c != 0])
        return math.log(n/df)

    def _calc_cval(self) -> None:
        pass

    def _find_most_important(self):
        pass

    def extract_important_terms(self) -> None:
        """Get a list of the most important terms in the corpus."""
        self._count_words()
        self._calc_tfidf()
        self._calc_cval()
        self._find_most_important()


if __name__ == '__main__':
    path_in = './preprocessed_corpus/'
    path_out = './preprocessed_corpus/'
    te = TermExractor(path_in, path_out, 2)
    te.extract_important_terms()
