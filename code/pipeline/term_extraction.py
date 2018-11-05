import json
import re
import os
import sys
import math
from typing import *

import numpy as np

from text_processing_unit import TextProcessingUnit


class TermExtractor(TextProcessingUnit):
    """Class to extract relevant terms from a corpus.

    To determine relevancy, use a combination of TFIDF and
    C-Value-Method.

    A tokenized, tagged and lemmatized corpus as a collection of json-file
    is expected as input.

    For a term to be 'important' and thus get extracted it is enough to
    have a high TFIDF or a high c-value.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 threshhold_tfidf: float=0.0,
                 threshhold_cvalue: float=0.0,
                 max_files: Union[int, None]=None
                 ) -> None:
        self.threshhold_tfidf = threshhold_tfidf
        self.threshhold_cvalue = threshhold_cvalue
        self._path_temp = './temp/'
        if not os.path.isdir(self._path_temp):
            os.mkdir(self._path_temp)
        self._triples = {}  # {term: {'f_b': m, 't_b': n, 'c_b': o}}
        self._term_candidates = []
        super().__init__(path_in, path_out, max_files)
        self._num_docs = self._get_num_docs()

    def _get_documents(self) -> Iterator[str]:
        raise NotImplementedError

    def _get_num_docs(self) -> int:
        raise NotImplementedError

    # --------------- TFIDF Methods ---------------

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
        print('Counting word occurences per file...')
        word_counts = {}
        i = 0
        for doc in self._get_documents():
            for key in doc:
                sent = doc[key]
                for word in sent:
                    if word[1].startswith('N'):
                        lemma = word[2]
                        if lemma not in word_counts:
                            word_counts[lemma] = np.zeros(
                                self._num_docs, dtype=int)
                        word_counts[lemma][i] += 1
            i += 1

        path = os.path.join(self._pat_out, 'word_counts.json')
        with open(path, 'w', encoding='utf8') as f:
            json.dump(word_counts, f, ensure_ascii=False)

    def _calc_idf(self, frequencies: List[int]) -> float:
        n = self._num_docs
        df = len([c for c in frequencies if c != 0])
        return math.log(n/df)

    def _calc_tfidf(self) -> None:
        print('Calculating tfidf...')
        tfidf = {}
        path = os.path.join(self._path_out, 'word_counts.json')
        with open(path, 'r', encoding='utf8') as f:
            word_counts = json.load(f)
        for word in word_counts:
            idf = self._calc_idf(word_counts[word])
            tfidf[word] = [tf*idf for tf in word_counts[word]]
        path = os.path.join(self._path_out, 'tfidf.json')
        with open(path, 'w', encoding='utf8') as f:
            json.dump(tfidf, f, ensure_ascii=False)

    # --------------- C-Value Methods ---------------

    def _calc_cval(self) -> None:
        """Calculate the c-value for candidate terms.

        Write results to json files of the form:
        {
            'candidate_term'<str>: c-value<int>
        }
        """
        self._extract_term_candidates()
        self._build_term_info()
        length = self._get_max_len()
        max_len_candidates = self._get_candidates_len_n(length)
        cval_dict = {}                              # {(a, term): cval}
        print('start cval calculations for terms of length {}...'.format(length))
        for a in max_len_candidates:
            term_a = a.split('_')
            f_a = max_len_candidates[a]['freq']
            cval = math.log2(len(term_a))*f_a # C-Value = log2(|a|)*f(a)
            if cval > self.threshhold_cvalue: # check if bigger than threshhold
                cval_dict[a] = cval           # add term to output dict
                for b in max_len_candidates[a]['subseqs']:
                    b = '_'.join(b)
                    f_b = self._get_freq(b)
                    t_b = f_a
                    c_b = 1
                    self._triples[b] = [f_b, t_b, c_b]

        length -= 1     # all term of max_len are processed, so reduce
                        # by one to go to the next shorter terms

        # process all shorter terms in descending order
        while length != 0:
            print('start cval calculations for terms of length {}...'.format(length))
            cand_len_n = self._get_candidates_len_n(length)
            for term_a in cand_len_n:
                f_a = cand_len_n[term_a]['freq']
                if term_a not in self._triples:   # if a appears for first time
                    length_a = len(term_a.split('_'))
                    cval = math.log2(length_a) * f_a
                else:
                    triple = self._triples[term_a]
                    t_a = triple[1]
                    c_a = triple[2]
                    cval = math.log2(len(term_a.split('_'))) * f_a - (1/c_a)*t_a

                if cval > self.threshhold_cvalue:
                    cval_dict[term_a] = cval
                    for b in cand_len_n[term_a]['subseqs']:
                        b = '_'.join(b)
                        if b in self._triples:
                            self._triples[b][1] += f_a   # increase t(b)
                            self._triples[b][2] += 1     # increase c(b)
                        else:
                            # if substr not known, create new triple
                            f_b = self._get_freq(b)
                            t_b = f_a
                            c_b = 1
                            self._triples[b] = [f_b, t_b, c_b]

            length -= 1

        path = os.path.join(self._path_out, 'triples.json')
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self._triples, f, ensure_ascii=False)

        path = os.path.join(self._path_out, 'cval.json')
        with open(path, 'w', encoding='utf8') as f:
            json.dump(cval_dict, f, ensure_ascii=False)

    def _extract_term_candidates(self) -> None:
        """Extract NP-variations as term candidates.

        For each sentence:
            - get all postags and concatenate by space
            - perform a regex search looking for NP patterns
                - use re.search to get a match obj with start/end
                - since only first match is returned, remove match
                and search again until no match is found

        Store extracted terms in self._term_candidates.
        """
        print('extract term candidates...')
        for doc in self._get_documents():
            for id_, sent in doc.items():
                # construct pos tag string
                poses = ''
                for i in range(len(sent)):
                    word = sent[i]
                    pos = word[1]
                    poses += pos+str(i)+' '
                poses = poses.strip(' ')
                # search string for patterns
                # 1. pattern: sequence of Nouns
                # pattern1 = re.compile(r'((NN[PS]{0,2}) ?)+')
                # 2. pattern: adjective-noun sequence
                pattern2 = re.compile(
                    r'((NN[PS]{0,2}\d+|JJ[RS]?\d+) )+NN[PS]{0,2}\d+')
                # only use the second pattern for more recall
                # (but probably lower precision)
                for match in re.finditer(pattern2, poses):
                    pos_seq = match.group()
                    lemmas = []
                    pos_list = pos_seq.split(' ')
                    for pos in pos_list:
                        word = self._get_word_by_pos_index(sent, pos)
                        lemmas.append(word[2])  # append the lemma
                    lemmatized_term = '_'.join(lemmas)
                    self._term_candidates.append(lemmatized_term)

    def _get_word_by_pos_index(self,
                               sent: List[List[Union[str, bool]]],
                               pos: str
                               ) -> List[Union[str, bool]]:
        pattern = re.compile(r'\w+?(\d+)')
        match = re.search(pattern, pos)
        if match:
            index = int(match.group(1))
            return sent[index]
        else:
            raise Exception('Error! Pos has no index.')

    def _build_term_info(self) -> None:
        """Gather all information for a term necessary to calc c-value.

        Write the information into a json file of the form:
        {
            'the_term': {
                'length': n,
                'subseqs': ['list_of', 'subseqs', ...],
                'freq': m
            }, ...
        }
        """
        print('count candidate terms...')
        self._add_term_count()
        print('build subseq index...')
        self._add_subseq_index()
        print('calculate term lengths...')
        self._add_term_lengths()

    def _add_term_count(self) -> None:
        """Count how many times each term appears in the corpus.

        Use term_candidates.json as input file. The json file is a
        list of lists of strings which can be seen as a list of
        candidate terms.

        Write the candidate term frequencies into term_infos.json.
        """
        term_candidates = [tc for tc in self._term_candidates]
        candidate_freqs = {}
        i = 0
        for tc in term_candidates:
            if tc in candidate_freqs:
                candidate_freqs[tc]['freq'] += 1
            else:
                candidate_freqs[tc] = {}
                candidate_freqs[tc]['freq'] = 1
            i += 1

        path = os.path.join(self._path_out, 'term_info.json')
        with open(path, 'w', encoding='utf8') as f:
            json.dump(candidate_freqs, f, ensure_ascii=False)

    def _add_term_lengths(self) -> None:
        path = os.path.join(self._path_out, 'term_info.json')
        with open(path, 'r', encoding='utf8') as f:
            term_info = json.load(f)

        for term in term_info:
            term_info[term]['length'] = len(term.split('_'))

        with open(path, 'w', encoding='utf8') as f:
            json.dump(term_info, f, ensure_ascii=False)

    def _add_subseq_index(self) -> None:
        """Use term_counts to build an index of subsequences of the form:

        Output:
            A json file that maps a term id to a list containing:
                the term, a list of subsequences
            {
                '1': ['the_term', ['a', 'list']]
            }
        """
        path = os.path.join(self._path_out, 'term_info.json')
        with open(path, 'r', encoding='utf8') as f:
            term_info = json.load(f)  # term_counts: Dict[Tuple[str], int]

        # find subsequences
        for term in term_info:
            term_list = term.split('_')
            term_info[term]['subseqs'] = self._get_subsequences(term_list)

        with open(path, 'w', encoding='utf8') as f:
            json.dump(term_info, f, ensure_ascii=False)

    @staticmethod
    def _get_subsequences(term: List[str]) -> List[List[str]]:
        """Get all subsequences of the input tuple. Return list of lists."""
        subsequences = []
        for i in range(len(term) + 1):
            for j in range(i + 1, len(term) + 1):
                sub = term[i:j]
                subsequences.append(sub)
        subsequences.remove(term)
        return subsequences

    def _get_max_len(self) -> int:
        """Get the lenght of the longest term."""
        print('get max length...')
        path = os.path.join(self._path_out, 'term_info.json')
        with open(path, 'r', encoding='utf8') as f:
            term_info = json.load(f)
        max_len = max([len(t.split('_')) for t in term_info])
        return max_len

    def _get_candidates_len_n(self, n: int) -> Dict[str, str]:
        """Get all candidate terms of length n."""
        path = os.path.join(self._path_out, 'term_info.json')
        with open(path, 'r', encoding='utf8') as f:
            term_info = json.load(f)
        terms_len_n = {}
        for term in term_info:
            if term_info[term]['length'] == n:
                terms_len_n[term] = term_info[term]
        return terms_len_n

    def _get_freq(self, term: str) -> int:
        """Get the frequency of a given string in the corpus."""
        count = 0
        for term_b in self._term_candidates:
            if term == term_b:
                count += 1
        return count

    def _filter_terms(self) -> None:
        """Use tfidf and c-value to filter the candidate terms.

        Write the resulting terms into a json file 'onto_terms.json' of
        the form:
        ['term_one', 'term_two', ...]
        """
        # load candidate terms
        path_cval = os.path.join(self._path_out, 'cval.json')
        with open(path_cval, 'r', encoding='utf8') as f:
            c_values = json.load(f)
        path_tfidf = os.path.join(self._path_out, 'tfidf.json')
        with open(path_tfidf, 'r', encoding='utf8') as f:
            tfidf_values = json.load(f)
        onto_terms = []
        for term in c_values:
            if c_values[term] > 8:
                onto_terms.append(term)

        for term in tfidf_values:
            max_val = max(tfidf_values[term])
            if max_val > 17:
                onto_terms.append(term)
            elif max_val > 11:
                try:
                    if c_values[term] > 6:
                        onto_terms.append(term)
                except KeyError:
                    pass

        path_onto_terms = os.path.join(self._path_out, 'onto_terms.json')
        with open(path_onto_terms, 'w', encoding='utf8') as f:
            json.dump(onto_terms, f, ensure_ascii=False)

    def extract_important_terms(self) -> None:
        """Get a list of the most important terms in the corpus."""
        self._count_words()
        self._calc_tfidf()
        self._calc_cval()
        self._filter_terms()


doc_type = Dict[int, List[List[Union[str, bool]]]]


class EUPRLTermExtractor(TermExtractor):

    def _get_num_docs(self) -> int:
        return len(self._fnames)

    def _get_documents(self) -> Iterator[doc_type]:
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                yield json.load(f)


class DBLPTermExtractor(TermExtractor):

    def _get_num_docs(self) -> int:
        count = 0
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                titles = json.load(f)
                count += len(titles)
        return count

    def _get_documents(self) -> Iterator[doc_type]:
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                titles = json.load(f)
                for i in titles:
                    yield titles[i]


class SPTermExtractor(TermExtractor):

    def _get_num_docs(self) -> int:
        count = 0
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                summaries = json.load(f)
                count += len(summaries)
        return count

    def _get_documents(self) -> Iterator[doc_type]:
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                summaries = json.load(f)
                for i in summaries:
                    yield summaries[i]


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == '__main__':
    from utility_functions import get_corpus_config
    corpus, config = get_corpus_config('term_extraction')
    if corpus == 'dblp':
        dp = DBLPTermExtractor(**config)
        dp.extract_important_terms()
    elif corpus == 'europarl':
        ep = EUPRLTermExtractor(**config)
        ep.extract_important_terms()
    elif corpus == 'sp':
        sp = SPTermExtractor(**config)
        sp.extract_important_terms()