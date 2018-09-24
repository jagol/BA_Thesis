import json
import re
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
                 threshhold_tfidf: float=0.0,
                 threshhold_cvalue: float=0.0,
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
        self.triples = {}  # {term: {'f_b': m, 't_b': n, 'c_b': o}}

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
        print('extract term candidates...')
        self._extract_term_candidates()
        print('collect information on candidate terms...')
        self._build_term_info()
        print('get max length...')
        length = self._get_max_len()
        print('get candidates of maximum length {}'.format(length))
        max_len_candidates = self._get_candidates_len_n(length)
        cval_dict = {}                              # {(a, term): cval}
        print('start c_val calculations for max length terms')
        for a in max_len_candidates:
            term_a = a['term'].split(';')
            f_a = a['freq']
            cval = math.log2(len(term_a))*f_a       # C-Value = log2(|a|)*f(a)
            if cval > self.threshhold_cvalue:       # check if bigger than threshhold
                cval_dict[a['term']] = cval         # add term to output dict
                # substr_triples = self._get_triples(substrings) # (f(b), t(b), c(b))
                for b in a['subseqs']:
                    b = ';'.join(b)
                    f_b = self._get_freq(b)
                    t_b = f_a
                    c_b = 1
                    self.triples[b] = [f_b, t_b, c_b]

        length -= 1     # all term of max_len are processed, so reduce
                        # by one to go to the next shorter terms

        # process all shorter terms in descending order
        while length != 1:
            print('start cval calculations for terms of length {}'.format(length))
            for a in self._get_candidates_len_n(length):
                term_a = a['term']
                f_a = a['freq']
                if term_a not in self.triples:   # if a appears for first time
                    cval = math.log2(len(term_a.split(';'))) * f_a
                else:
                    triple = self.triples[term_a]
                    t_a = triple[1]
                    c_a = triple[2]
                    cval = math.log2(len(term_a.split(';'))) * f_a - (1/c_a)*t_a

                if cval > self.threshhold_cvalue:
                    cval_dict[term_a] = cval
                    for b in a['subseqs']:
                        b = ';'.join(b)
                        if b in self.triples:
                            self.triples[b][1] += f_a   # increase t(b)
                            self.triples[b][2] += 1     # increase c(b)
                        else:
                            # if substr not known, create new triple
                            f_b = self._get_freq(b)
                            t_b = f_a
                            c_b = 1
                            self.triples[b] = [f_b, t_b, c_b]


            length -= 1

        with open('cval.json', 'w', encoding='utf8') as f:
            json.dump(cval_dict, f, ensure_ascii=False)

    def _get_max_len(self) -> int:
        with open('term_info.json', 'r', encoding='utf8') as f:
            term_info = json.load(f)
        max_len = max([len(t['term'].split(';')) for id, t in term_info.items()])
        return max_len

    def _get_candidates_len_n(self, n) -> None:
        """Return all candidate terms of length n"""
        with open('term_info.json', 'r', encoding='utf8') as f:
            term_info = json.load(f)
        terms_len_n = []
        for i in term_info:
            term = term_info[i]
            if term['length'] == n:
                terms_len_n.append(term)
        return terms_len_n

    def _get_substrings(self, a):
        """"""
        pass

    def _extract_term_candidates(self) -> None:
        """Extract NP-variations as term candidates.

        For each sentence:
            - get all postags and concatenate by space
            - perform a regex search looking for NP patterns
                - use re.search to get a match obj with start/end
                - since only first match is returned, remove match
                and search again until no match is found
        Write found terms to json file in the format:
        {
            1: ['first', 'candidate', 'term'],
            2: ['second', 'candidate', 'term'],
            ...
        }
        """
        extracted_terms = []
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                sents = json.load(f)
            for id_, sent in sents.items():
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
                pattern2 = re.compile(r'((NN[PS]{0,2}\d|JJ[RS]?\d) )+NN[PS]{0,2}\d')
                # matches1 = re.search(pattern1, poses)
                # only use the second pattern for more recall
                # (but probably lower precision)
                match = True    # starting value of match can be anything
                                # that evalutates to True if used as a bool
                # look for matches and then remove them until no matches are found
                matches = []
                while match:
                    match = re.search(pattern2, poses)
                    if match:
                        matches.append(match.group())
                        # remove the matched pos-tags from poses string
                        poses = poses[:match.start()]+poses[match.end():]

                # find the lemmas corresponding to the matched pos tags
                for pos_seq in matches:
                    lemmas = []
                    pos_list = pos_seq.split(' ')
                    for pos in pos_list:
                        pos_id = int(pos[-1])
                        word = sent[pos_id]
                        lemmas.append(word[2]) # append the lemmma
                    extracted_terms.append(lemmas)

        with open('term_candidates.json', 'w', encoding='utf8') as f:
            json.dump(dict(enumerate(extracted_terms)), f, ensure_ascii=False)

    def _add_term_count(self) -> None:
        """Count how many times each term appears in the corpus.

        Use term_candidates.json as input file. The json file is a
        list of lists of strings which can be seen as a list of
        candidate terms.

        Format:
        {
            '1': {
                'term': ['a', 'term'],
                'freq': n
                }
        }
        """
        with open('term_candidates.json', 'r', encoding='utf8') as f:
            term_candidates = json.load(f)

        term_candidates = [';'.join(tc) for id_, tc in term_candidates.items()]

        candidate_freqs = {}
        i = 0
        for tc in term_candidates:
            if tc in candidate_freqs:
                candidate_freqs[i]['freq'] += 1
            else:
                candidate_freqs[i] = {}
                candidate_freqs[i]['term'] = tc
                candidate_freqs[i]['freq'] = 1
            i += 1

        with open('term_info.json', 'w', encoding='utf8') as f:
            json.dump(candidate_freqs, f, ensure_ascii=False)

    def _build_term_info(self) -> None:
        """Gather all information for a term necessary to calc c-value.

        Create a json file, which contains for each term:
        - an id
        - a list of all subsequences of a term
        - the length of the term
        - the frequency of the term
        - the triple (f(b), t(b), c(b))

        form:
        {
            id: {
                'term': 'the;term',
                'length': n,
                'subseqs': ['list', 'of', 'subseqs'],
                'f': k,
                't': l,
                'c': i
            }, ...
        }
        """
        print('count candidate terms...')
        self._add_term_count()
        print('build subseq index...')
        self._add_subseq_index()
        print('calculate term lengths')
        self._add_term_lengths()

    def _add_term_lengths(self) -> None:
        with open('term_info.json', 'r', encoding='utf8') as f:
            term_info = json.load(f)

        for i in term_info:
            term = term_info[i]['term']
            term_info[i]['length'] = len(term.split(';'))

        with open('term_info.json', 'w', encoding='utf8') as f:
            json.dump(term_info, f)

    def _add_subseq_index(self) -> None:
        """Use term_counts to build an index of subsequences of the form:

        Args:
            term_counts: a dict that maps terms onto their frequency in
                in the corpus.
        Output:
            A json file that maps a term id to a list containing:
                the term, a list of subsequences
            {
                '1': ['the;term', ['a', 'list']]
            }
        """
        with open('term_info.json', 'r', encoding='utf8') as f:
            term_info = json.load(f) # term_counts: Dict[Tuple[str], int]

        # find subsequences
        for i in term_info:
            term = term_info[i]['term'].split(';')
            term_info[i]['subseqs'] = self._get_subsequences(term)

        # build index for subsequences
        for i in term_info:
            for j in term_info:
                subseqs_i = term_info[i]['subseqs']
                term_info[i]['subseq_index'] = []
                term_j = term_info[j]['term'].split(';')
                if term_j in subseqs_i:
                    term_info[i]['subseq_index'].append(j)

        with open('term_info.json', 'w', encoding='utf8') as f:
            json.dump(term_info, f, ensure_ascii=False)

    @staticmethod
    def _get_subsequences(term):
        """Get all subsequences of the input tuple. Return list of tuples."""
        subsequences = []
        for i in range(len(term) + 1):
            for j in range(i + 1, len(term) + 1):
                sub = term[i:j]
                subsequences.append(sub)
        subsequences.remove(term)
        return subsequences

    def _get_triples(self, substrings):
        # (f(b), t(b), c(b))
        triples = []
        for substr in substrings:
            triples.append((self._get_freq(substr), self._get_freq(substr), 1))
        return triples

    def _get_freq(self, substr: str) -> int:
        """"""
        with open('term_candidates.json', 'r', encoding='utf8') as f:
            term_candidates = json.load(f)

        count = 0
        for i in term_candidates:
            if substr == term_candidates[i]:
                count += 1
        return count

    def extract_important_terms(self) -> None:
        """Get a list of the most important terms in the corpus."""
        # self._count_words()
        # self._calc_tfidf()
        self._calc_cval()

if __name__ == '__main__':
    path_in = './preprocessed_corpus/'
    path_out = './preprocessed_corpus/'
    te = TermExractor(path_in, path_out, 2)
    te.extract_important_terms()
