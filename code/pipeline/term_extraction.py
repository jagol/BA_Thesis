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
        self._extract_term_candidates()
        term_counts = self._count_term_candidates() # {(term, count): frequency}
        self._build_subseq_index(term_counts)
        max_len_candidates = self._get_max_len_candidates()
        # print(max_len_candidates)
        cval_dict = {}  # {(a, term): cval}
        for a in max_len_candidates:
            a = tuple(a)
            cval = math.log2(len(a))*term_counts[a] # C-Value = log2(|a|)*f(a)
            cval_dict[a] = cval
            substrings = self._get_substrings(a)
            substr_triples = self._get_triples(substrings) # (f(b), t(b), c(b))
            for t in substr_triples:
                pass

    def _get_substrings(self, a):
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
            json.dump(dict(enumerate(extracted_terms)), f)

    def _count_term_candidates(self) -> Dict[Tuple[str], int]:
        """Count how many times each term appears in the corpus.

        Use term_candidates.json as input file. The json file is a
        list of lists of strings which can be seen as a list of
        candidate terms.
        """
        with open('term_candidates.json', 'r', encoding='utf8') as f:
            term_candidates = json.load(f)
        # format term candidates to list of lists: [[cand, 1], [cand,  2], ...]
        term_candidates = [tuple(tc) for id_, tc in term_candidates.items()]

        candidate_freqs = {}
        for tc in term_candidates:
            if tc in candidate_freqs:
                candidate_freqs[tc] += 1
            else:
                candidate_freqs[tc] = 1

        return candidate_freqs

    def _build_subseq_index(self, term_counts: Dict[Tuple[str], int]) -> None:
        """Use term_counts to build an index of subsequences of the form:

        Args:
            term_counts: a dict that maps terms onto their frequency in
                in the corpus.
        Output:
            A json file that maps a term id to a list containing:
                the term, a list of subsequences, a triple containing
                (f(b), t(b), c(b))
            {
                '1': [['the', 'term'], ['a', 'list']]
            }
        """
        substr_index = {}

        i = 0
        for term in term_counts:
            subsequences = self._get_subsequences(term)
            triples = self._get_triples(subsequences)
            substr_index[i] = [term, subsequences, triples, []]
            i += 1

        # find subsequences
        for i in substr_index:
            term = substr_index[i][0]
            for j in substr_index:
                subsequences = substr_index[j][1]
                if term in subsequences:
                    substr_index[j][3].append(i)
        with open('subseq_index.json', 'w', encoding='utf8') as f:
            json.dump(substr_index, f)

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
        pass

    def _get_max_len_candidates(self) -> List[List[str]]:
        with open('subseq_index.json', 'r', encoding='utf8') as f:
            term_candidates = json.loads(f.read())
        term_candidates = term_candidates.items()
        max_len = max([len(tc[0]) for id_, tc in term_candidates])
        max_len_candidates = []
        for id_, tc in term_candidates:
            if len(tc[0]) == max_len:
                max_len_candidates.append(tc)
        return max_len_candidates

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
