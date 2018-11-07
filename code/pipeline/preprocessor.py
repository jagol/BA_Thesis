import os
import re
import json
import gzip
from typing import List, Tuple, Dict, Union, BinaryIO

import spacy

from text_processing_unit import TextProcessingUnit
from taxonomic_relation_extraction import HearstHypernymExtractor

# ----------------------------------------------------------------------
# type definitions

# Type of hypernym relations with hypernym as key and a list of hyponyms
# as values.
rels_type = Dict[str, List[str]]
# Type of processed sentence. The sentence is a list of words. Each word
# is a tuple consisting of token, pos-tag, lemma, stop-word.
nlp_sent_type = List[Tuple[str, str, str, bool]]
# Type of an index representation of a sentence (token or lemma) where
# terms are joined indices of individual indices.
idx_repr_type = List[Union[str, int]]

# ----------------------------------------------------------------------


class Preprocessor(TextProcessingUnit):
    """Class to preprocess a corpus for ontology learning.

    Handle tokenization, pos-tagging, lemmatization, indexing and
    implementation of Hearst-Patterns for a corpus.

    For a given corpus the following corpus files are produced:
        - pp_corpus.txt: the 'normal' string represented corpus
            processed as described below.
        - token_idx_corpus.txt: the token index represented corpus
            processed as described below.
        - lemma_idx_corpus.txt: the lemma index represented corpus
            processed as described below.

    In all corpus files the tokens are space separated, there is one
    sentence per line and an additional newline at the end of the
    document. In all corpus files multiword expressions deemed as terms
    are concatenated (by '_') to one word.

    Additional produced files are:
        - hierarchical_relations.txt: one line per relation, items are
            space separated. The first item is a hypernym. All the
            following items are hyponyms.
        - token_to_idx.txt: The file contains one token per line.
            Each line is of the form: Token SPACE Index
        - lemma_to_idx.txt: The file contains one lemma per line.
            Each line is of the form: Lemma SPACE Index

    All files are written to the specified output directory
    ('path_out' in __init__).

    NOTE: Formats with hierarchical structures like 'JSON' are avoided
    so it is not necessary to load an entire file into memory.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 path_lang_model: str,
                 encoding: str,
                 max_files: int = None
                 ) -> None:
        """Initialize Preprocessor.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
            path_lang_model: path to the spacy language model
            encoding: encoding of the text files
            max_files: max number of files to be processed
        """
        self.path_in = path_in
        self.path_out = path_out
        self.encoding = encoding
        self._nlp = spacy.load(path_lang_model)
        self._token_to_idx = {}   # {token: idx}
        self._token_idx = 0       # index counter for tokens
        self._lemma_to_idx = {}   # {lemma: idx}
        self._lemma_idx = 0       # index counter for lemmas
        # Pattern to extract sequences of nouns and adjectives ending
        # with a noun.
        self._term_pattern = re.compile(
            r'(JJ[RS]{0,2}\d+ |NN[PS]{0,2}\d+ )*NN[PS]{0,2}\d+')
        # self._term_pattern = re.compile('(NN[PS]{0,2}\d+ )+')
        self._pp_corpus = []
        self._token_idx_corpus = []
        self._lemma_idx_corpus = []
        super().__init__(path_in, path_out, max_files)

    def preprocess_corpus(self):
        raise NotImplementedError


class DBLPPreprocessor(Preprocessor):
    """Class to preprocess the dblp corpus.

    Only the paper titles are considered as part of the corpus and are
    extracted. The full text is not available in mark up. All other information
    like author, year etc is discarded.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 path_lang_model: str,
                 encoding: str,
                 max_files: int = None
                 ) -> None:

        self._num_titles = 4327497
        self._num_sents = 4189903
        self._files_processed = 1
        self._titles_proc = 0
        self._hyper_hypo_rels = {}  # {hypernym: [hyponym1, hyponym2, ...]}
        self._title_pattern = re.compile(r'<(\w+)>(.*)</\w+>')
        super().__init__(path_in, path_out, path_lang_model, encoding,
                         max_files)

    def preprocess_corpus(self) -> None:
        """Preprocess the dblp corpus.

        Preprocessing includes lemmatization and concatenation of
        term candidates.

        Output: A file containing one sentence per line. After the
        end of each document there is an additional newline.
        """
        # open the three output files
        # path_pp_corpus = os.path.join(self.path_out, 'pp_corpus.txt')
        # self._pp_corpus = open(path_pp_corpus, 'w', encoding='utf8')
        # path_token_idx_corpus = os.path.join(
        #     self.path_out, 'token_idx_corpus.txt')
        # self._token_idx_corpus = open(
        #     path_token_idx_corpus, 'w', encoding='utf8')
        # path_lemma_idx_corpus = os.path.join(
        #     self.path_out, 'lemma_idx_corpus.txt')
        # self._lemma_idx_corpus = open(
        #     path_lemma_idx_corpus, 'w', encoding='utf8')

        path_infile = os.path.join(self.path_in)  # , self._fnames[0])
        with gzip.open(path_infile, 'r') as f:
            self._files_processed += 1
            for title in self._title_getter(f):
                self._process_title(title)

        # self._pp_corpus.close()
        # self._token_idx_corpus.close()
        # self._lemma_idx_corpus.close()

        self._write_pp_corpus_to_file(self._pp_corpus, 'pp_corpus.txt')
        self._write_idx_corpus_to_file(
            self._token_idx_corpus, 'token_idx_corpus.txt')
        self._write_idx_corpus_to_file(
            self._lemma_idx_corpus, 'lemma_idx_corpus.txt')
        with open('token_to_idx.json', 'w', encoding='utf8') as f:
            json.dump(self._token_to_idx, f)
        with open('lemma_to_idx.json', 'w', encoding='utf8') as f:
            json.dump(self._lemma_to_idx, f)

    def _title_getter(self,
                      f: BinaryIO
                      ) -> str:
        """Yield all titles of the dblp corpus file.

        All lines that do not have the tag title or that have the tag
        but just the generic content 'Home Page' are filteret out.

        Args:
            f: input file
        """
        for line in f:
            line = line.decode('utf8')
            match = re.search(self._title_pattern, line)
            if match:
                tag, content = match.groups()
                if tag == 'title' and content != 'Home Page':
                    yield content

    def _process_title(self,
                       title: str
                       ) -> None:
        pp_title = []
        token_idx_title = []
        lemma_idx_title = []

        # process each sentence of title
        nlp_title = self._nlp(title)
        for sent in nlp_title.sents:
            nlp_sent = [(token.text, token.tag_, token.lemma_,
                         token.is_stop)
                        for token in sent]
            pp_sent, token_idx_sent, lemma_idx_sent, rels =\
                self._process_sent(nlp_sent)

            # add extracted relations to relation dict
            self._add_rels(rels)

            # append processed sentences to title
            pp_title.append(pp_sent)
            token_idx_title.append(token_idx_sent)
            lemma_idx_title.append(lemma_idx_sent)

        # add processed title to corpus
        self._pp_corpus.append(pp_title)
        self._token_idx_corpus.append(token_idx_title)
        self._lemma_idx_corpus.append(lemma_idx_title)

        # update the command line
        self._titles_proc += 1
        self._update_cmd()

    def _process_sent(self,
                      nlp_sent: nlp_sent_type
                      ) -> Tuple[str, idx_repr_type, idx_repr_type, rels_type]:
        """Process a sentence.

        For the given sentence, produce a:
            - a 'normal' string represenation
            - token index representation
            - lemma index representation
        where all tokens are separated by space and all terms are
        concatenated to one word. A term is defined by
        'self._term_pattern'. Additionally extract all
        hypernym-hyponym-relations in the sentence using
        Hearst-Patterns.

        Args:
            nlp_sent: see in type definitions
        Return:
            A list of consisting of:
                - the 'normal' string represenation
                - the token index represenation
                - the lemma index represenation
                - a dictionary of extracted hypernym-hyponym-relations
        """
        tokens = [word[0] for word in nlp_sent]
        lemmas = [word[2] for word in nlp_sent]
        term_indices = self._get_term_indices(nlp_sent)

        # get token index representation for sentence
        token_idx_sent = []
        for token in tokens:
            if token not in self._token_to_idx:
                self._token_to_idx[token] = self._token_idx
                self._token_idx += 1
            token_idx_sent.append(self._token_to_idx[token])
        token_idx_sent = self._concat_term_idxs(token_idx_sent, term_indices)
        # get lemma index representation for sentence
        lemma_idx_sent = []
        for lemma in lemmas:
            if lemma not in self._lemma_to_idx:
                self._lemma_to_idx[lemma] = self._lemma_idx
                self._lemma_idx += 1
            lemma_idx_sent.append(self._lemma_to_idx[lemma])
        lemma_idx_sent = self._concat_term_idxs(lemma_idx_sent, term_indices)
        # get relations
        rels = HearstHypernymExtractor.get_rels(nlp_sent)

        # get lemmatized string sentence with concatenations
        pp_sent = self._concat_terms(lemmas, term_indices)

        return pp_sent, token_idx_sent, lemma_idx_sent, rels

    # def _concat_term_idxs(self,
    #                   idx_sent: List[int],
    #                   term_indices: List[List[int]]
    #                   ) -> List[Union[int, str]]:
    #     """Concatenate the each term in a sentence to one word.
    #
    #     Args:
    #         nlp_sent: see in type definitions
    #         idx_sent: an index representation of a sentence
    #     Return:
    #         A list of indices making up the sentence. Multiword
    #         expressions are represented by concatenating the indices
    #         of the words contained in the expression by '_'.
    #     """
    #     return self._concat_indices(idx_sent, term_indices)

    def _get_term_indices(self,
                          nlp_sent: nlp_sent_type
                          ) -> List[List[int]]:
        """Get the indices of multiword terms in a sentence.

        Args:
            nlp_sent: see in type definitions
        Return:
            List of terms. Each term is a list of sentence indices,
            which make up the term.
        """
        pos_words = []
        for i in range(len(nlp_sent)):
            word = nlp_sent[i]
            pos_word = word[1] + str(i)
            pos_words.append(pos_word)
        pos_sent = ' '.join(pos_words)
        matches = re.finditer(self._term_pattern, pos_sent)
        term_indices = [self._get_indices(match.group()) for match in matches]
        return [indices for indices in term_indices if len(indices) > 1]

    @staticmethod
    def _get_indices(match: str):
        """Get the indices in match.

        The indices are concatenated with their corresponding words.
        For example the input 'the0 pig1 is2 is3 pink4' leads to the
        output [0, 1, 2, 3, 4].

        Args:
            match: the input string with words and indices
        """
        pattern = re.compile(r'(\w+?)(\d+)')
        indices = []
        for pos in match.split(' '):
            if pos and pos[-1].isdigit():
                index = int(re.search(pattern, pos).group(2))
                indices.append(index)
        return indices

    @staticmethod
    def _concat_term_idxs(idx_sent: List[int],
                          term_indices: List[List[int]]
                          ) -> List[Union[int, str]]:
        """Concatenate the indices in idx_sent given by term_indices.

        Use an underscore as concatenator. For example: If idx_sent is
        [7, 34, 5] and term_indices is [[1, 2]] the output should be
        [7, 34_5]. Note: While idx_sent contains an actual index
        represenation term_indices only contains list indices that
        indicate which index representations in idx_sent belong together
        as one term.

        Args:
            idx_sent: a sentence representation using indices
            term_indices: A list of lists. Each list is a term and
                contains the indices belonging to that term.
        """
        for ti in term_indices[::-1]:
            str_idxs = [str(idx) for idx in idx_sent[ti[0]:ti[-1] + 1]]
            joined = '_'.join(str_idxs)
            idx_sent[ti[0]:ti[-1] + 1] = [joined]

        return idx_sent

    def _add_rels(self, rels: rels_type) -> None:
        """Add given hierarchical relations to relation dictionary.

        Args:
            rels: a dict containing hypernym-hyponym relations
        """
        for hypernym in rels:
            hyponyms = rels[hypernym]
            if hypernym in self._hyper_hypo_rels:
                self._hyper_hypo_rels[hypernym].extend(hyponyms)
            else:
                self._hyper_hypo_rels[hypernym] = hyponyms

    @staticmethod
    def _concat_terms(lemmas: List[str],
                      term_indices: List[List[int]]
                      ) -> str:
        """Concatenate multiword terms by '_'.

        Args:
            lemmas: A list of lemmatized words
            term_indices: A list of lists. Each list is a term and
                contains the indices belonging to that term.
        Return:
            Sentence as a string with concatenated multiword terms.
        """
        for ti in term_indices[::-1]:
            joined = '_'.join(lemmas[ti[0]:ti[-1] + 1])
            lemmas[ti[0]:ti[-1] + 1] = [joined]
        sent = ' '.join(lemmas)
        return sent

    def _write_pp_corpus_to_file(self,
                                 pp_corpus: List[List[str]],
                                 fname: str
                                 ) -> None:
        path_pp_corpus = os.path.join(self.path_out, fname)
        with open(path_pp_corpus, 'w', encoding='utf8') as f:
            f.write(self._pp_corpus_to_string(pp_corpus))

    def _write_idx_corpus_to_file(self,
                                  idx_corpus: List[List[Union[int, str]]],
                                  fname: str
                                  ) -> None:
        path_idx_corpus = os.path.join(
            self.path_out, fname)
        with open(path_idx_corpus, 'w', encoding='utf8') as f:
            f.write(self._idx_corpus_to_string(idx_corpus))

    @staticmethod
    def _pp_corpus_to_string(pp_corpus: List[List[str]]):
        corpus_as_str = ''
        for doc in pp_corpus:
            for sent in doc:
                line = sent + '\n'
                corpus_as_str += line
            corpus_as_str += '\n'
        return corpus_as_str

    @staticmethod
    def _idx_corpus_to_string(idx_corpus):
        corpus_as_str = ''
        for doc in idx_corpus:
            for sent in doc:
                sent = [str(i) for i in sent]
                line = ' '.join(sent) + '\n'
                corpus_as_str += line
            corpus_as_str += '\n'
        return corpus_as_str

    def _update_cmd(self) -> None:
        """Update the information on the command line."""
        if self._titles_proc == self._num_titles:
            msg = 'Processing: title {} of {}'
            print(msg.format(self._titles_proc, self._num_titles))
        else:
            msg = 'Processing: title {} of {}\r'
            print(msg.format(self._titles_proc, self._num_titles),
                  end='\r')


class SPPreprocessor(Preprocessor):
    """Class to preprocess the dblp corpus.

    Handle tokenization, pos-tagging and lemmatization of the sp corpus.
    The corpus is a collection of paperabstract and stored in one file
    with a newline as separator.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 path_lang_model: str,
                 encoding: str,
                 max_files: int = None
                 ) -> None:

        self._num_summaries = 94476
        self._summaries = {}
        # Dict[int, Dict[int, List[List[Union[str, bool]]]]]
        self._sum_processed = 0
        super().__init__(path_in, path_out, path_lang_model, encoding,
                         max_files)

    def preprocess_corpus(self) -> None:
        """Preprocess the SP-corpus. Write output to json file."""
        fpath = os.path.join(self.path_in, self._fnames[0])
        with open(fpath, 'r', encoding=self.encoding) as f:
            for line in f:
                nlp_summary = self._nlp(line.strip('\n'))
                self._summaries[self._sum_processed] = {}
                self._add_sents(nlp_summary)
                self._sum_processed += 1
                self._update_cmd()
                if self._sum_processed % 10000 == 0:
                    self._write_json(self._summaries)
                    self._summaries = {}

        self._write_json(self._summaries)

    def _add_sents(self, nlp_summary: spacy.tokens.doc.Doc) -> None:
        for i, sent in enumerate(nlp_summary.sents):
            nlp_sent = [(token.text, token.tag_, token.lemma_, token.is_stop)
                        for token in sent]
            self._summaries[self._sum_processed][i] = nlp_sent

    def _write_json(self, summaries: Dict[int, Dict[int, str]]) -> None:
        fname = '{}.json'.format(self._sum_processed)
        f_out = os.path.join(self.path_out, fname)
        with open(f_out, 'w', encoding='utf8') as f:
            json.dump(summaries, f)

    def _update_cmd(self) -> None:
        """Update the information on the command line."""
        if self._sum_processed == self._num_summaries:
            msg = 'Processing: summary {} of {}'
            print(msg.format(self._sum_processed, self._num_summaries))
        else:
            msg = 'Processing: summary {} of {}\r'
            print(msg.format(self._sum_processed, self._num_summaries),
                  end='\r')


if __name__ == '__main__':
    from utility_functions import get_corpus_config
    corpus, config = get_corpus_config('preprocessing')
    if corpus == 'dblp':
        dp = DBLPPreprocessor(**config)
        dp.preprocess_corpus()
    elif corpus == 'sp':
        sp = SPPreprocessor(**config)
        sp.preprocess_corpus()
