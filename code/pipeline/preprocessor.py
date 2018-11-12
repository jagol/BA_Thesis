import os
import re
import json
import gzip
import time
from typing import List, Tuple, Dict, Union, BinaryIO
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

import utility_functions
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
            (now as json)
        - lemma_to_idx.txt: The file contains one lemma per line.
            Each line is of the form: Lemma SPACE Index
            (now as json)
        - lemma_counts.json: Map each lemma to it's number of occurences
            in the corpus.

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
                 max_docs: int = None
                 ) -> None:
        """Initialize Preprocessor.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
            path_lang_model: path to the spacy language model
            encoding: encoding of the text files
            max_docs: max number of documents to be processed
        """
        self.path_in = path_in
        self.path_out = path_out
        self.encoding = encoding
        self._nlp = spacy.load(path_lang_model)
        self._token_to_idx = {}   # {token: idx}
        self._token_idx = 0       # index counter for tokens
        self._lemma_to_idx = {}   # {lemma: idx}
        self._lemma_idx = 0       # index counter for lemmas
        self._lemma_count = {}    # {lemma_idx: num of occurences}
        # Pattern to extract sequences of nouns and adjectives ending
        # with a noun.
        self._term_pattern = re.compile(
            r'(JJ[RS]{0,2}\d+ |NN[PS]{0,2}\d+ |IN\d+ )*NN[PS]{0,2}\d+')
        # self._term_pattern = re.compile('(NN[PS]{0,2}\d+ )+')
        self.term_index = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        # {lemma_idx: {doc_id: {sent_id: word_id}}
        # in term_index: list all extracted terms and where
        # in the corpus they occur
        self._pp_corpus = []
        self._token_idx_corpus = []
        self._lemma_idx_corpus = []
        self._start_time = 0
        super().__init__(path_in, path_out, max_docs)

    def preprocess_corpus(self):
        raise NotImplementedError

    def calc_tfidf(self, corpus_path: str) -> None:
        """Calculate all the TFIDF values for a given corpus file.

        Documents are separated by an empty line. There is one sentence
        per line. Write the scores to the file 'tfidf_scores.txt'. The
        File has the following format:
        lemma_id score-doc-1 score-doc-2 ...

        Args:
            corpus_path: path to the corpus file
        """
        docs = utility_functions.get_docs(corpus_path, sent_tokenized=False)
        tfidf = TfidfVectorizer(analyzer='word',
                                tokenizer=identity_function,
                                preprocessor=identity_function,
                                token_pattern=None)
        print('fit-transform tfidf')
        tfidf_matrix = tfidf.fit_transform(docs).toarray()
        lemma_ids = tfidf.get_feature_names()
        path_out = os.path.join(self.path_out, 'tfidf.txt')
        print('write tfidf to file...')
        with open(path_out, 'w', encoding='utf8') as f:
            for i, row in enumerate(zip(*tfidf_matrix)):
                lemma_id = lemma_ids[i]
                values = ' '.join(['{:.2f}'.format(x) for x in row])
                line = '{} {}'.format(lemma_id, values)
                f.write(line)


class DBLPPreprocessor(Preprocessor):
    """Class to preprocess the dblp corpus.

    Only the paper titles are considered as part of the corpus and are
    extracted. The full text is not available in mark up. All other information
    like author, year etc is discarded. The titles are referred to as
    documents since they serve here the function of a document.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 path_lang_model: str,
                 encoding: str,
                 max_docs: int = None
                 ) -> None:

        self._num_docs = 4327497
        self._num_sents = 4189903
        self._files_processed = 1
        self._docs_proc = 0
        self._file_write_threshhold = 50000
        self._hyper_hypo_rels = {}  # {hypernym: [hyponym1, hyponym2, ...]}
        self._title_pattern = re.compile(r'<(\w+)>(.*)</\w+>')
        super().__init__(path_in, path_out, path_lang_model, encoding,
                         max_docs)

    def _get_upper_bound(self) -> int:
        if self._max_docs:
            return min(self._max_docs, self._num_docs)
        return self._num_docs

    def preprocess_corpus(self) -> None:
        """Preprocess the dblp corpus.

        Preprocessing includes lemmatization and concatenation of
        term candidates.

        Output: A file containing one sentence per line. After the
        end of each document there is an additional newline.
        """
        self._start_time = time.time()
        path_infile = os.path.join(self.path_in)  # , self._fnames[0])
        with gzip.open(path_infile, 'r') as f:
            self._files_processed += 1
            for doc in self._doc_getter(f):
                self._process_doc(doc)

                if self._docs_proc >= self._upper_bound:
                    break

                if self._docs_proc % self._file_write_threshhold == 0:
                    # write contents to file periodically to avoid too
                    # much memory usage
                    # print infos when writing to file
                    self._update_cmd_time_info()
                    self._write_pp_corpus_to_file(
                        self._pp_corpus, 'pp_corpus.txt')
                    self._write_idx_corpus_to_file(
                        self._token_idx_corpus, 'token_idx_corpus.txt')
                    self._write_idx_corpus_to_file(
                        self._lemma_idx_corpus, 'lemma_idx_corpus.txt')

                    # clear corpus after writing
                    self._pp_corpus = []
                    self._token_idx_corpus = []
                    self._lemma_idx_corpus = []

        self._update_cmd_time_info(end=True)
        self._write_pp_corpus_to_file(self._pp_corpus, 'pp_corpus.txt')
        self._write_idx_corpus_to_file(
            self._token_idx_corpus, 'token_idx_corpus.txt')
        self._write_idx_corpus_to_file(
            self._lemma_idx_corpus, 'lemma_idx_corpus.txt')

        fpath_token_to_idx = os.path.join(self.path_out, 'token_to_idx.json')
        with open(fpath_token_to_idx, 'w', encoding='utf8') as f:
            json.dump(self._token_to_idx, f)
        del self._token_to_idx

        fpath_lemma_to_idx = os.path.join(self.path_out, 'lemma_to_idx.json')
        with open(fpath_lemma_to_idx, 'w', encoding='utf8') as f:
            json.dump(self._lemma_to_idx, f)

        fpath_lemma_counts = os.path.join(self.path_out, 'lemma_counts.json')
        with open(fpath_lemma_counts, 'w', encoding='utf8') as f:
            json.dump(self._lemma_count, f)
        del self._lemma_count

        fpath_term_index = os.path.join(self.path_out, 'term_index.json')
        with open(fpath_term_index, 'w', encoding='utf8') as f:
            json.dump(self.term_index, f)
        del self.term_index

        self._write_hierarchical_rels_to_file('hierarchical_relations.txt')
        del self._lemma_to_idx

        print('Calculate tfidf...')
        fpath = os.path.join(self.path_out, 'lemma_idx_corpus.txt')
        self.calc_tfidf(fpath)
        print('Preprocessing done.')

    def _doc_getter(self,
                      f: BinaryIO
                      ) -> str:
        """Yield all titles (docs) of the dblp corpus file.

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

    def _process_doc(self,
                       doc: str
                       ) -> None:
        pp_doc = []
        token_idx_doc = []
        lemma_idx_doc = []

        # process each sentence of doc
        nlp_doc = self._nlp(doc)
        for i, sent in enumerate(nlp_doc.sents):
            nlp_sent = [(token.text, token.tag_, token.lemma_,
                         token.is_stop)
                        for token in sent]
            pp_sent, token_idx_sent, lemma_idx_sent, rels, term_indices =\
                self._process_sent(nlp_sent)

            # add extracted relations to relation dict
            self._add_rels(rels)

            # append processed sentences to doc
            pp_doc.append(pp_sent)
            token_idx_doc.append(token_idx_sent)
            lemma_idx_doc.append(lemma_idx_sent)

            # add term occurences to term index
            for word_index in term_indices:
                tidx = lemma_idx_sent[word_index]
                self.term_index[tidx][self._docs_proc][i].append(word_index)

        # add processed doc to corpus
        self._pp_corpus.append(pp_doc)
        self._token_idx_corpus.append(token_idx_doc)
        self._lemma_idx_corpus.append(lemma_idx_doc)

        # update the command line
        self._docs_proc += 1
        self._update_cmd_counter()

    def _process_sent(self,
                      nlp_sent: nlp_sent_type
                      ) -> Tuple[str, idx_repr_type, idx_repr_type,
                                 rels_type, List[int]]:
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

        # get indices of all term positions in sentence
        term_idxs = []
        for i in range(len(lemma_idx_sent)):
            if isinstance(lemma_idx_sent[i], str):
                term_idxs.append(i)
        # update lemma counts
        for lemma_idx in lemma_idx_sent:
            self._add_count(lemma_idx)

        # get relations
        rels = HearstHypernymExtractor.get_rels(nlp_sent)
        rels = self._concat_rels(rels)

        # get lemmatized string sentence with concatenations
        pp_sent = self._concat_terms(lemmas, term_indices)

        return pp_sent, token_idx_sent, lemma_idx_sent, rels, term_idxs

    @staticmethod
    def _concat_rels(rels):
        rels_out = {}
        for hyper in rels:
            hyper_out = re.sub(' ', '_', hyper)
            hypos_out = [re.sub(' ', '_', hypo) for hypo in rels[hyper]]
            rels_out[hyper_out] = hypos_out
        return rels_out

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
        return term_indices

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
            # joined += '_'
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

    def _add_count(self, lemma_idx: str) -> None:
        if lemma_idx in self._lemma_count:
            self._lemma_count[lemma_idx] += 1
        else:
            self._lemma_count[lemma_idx] = 1

    def _write_pp_corpus_to_file(self,
                                 pp_corpus: List[List[str]],
                                 fname: str
                                 ) -> None:
        mode = self._get_write_mode()
        path_pp_corpus = os.path.join(self.path_out, fname)
        with open(path_pp_corpus, mode, encoding='utf8') as f:
            f.write(self._pp_corpus_to_string(pp_corpus))

    def _write_idx_corpus_to_file(self,
                                  idx_corpus: List[List[Union[int, str]]],
                                  fname: str
                                  ) -> None:
        mode = self._get_write_mode()
        path_idx_corpus = os.path.join(self.path_out, fname)
        with open(path_idx_corpus, mode, encoding='utf8') as f:
            f.write(self._idx_corpus_to_string(idx_corpus))

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_proc == self._file_write_threshhold:
            return 'w'
        return 'a'

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

    def _write_hierarchical_rels_to_file(self, fname: str):
        """Write the file hierarchical_relations.txt.

        Write all patterns extracted through hearst relations to file
        and include their ids. File format:
        Hypernym SPACE id TAB Hyponym1 SPACE id1 TAB ...
        """
        fpath = os.path.join(self.path_out, fname)
        with open(fpath, 'w', encoding='utf8') as f:
            for hyper in self._hyper_hypo_rels:
                self._lookup(hyper, self._lemma_to_idx)
                hyper_id = self._lookup(hyper, self._lemma_to_idx)
                hypos = self._hyper_hypo_rels[hyper]
                hypo_ids = [self._lookup(hypo, self._lemma_to_idx)
                            for hypo in hypos]
                hypos_w_ids = zip(hypos, hypo_ids)
                hypo_strs = ['{} {}'.format(h, id_) for h, id_ in hypos_w_ids]
                hypo_str = '\t'.join(hypo_strs)
                line = '{} {}\t{}\n'.format(hyper, hyper_id, hypo_str)
                f.write(line)

    @staticmethod
    def _lookup(hyper: str, idx_dict: Dict[str, int]) -> str:
        """Look up a key in a dictionary.

        If the key contains one or more '_' in the string. The string
        is split by this separator. Each split is looked up individually
        in the given dict. The results then are joined again by the
        separator.
        """
        try:
            idxs = [str(idx_dict[h]) for h in hyper.split('_')]
            return '_'.join(idxs)
        except KeyError:
            idxs = []
            keys_not_exist = []
            for h in hyper.split(' '):
                try:
                    idxs.append(str(idx_dict[h]))
                except KeyError:
                    idxs.append('unk')
                    keys_not_exist.append(h)
            msg = 'ERROR! Indices for {} in {} do not exist.'
            print(msg.format(str(keys_not_exist), hyper))

    def _update_cmd_counter(self) -> None:
        """Update the information on the command line."""
        if self._docs_proc == self._upper_bound:
            msg = 'Processing: document {} of {}'
            print(msg.format(self._docs_proc, self._upper_bound))
        else:
            msg = 'Processing: document {} of {}\r'
            print(msg.format(self._docs_proc, self._upper_bound), end='\r')

    def _update_cmd_time_info(self, end=False):
        """Update time information on the command line.

        Args:
            end: set to True if last writing procedure
        """
        time_stamp = time.time()
        time_passed = time_stamp - self._start_time
        msg = ('Writing {} documents to file. '
               'Written {} documents to file in total. '
               'Time passed: {:2f}')
        if end:
            print(msg.format(self._docs_proc % self._file_write_threshhold,
                             self._docs_proc, time_passed))
        else:
            print(msg.format(self._file_write_threshhold,
                             self._docs_proc, time_passed))


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
                 max_docs: int = None
                 ) -> None:

        self._num_summaries = 94476
        self._summaries = {}
        # Dict[int, Dict[int, List[List[Union[str, bool]]]]]
        self._sum_processed = 0
        super().__init__(path_in, path_out, path_lang_model, encoding,
                         max_docs)

    def preprocess_corpus(self) -> None:
        """Preprocess the SP-corpus. Write output to json file."""
        fpath = os.path.join(self.path_in, self._fnames[0])
        with open(fpath, 'r', encoding=self.encoding) as f:
            for line in f:
                nlp_summary = self._nlp(line.strip('\n'))
                self._summaries[self._sum_processed] = {}
                self._add_sents(nlp_summary)
                self._sum_processed += 1
                self._update_cmd_counter()
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

    def _update_cmd_counter(self) -> None:
        """Update the information on the command line."""
        if self._sum_processed == self._num_summaries:
            msg = 'Processing: summary {} of {}'
            print(msg.format(self._sum_processed, self._num_summaries))
        else:
            msg = 'Processing: summary {} of {}\r'
            print(msg.format(self._sum_processed, self._num_summaries),
                  end='\r')


def identity_function(x):
    return x


def build_lemma_occurence_index():
    pass


if __name__ == '__main__':
    from utility_functions import get_corpus_config
    corpus, config = get_corpus_config('preprocessing')
    config['max_docs'] = 205
    if corpus == 'dblp':
        dp = DBLPPreprocessor(**config)
        dp.preprocess_corpus()
    elif corpus == 'sp':
        sp = SPPreprocessor(**config)
        sp.preprocess_corpus()
