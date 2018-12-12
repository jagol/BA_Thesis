import os
import re
import gzip
import time
from typing import List, Tuple, BinaryIO
import spacy
from text_processing_unit import TextProcessingUnit

# ----------------------------------------------------------------------
# type definitions

# Type of processed sentence. The sentence is a list of words. Each word
# is a tuple consisting of token, pos-tag, lemma, stop-word.
nlp_sent_type = List[Tuple[str, str, str, bool]]

# ----------------------------------------------------------------------


class LingPreprocessor(TextProcessingUnit):
    """Class to preprocess a corpus for ontology learning.

    Handle tokenization, pos-tagging, lemmatization and recognizing
    stop words.

    For a given corpus a file named 'ling_pp_<corpus_name>.txt' of the
    following format is produced:
    There is one sentence per line.
    Documents are separated by empty lines.
    A line is of the form:
    token0 SPACE Pos-Tag0 SPACE lemma0 SPACE isStop0 TAB token1 SPACE...

    The file is written to '<path_out>/processed_corpus/'
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 path_lang_model: str,
                 max_docs: int = None
                 ) -> None:
        """Initialize Preprocessor.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
            path_lang_model: path to the spacy language model
            max_docs: max number of documents to be processed
        """
        super().__init__(path_in, path_out, max_docs)
        self.path_in = path_in
        self.path_out_dir = os.path.join(path_out, 'processed_corpus/')
        self._nlp = spacy.load(path_lang_model)
        self._start_time = 0
        self._file_write_threshhold = 10000
        self._pp_corpus = []
        self.f_out_name = 'ling_pp_corpus.txt'
        self.f_out = os.path.join(self.path_out_dir, self.f_out_name)

    def _doc_getter(self,
                    f: BinaryIO
                    ) -> str:
        raise NotImplementedError

    def preprocess_corpus(self):
        """Tokenize, pos-tag, lemmatize, and get stop-words for corpus.

        The output file named <ling_pp_corpus.txt> has the format:
        One sentence per line. Each line has the form:
        token1 SPACE tag1 SPAC lemma1 SPACE isStop1 TAB token2 ...
        Documents are separated by empty lines.
        """
        self._start_time = time.time()
        path_infile = os.path.join(self.path_in)
        with gzip.open(path_infile, 'r') as f:
            for doc in self._doc_getter(f):
                self._process_doc(doc)

                if self._docs_processed >= self._upper_bound:
                    break

                if self._docs_processed % self._file_write_threshhold == 0:
                    # write contents to file periodically to avoid too
                    # much memory usage
                    # print infos when writing to file
                    self._update_cmd_time_info()
                    self._write_pp_corpus_to_file()

                    # clear corpus after writing
                    self._pp_corpus = []

        self._update_cmd_time_info(end=True)
        self._write_pp_corpus_to_file()

    def _process_doc(self,
                     doc: str
                     ) -> None:
        """Tokenize, pos-tag, lemmatize, and get stop-words for corpus."""
        pp_doc = []

        # process each sentence of doc
        nlp_doc = self._nlp(doc)
        for i, sent in enumerate(nlp_doc.sents):
            nlp_sent = [(token.text, token.tag_, token.lemma_,
                         token.is_stop)
                        for token in sent]

            # append processed sentences to doc
            pp_doc.append(nlp_sent)

        # add processed doc to corpus
        self._pp_corpus.append(pp_doc)

        # update the command line
        self._docs_processed += 1
        self._update_cmd_counter()

    def _write_pp_corpus_to_file(self) -> None:
        mode = self._get_write_mode()
        path_pp_corpus = os.path.join(self.path_out_dir, self.f_out_name)
        with open(path_pp_corpus, mode, encoding='utf8') as f:
            f.write(self._pp_corpus_to_string())

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_processed == self._file_write_threshhold:
            return 'w'
        return 'a'

    def _pp_corpus_to_string(self) -> str:
        """Return all documents in self._pp_corpus as one string."""
        corpus_as_str = ''
        for doc in self._pp_corpus:
            for sent in doc:
                list_of_words = []
                for word in sent:
                    word_str = ' '.join([str(w) for w in word])
                    list_of_words.append(word_str)
                sent = '\t'.join(list_of_words)
                line = sent + '\n'
                corpus_as_str += line
            corpus_as_str += '\n'
        return corpus_as_str

    def _update_cmd_counter(self) -> None:
        """Update the information on the command line."""
        if self._docs_processed == self._upper_bound:
            msg = 'Processing: document {} of {}'
            print(msg.format(self._docs_processed, self._upper_bound))
        else:
            msg = 'Processing: document {} of {}\r'
            print(msg.format(self._docs_processed, self._upper_bound),
                  end='\r')

    def _update_cmd_time_info(self, end=False):
        """Update time information on the command line.

        Args:
            end: set to True if last writing procedure
        """
        time_stamp = time.time()
        time_passed = time_stamp - self._start_time
        if end:
            docs_proc_now = self._docs_processed % self._file_write_threshhold
            if docs_proc_now == 0:
                msg = ('Written {} documents to file in total. '
                       'Time passed: {:2f}')
                print(msg.format(self._docs_processed, time_passed))
            else:
                msg = ('Writing {} documents to file. '
                       'Written {} documents to file in total. '
                       'Time passed: {:2f}')
                print(msg.format(
                    docs_proc_now, self._docs_processed, time_passed))
        else:
            msg = ('Writing {} documents to file. '
                   'Written {} documents to file in total. '
                   'Time passed: {:2f}')
            print(msg.format(self._file_write_threshhold,
                             self._docs_processed, time_passed))


class DBLPLingPreprocessor(LingPreprocessor):
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
                 max_docs: int = None
                 ) -> None:
        self._num_docs = 4327497
        self._num_sents = 4189903
        self._docs_processed = 0
        self._title_pattern = re.compile(r'<(\w+)>(.*)</\w+>')
        super().__init__(path_in, path_out, path_lang_model, max_docs)

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


class SPLingPreprocessor(LingPreprocessor):
    """Class to preprocess the dblp corpus.

    Handle tokenization, pos-tagging and lemmatization of the sp corpus.
    The corpus is a collection of paperabstract and stored in one file
    with a newline as separator.
    """

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 path_lang_model: str,
                 max_docs: int = None
                 ) -> None:
        self._num_docs = 94476
        self._num_sents = 0
        self._docs_processed = 0
        super().__init__(path_in, path_out, path_lang_model, max_docs)

    def _doc_getter(self,
                    f: BinaryIO
                    ) -> str:
        pass


def main():
    from utility_functions import get_config, get_cmd_args, prep_output_dir
    config = get_config()
    args = get_cmd_args()
    path_in = config['paths'][args.location][args.corpus]['path_in']
    path_out = config['paths'][args.location][args.corpus]['path_out']
    path_lang_model = config['paths'][args.location]['path_lang_model']
    max_docs = 1000
    prep_output_dir(path_out)
    if args.corpus == 'dblp':
        dp = DBLPLingPreprocessor(path_in, path_out, path_lang_model, max_docs)
        dp.preprocess_corpus()
    elif args.corpus == 'sp':
        sp = SPLingPreprocessor(path_in, path_out, path_lang_model, max_docs)
        sp.preprocess_corpus()


if __name__ == '__main__':
    main()
