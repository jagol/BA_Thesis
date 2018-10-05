import os
import re
import json
import gzip
import subprocess
from typing import *

import spacy

from text_processing_unit import TextProcessingUnit


class Preprocessor(TextProcessingUnit):
    """Class to preprocess a corpus.

    Handle tokenization, pos-tagging and lemmatization of a corpus.
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
        super().__init__(path_in, path_out, max_files)

    def preprocess_corpus(self):
        raise NotImplementedError
        

class EUPRLPreprocessor(Preprocessor):
    """Class to preprocess the Europarl corpus.

    Handle tokenization, pos-tagging and lemmatization of the english
    part of the europarl corpus.
    """

    def _write_json(self,
                    annotated_sents: List[List[Tuple[str, str, str, bool]]]
                    ) -> None:
        """Write annotated_sents to a file in json.

        Json is formatted as follows:
        {
            1: [
                [token1, tag1, lemma1, is_stop1], ...
            ],
            2: [
                [token1, tag1, lemma1, is_stop1], ...
            ]
        }

        Args:
            annotated_sents: list of sentences, each sentence is a list
            of tokens, each token is a tuple if the form
            (token, tag, lemma, is_stop_word)
        """
        fpath = os.path.join(self.path_out, str(self._files_processed)+'.json')
        with open(fpath, 'w', encoding=self.encoding) as f:
            json.dump(dict(enumerate(annotated_sents)), f, ensure_ascii=False)

    def preprocess_corpus(self) -> None:
        """Tokenize, pos-tag and lemmatize corpus. Mark stop words.

        Write annotated text files as json into path_out.
        """
        print(10*'-'+' preprocessing corpus '+10*'-')
        print('Input taken from: {}'.format(self.path_in))
        for i in range(self._upper_bound):
            self._files_processed += 1
            self._process_file(self._fnames[i])

        print('Output written to: {}'.format(self.path_out))
        print(42*'-')

    def _process_file(self, fname: str) -> None:
        """Tokenize, pos-tag, lemmatize and mark stop words for a file.

        Write annotated text file as json into path_out.
        """
        self._sents_processed = 0
        fpath = os.path.join(self.path_in, fname)
        self._get_file_length(fpath)
        annotated_sents = []
        with open(fpath, 'r', encoding=self.encoding) as f:
            for sent in f:
                sent = sent.strip('\n')
                nlp_sent = self._nlp(sent)
                annotated_sent = [(token.text, token.tag_, token.lemma_,
                                   token.is_stop) for token in nlp_sent]
                annotated_sents.append(annotated_sent)
                self._sents_processed += 1
                self._update_cmd()

        self._write_json(annotated_sents)

    def _get_file_length(self, fpath: str) -> None:
        cmd = ['wc', fpath]
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf8')
        self._num_sents = int(out.strip(' ').split(' ')[0])


class DBLPPreprocessor(Preprocessor):
    """Class to preprocess the dblp corpus.

    Handle tokenization, pos-tagging and lemmatization of the dblp corpus.
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

        super().__init__(path_in, path_out, path_lang_model, encoding,
                        max_files)

    def preprocess_corpus(self):
        annotated_titles = {}
        pattern = re.compile(r'<(\w+)>(.*)</\w+>')
        fpath = os.path.join(self.path_in, self._fnames[0])
        with gzip.open(fpath, 'r') as f:
            self._num_sents = 4189903
            j = 0
            self._files_processed = 1
            for line in f:
                line = line.decode('utf8')
                match = re.search(pattern, line)
                if match:
                    tag, content = match.groups()
                    if tag == 'title' and content != 'Home Page':
                        nlp_title = self._nlp(content)
                        ant_title = [
                            (token.text, token.tag_, token.lemma_,
                             token.is_stop) for token in nlp_title]
                        annotated_titles[self._sents_processed] = ant_title
                        self._sents_processed += 1
                        self._update_cmd()
                        if self._sents_processed % 10000 == 0:
                            self._write_json(annotated_titles, j)
                            j += 1
                            annotated_titles = {}
            else:
                self._write_json(annotated_titles, j)
                j += 1

    def _write_json(self,
                    annotated_titles: Dict[int, Tuple[Union[str, bool]]],
                    j: int
                    ) -> None:
        f_out = os.path.join(self.path_out, str(j) + '.json')
        with open(f_out, 'w', encoding='utf8') as f:
            json.dump(annotated_titles, f)


if __name__ == '__main__':
    from utility_functions import get_corpus_config
    corpus, config = get_corpus_config('preprocessing')
    if corpus == 'dblp':
        dp = DBLPPreprocessor(**config)
        dp.preprocess_corpus()
    elif corpus == 'europarl':
        ep = EUPRLPreprocessor(**config)
        ep.preprocess_corpus()
