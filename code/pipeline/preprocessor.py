import os
import json
import subprocess
from typing import *
import spacy
from text_processing_unit import TextProcessingUnit


class Preprocessor(TextProcessingUnit):
    """Class to preprocess a corpus.

    Handle tokenization, pos-tagging and lemmatization of a corpus.
    The corpus is assumed to be a folder containing documents (textfiles)
    with one sentence per line.
    """

    def __init__(self, path_in: str, path_out: str, path_lang_model: str,
                 encoding: str, max_files: int = None):
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

    def process_corpus(self) -> None:
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


if __name__ == '__main__':
    from utility_functions import get_config
    config = get_config('preprocessing')
    pp = Preprocessor(**config)
    pp.process_corpus()
