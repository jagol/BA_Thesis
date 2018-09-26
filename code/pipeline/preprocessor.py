import os
import json
import subprocess
from typing import *
import spacy


class Preprocessor:
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
        self._max_files = max_files
        self._nlp = spacy.load(path_lang_model)
        self._files_processed = 0
        self._sents_processed = 0
        self._fnames = [fname for fname in os.listdir(self.path_in)
                    if os.path.isfile(os.path.join(self.path_in, fname))]
        self._fnames.sort()
        self._num_files = len(self._fnames)
        self._upper_bound = 0

        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)

    def _write_json(self, annotated_sents: List[List[Tuple[str, str, str, bool]]]):
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
        if self._max_files:
            self._upper_bound = min(len(self._fnames), self._max_files)
        else:
            self._upper_bound = len(self._fnames)

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

    def _update_cmd(self) -> None:
        """Update the information on the command line."""
        final_msg = False
        if self._files_processed == self._upper_bound:
            if self._sents_processed == self._num_sents:
                msg = 'Processing: sentence {}, file {} of {}'
                print(msg.format(self._sents_processed, self._files_processed,
                                 self._num_files))
                final_msg = True
        if not final_msg:
            msg = 'Processing: sentence {}, file {} of {}\r'
            print(msg.format(self._sents_processed, self._files_processed,
                            self._num_files) , end='\r')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--server',
        help="indicate if local paths or server paths should be used",
        action='store_true')
    args = parser.parse_args()
    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        if args.server:
            configs_server_pp = configs['server']['preprocessing']
            path_in = configs_server_pp['path_in']
            path_out = configs_server_pp['path_out']
            spacy_model = configs_server_pp['spacy_model']
            encoding = configs_server_pp['encoding']
        else:
            configs_local_pp = configs['local']['preprocessing']
            path_in = configs_local_pp['path_in']
            path_out = configs_local_pp['path_out']
            spacy_model = configs_local_pp['spacy_model']
            encoding = configs_local_pp['encoding']

    pp = Preprocessor(path_in, path_in, spacy_model, encoding)
    pp.process_corpus()
