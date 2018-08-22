import os
import json
import subprocess
import spacy

PATH_IN = './raw_corpus/europarl-v7.de-en.en'
# PATH_IN = './raw_corpus/'
PATH_OUT = './preprocessed_corpus/'
ENCODING = 'utf8'


class Preprocessor:
    """Class to preprocess a corpus.

    Handle tokenization, pos-tagging and lemmatization of a corpus.
    The corpus is assumed to contain textfile(s) with one sentence
    per line.
    """

    def __init__(self, path_in, path_out, encoding, spf=2000, stop=None):
        """Initialize Preprocessor.

        :param path_in: string, path to corpus, can be file or directory
        :param path_out: string, path to output directory
        :param encoding: string, encoding of the text files
        :param spf: int, number of Sentences Per File written to output.
                Example: For an inputfile with 5000 sentences and
                sents_out set to 2000, three output files
                (spf 2000, 2000 and 1000 lines) will be created.
        :param stop: int or None, set upper boundary for number of
                sentences to be processed
        """
        self.path_in = path_in
        self.path_out = path_out
        self.encoding = encoding
        self._dir = os.path.isdir(self.path_in)
        self._spf = spf
        self._stop = stop
        self._nlp = spacy.load('en_core_web_sm')
        self._files_processed = 0
        self._sents_processed = 0
        self._num_sents = 0

        if os.path.isdir(self.path_in):
            self._fnames = [fname for fname in os.listdir(self.path_in)
                        if os.path.isfile(os.path.join(self.path_in, fname))]
        else:
            self._fnames = [self.path_in.split('/')[-1]]
            self.path_in = '/'.join(self.path_in.split('/')[:-1])

        self._fnames.sort()
        self._num_files = len(self._fnames)

        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)

    def _write_json(self, annotated_sents):
        """Write annotated_sents to a file in json.

        Json is formatted as follows:
        {
            1: [
                [token1, tag1, lemma1], ...
            ],
            2: [
                [token1, tag1, lemma1], ...
            ]
        }

        :param annotated_sents: list[list[tuple(string, string, string)]]
                list of sentences, each sentence is a list of tokens,
                each token is a tuple if the form (token, tag, lemma)
        """
        fname = str(self._sents_processed) + '.json'
        with open(self.path_out + fname, 'w', encoding=self.encoding) as f:
            json.dump(dict(enumerate(annotated_sents)), f, ensure_ascii=False)

    def process_corpus(self):
        """Tokenize, pos-tag and lemmatize corpus.

        Write annotated text files as json into path_out.
        """
        if self._dir:
            for i in range(self._num_files):
                self._files_processed += 1
                fname = self._fnames[i]
                self._process_file(os.path.join(self.path_in, fname))
        else:
            self._files_processed += 1
            self._process_file(os.path.join(self.path_in, self._fnames[0]))

    def _get_upper_bound(self):
        """Get the number of sentences that will be processed from file.

        If there is a stop smaller than the number of sentences in the
        file, the stop becomes the upper bound. Else the number of
        sentences in the file become the upper bound.
        """
        if self._stop:
            return min(self._stop, self._num_sents)
        else:
            return self._num_sents

    def _process_file(self, fpath):
        """Tokenize, pos-tag and lemmatize sentences of a file.

        Write annotated text file as json into path_out.

        :param fpath: string, path to file
        """
        annotated_sents = []
        cmd = ['wc', fpath]
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE).communicate()[0].decode('utf8')

        self._num_sents = int(out.strip(' ').split(' ')[0])
        upper_bound = self._get_upper_bound()
        with open(fpath, 'r', encoding=self.encoding) as f:
            for sent in f:
                self._sents_processed += 1
                nlp_sent = self._nlp(sent)
                annotated_sents.append([(token.text, token.tag_, token.lemma_)
                                        for token in nlp_sent])

                if self._sents_processed % self._spf == 0:
                    self._write_json(annotated_sents)
                    annotated_sents = []

                if self._sents_processed == upper_bound:
                    msg = 'Processing: sentence {} of {}\tfile {} of {}'
                    print(msg.format(self._sents_processed, upper_bound,
                                     self._files_processed, self._num_files))
                    break

                else:
                    msg = 'Processing: sentence {} of {}\tfile {} of {}\r'
                    print(msg.format(self._sents_processed, upper_bound,
                                     self._files_processed, self._num_files),
                          end='\r')

        self._sents_processed = 0


if __name__ == '__main__':
    pp = Preprocessor(PATH_IN, PATH_OUT, ENCODING, spf=100, stop=300)
    pp.process_corpus()
