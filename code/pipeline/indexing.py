import os
import json
import time
from typing import *
from utility_functions import get_docs


class Indexer:

    """Class to index tokens and lemmas.

    If the main index_tokens and index_lemmas are executed the following
    files are produced:
    - 'token_idx_corpus': Same structure as 'pp_token_corpus.txt' but
        all tokens are replaced with their index.
    - 'lemma_idx_corpus': Same structure as 'pp_lemma_corpus.txt' but
        all tokens are replaced with their index.
    - 'token_to_idx.json': Maps each token to it's index.
    - 'idx_to_token.json': Maps each index to it's tokens.
    - 'lemma_to_idx.json': Maps each lemma to it's index.
    - 'idx_to_lemma.json': Maps each index to it's lemma.
    """

    def __init__(self, path: str) -> None:
        """Initialize an indexer obejct.

        Args:
            path: The path to the output directory.
        """
        self.path = os.path.join(path, 'processed_corpus')
        self.path_in_tokens = os.path.join(self.path, 'pp_token_corpus.txt')
        self.path_in_lemmas = os.path.join(self.path, 'pp_lemma_corpus.txt')
        self.path_out_corpus = os.path.join(path, 'processed_corpus')
        self.path_out_idx_tokens = os.path.join(
            self.path_out_corpus, 'token_idx_corpus.txt')
        self.path_out_idx_lemmas = os.path.join(
            self.path_out_corpus, 'lemma_idx_corpus.txt')
        self.path_out_indexing = os.path.join(path, 'indexing')
        self.path_token_to_idx = os.path.join(
            self.path_out_indexing, 'token_to_idx.json')
        self.path_lemma_to_idx = os.path.join(
            self.path_out_indexing, 'lemma_to_idx.json')
        self.path_idx_to_token = os.path.join(
            self.path_out_indexing, 'idx_to_token.json')
        self.path_idx_to_lemma = os.path.join(
            self.path_out_indexing, 'idx_to_lemma.json')

        self.path_token_terms = os.path.join(
            self.path_out_corpus, 'token_terms.txt')
        self.path_lemma_terms = os.path.join(
            self.path_out_corpus, 'lemma_terms.txt')

        self.path_idx_token_terms = os.path.join(
            self.path_out_corpus, 'token_terms_idxs.txt')
        self.path_idx_lemma_terms = os.path.join(
            self.path_out_corpus, 'lemma_terms_idxs.txt')

        self.path_hierarch_rels_tokens = os.path.join(
            path, 'hierarchy/hierarchical_relations_tokens.json')
        self.path_hierarch_rels_lemmas = os.path.join(
            path, 'hierarchy/hierarchical_relations_lemmas.json')
        self.path_hierarch_rels_tokens_idx = os.path.join(
            path, 'hierarchy/hierarch_rels_tokens_idx.json')
        self.path_hierarch_rels_lemmas_idx = os.path.join(
            path, 'hierarchy/hierarch_rels_lemmas_idx.json')

        self._file_write_threshhold = 10000
        self._docs_processed = 0
        self._start_time = 0
        self._upper_bound = None

    def index_tokens(self):
        print('indexing of tokens...')
        token_to_idx, idx_to_token = self.index(self.path_in_tokens,
                                                self.path_out_idx_tokens,
                                                self.path_token_terms)

        print('Writing idx-word-mappings to file...')
        with open(self.path_token_to_idx, 'w', encoding='utf8') as f:
            json.dump(token_to_idx, f)

        with open(self.path_idx_to_token, 'w', encoding='utf8') as f:
            json.dump(idx_to_token, f)

        print('Generate token-idx-terms...')
        self._terms_to_idxs('token', token_to_idx)

    def index_lemmas(self):
        print('indexing of lemmas...')
        lemma_to_idx, idx_to_lemma = self.index(self.path_in_lemmas,
                                                self.path_out_idx_lemmas,
                                                self.path_lemma_terms)

        print('Writing idx-word-mappings to file...')
        with open(self.path_lemma_to_idx, 'w', encoding='utf8') as f:
            json.dump(lemma_to_idx, f)

        with open(self.path_idx_to_lemma, 'w', encoding='utf8') as f:
            json.dump(idx_to_lemma, f)

        print('Generate lemma-idx-terms...')
        self._terms_to_idxs('lemma', lemma_to_idx)

    def index(self,
              path_in: str,
              path_out: str,
              path_terms: str
              ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create an index representation for the input corpus.

        Create a file where the corpus is represented with token
        indices.
        Create a file where the corpus is represented with lemma
        indices.
        Create json files to map tokens/lemmas to idxs and.
        Create json files to map idxs to tokens/lemmas and.
        """
        self._docs_processed = 0
        self._start_time = time.time()

        # terms = set()
        # with open(path_terms, 'r', encoding='utf8') as fin:
        #     for line in fin:
        #         terms.add(line.strip('\n'))

        word_to_idx = {}
        idx_to_word = {}
        i = 0
        corpus_idx = []
        for doc in get_docs(path_in):
            doc_idx = []
            for sent in doc:
                for word in sent:
                    if word not in word_to_idx:
                        # if word in terms:
                        #     idx = int('00'+str(i))
                        #     print(30*'-')
                        #     print(i, idx)
                        #     word_to_idx[word] = idx
                        #     idx_to_word[idx] = word
                        #     i += 1
                        # else:
                        word_to_idx[word] = i
                        idx_to_word[i] = word
                        i += 1
                idx_sent = [word_to_idx[word] for word in sent]
                doc_idx.append(idx_sent)
            corpus_idx.append(doc_idx)
            doc_idx = []
            self._docs_processed += 1
            self._update_cmd_counter()

            if self._docs_processed % self._file_write_threshhold == 0:
                self._update_cmd_time_info()
                self.write_corpus(corpus_idx, path_out)
                corpus_idx = []

        self._update_cmd_time_info(end=True)
        self.write_corpus(corpus_idx, path_out)
        return word_to_idx, idx_to_word

    @staticmethod
    def docs_to_string(docs: List[List[List[str]]]) -> str:
        docs_str = ''
        for doc in docs:
            for sent in doc:
                sent_str = ' '.join([str(w) for w in sent])
                line = sent_str + '\n'
                docs_str += line
            docs_str += '\n'
        return docs_str

    def write_corpus(self, docs, f_out):
        mode = self._get_write_mode()
        with open(f_out, mode, encoding='utf8') as f:
            f.write(self.docs_to_string(docs))

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_processed <= self._file_write_threshhold:
            return 'w'
        return 'a'

    def _terms_to_idxs(self,
                       level: str,
                       term_to_idx: Dict[str, int]
                       ) -> None:
        """Use a term file to produce a file with indices of terms.

        Args:
            level: 'token' or 'lemma', indicate if tokens or lemmas
                should be processed.
        """
        if level == 'token':
            path_in = self.path_token_terms
            path_out = self.path_idx_token_terms
        elif level == 'lemma':
            path_in = self.path_lemma_terms
            path_out = self.path_idx_lemma_terms

        terms = set()

        with open(path_in, 'r', encoding='utf8') as fin:
            for line in fin:
                terms.add(line.strip('\n'))

        with open(path_out, 'w', encoding='utf8') as fout:
            for t in terms:
                fout.write(str(term_to_idx[t]) + '\n')

    def hierarch_rels_to_token_idx(self) -> None:
        """Convert 'hierarchical_relations_tokens.json' to index repr.

        Output 'hierarch_rels_token_idx.json'.
        """
        with open(self.path_token_to_idx, 'r', encoding='utf8') as f:
            token_to_idx = json.load(f)
        with open(self.path_hierarch_rels_tokens, 'r', encoding='utf8') as f:
            hr = json.load(f)

        hr_idx = {}
        for hyper in hr:
            hyper_idx = token_to_idx[hyper]
            hypos = hr[hyper]
            hypos_idxs = [token_to_idx[hypo] for hypo in hypos]
            hr_idx[hyper_idx] = hypos_idxs

        with open(self.path_hierarch_rels_tokens_idx, 'w', encoding='utf8') as f:
            json.dump(hr_idx, f)

    def hierarch_rels_to_lemma_idx(self) -> None:
        """Convert 'hierarchical_relations_lemmas.json' to index repr.

        Output 'hierarch_rels_lemma_idx.json'.
        """
        with open(self.path_lemma_to_idx, 'r', encoding='utf8') as f:
            lemma_to_idx = json.load(f)
        with open(self.path_hierarch_rels_lemmas, 'r', encoding='utf8') as f:
            hr = json.load(f)

        hr_idx = {}
        for hyper in hr:
            hyper_idx = lemma_to_idx[hyper]
            hypos = hr[hyper]
            hypos_idxs = [lemma_to_idx[hypo] for hypo in hypos]
            hr_idx[hyper_idx] = hypos_idxs

        with open(self.path_hierarch_rels_lemmas_idx, 'w', encoding='utf8') as f:
            json.dump(hr_idx, f)

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


def main():
    from utility_functions import get_config, get_cmd_args
    config = get_config()
    args = get_cmd_args()
    path = config['paths'][args.location][args.corpus]['path_out']
    idxer = Indexer(path)
    idxer.index_tokens()
    idxer.index_lemmas()
    idxer.hierarch_rels_to_token_idx()
    idxer.hierarch_rels_to_lemma_idx()


if __name__ == '__main__':
    main()
