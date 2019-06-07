import os
import pickle
# import sqlite3
import logging
from typing import Dict, List, Set, Any, Iterator
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from gensim import utils
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from utility_functions import get_docs


embeddings_type = List[Iterator[float]]


class Embeddings:
    """Interface to all embeddings used in the pipeline."""

    @staticmethod
    def train(path_corpus: str,
              fname: str,
              path_out_dir: str,
              term_ids: Set[int],
              doc_ids: Set[int]
              ) -> str:
        raise NotImplementedError

    @staticmethod
    def load_term_embeddings(term_ids: Set[int],
                             emb_path: str,
                             idx_to_term: Dict[int, str]
                             ) -> Dict[int, List[float]]:
        """Get all embeddings for the given terms from the given file.

        Args:
            term_ids: The ids of the input terms.
            emb_path: The path to the given embedding file.
            idx_to_term: Maps term_id to term.
        Return:
            A dictionary of the form: {term_id: embedding}
        """
        pck = False
        if emb_path.endswith('.pickle'):
            pck = True
            print('  SPECIAL CASE: load embeddings from pickle...')
            # if stored as pickle, each term-id is mapped to multiple
            # embeddings. So average the embeddings per term id.
            with open(emb_path, 'rb') as f:
                emb_dict = pickle.load(f)
                # print('  Calculating average embeddings...')
                # model = {}
                # for term_id in emb_dict:
                #     embs = []
                #     for doc_id in emb_dict[term_id]:
                #         embs.extend(emb_dict[term_id][doc_id])
                #     model[term_id] = np.mean(embs, axis=0)
                model = {tid: emb for tid, emb in emb_dict.items()}
        else:
            logging.getLogger("gensim.models").setLevel(logging.WARNING)
            logging.getLogger("gensim.scripts.glove2word2vec").setLevel(
                logging.WARNING)
            logging.getLogger("gensim").setLevel(logging.WARNING)
            print('Load embeddings from:')
            print(emb_path)
            try:
                model = KeyedVectors.load(emb_path)
            except:
                model = Word2VecKeyedVectors.load_word2vec_format(emb_path,
                                                                  binary=True)
        term_id_to_emb = {}
        global_embs_ids = []
        for term_id in term_ids:
            try:
                if pck:
                    term_id_to_emb[term_id] = model[term_id]
                else:
                    term_id_to_emb[term_id] = model.wv[str(term_id)]
            except KeyError:
                global_embs_ids.append((term_id, idx_to_term[term_id]))
                # term_id_to_emb[term_id] = term_ids_to_embs_global[term_id]
        if global_embs_ids:
            print('WARNING: No embeddings found for:', global_embs_ids)
            print('WARNING: {} terms excluded.'.format(len(global_embs_ids)))
        return term_id_to_emb


class ElmoE(Embeddings):

    # get_term_embs_stmt = 'SELECT DocID, Embedding WHERE TermID = {}'

    def __init__(self):
        self.elmo = ElmoEmbedder()

    def get_embeddings(self, sent: List[str], mode: int) -> embeddings_type:
        """Get embeddings for all tokens in <sent>.

        Args:
            sent: The sentence to be embedded.
            mode: determines what is returned.
                if 0: The context insensitive repr is returned.
                if 1: The last LSTM-layer is returned (context
                    sensitive repr).
                if 2: The average of the context insensitive repr. and
                    the two lstm layers is returned.
        """
        # character_ids = batch_to_ids([sent])
        # embeddings = self._elmo(character_ids)
        # layer1 = embeddings['elmo_representations'][0][0]
        # layer2 = embeddings['elmo_representations'][1][0]
        # concatenation = [torch.cat((tpl[0], tpl[1]), 0)
        #                  for tpl in zip(layer1, layer2)]
        # return concatenation
        embeddings = self.elmo.embed_sentence(sent)
        if mode == 0:
            return embeddings[0]
        elif mode == 1:
            return embeddings[2]
        elif mode == 2:
            average = [np.mean((tpl[0], tpl[1], tpl[2]), axis=0) for tpl in
                       zip(embeddings[0], embeddings[1], embeddings[2])]
            return average

    # @classmethod
    # def get_avg_emb_by_term_id(cls, term_id: int, conn) -> np.ndarray:
    #     """Get average term embedding for the given term-id.
    #
    #     Args:
    #         term_id: The id of the term.
    #         conn: The connection to the elmo-database.
    #     """
    #     embeddings = cls.get_embs_by_id(term_id, conn)  # {doc_id: emb}
    #     return np.mean(embeddings.keys(), 0)
    #
    # @classmethod
    # def get_embs_by_id(cls, term_id: int, conn) -> Dict[int, np.ndarray]:
    #     """Get all embeddings of a term by document.
    #
    #     Args:
    #         term_id: The id of the term.
    #         conn: The connection to the elmo-database.
    #     """
    #     cursor = conn.execute(cls.get_term_embs_stmt.format(term_id))
    #     for row in cursor:
    #         doc_id = int(row[0])
    #         emb_str = row[1]
    #         emb = np.array([float(i) for i in emb_str.split(',')])
    #

    @staticmethod
    def train(path_corpus: str,
              fname: str,
              path_out_dir: str,
              term_ids: Set[int],
              doc_ids: Set[int]
              ) -> str:
        """'Train ELMo embeddings. This means averaging for context.

        ******
        IMPORTANT: At the moment no averaging is done! So the input
        embeddings are just returned as output embeddings!
        ******

        Args:
            path_corpus: The path to the text file used for training.
            fname: The filename for the embedding file.
            path_out_dir: The path to the output directory.
            term_ids: The set of current term-ids.
            doc_ids: The set of doc-ids making up the current subcorpus.
        Return:
            The path to the embedding file.
        """
        raw_path = 'embeddings/{}.vec'.format(fname)
        path_out = os.path.join(path_out_dir, raw_path)
        # raw_path_elmo_context = 'embeddings/embs_token_ELMo_avg.pickle'
        # path_elmo_context = os.path.join(path_out_dir, raw_path_elmo_context)
        # elmo_c_embs = pickle.load(path_elmo_context)
        # averaged_embs = self.get_averaged_embs

        # *** tmp lines ***
        tmp_path_in = os.path.join(path_out_dir,
                                   'embeddings/embs_token_ELMo_avg.pickle')
        averaged_embs = pickle.load(open(tmp_path_in, 'rb'))
        averaged_embs = {str(k): v for k, v in averaged_embs.items()}
        # *** tmp lines ***
        key = list(averaged_embs.keys())[0]
        vector_size = len(averaged_embs[key])
        m = Word2VecKeyedVectors(vector_size=vector_size)
        m.vocab = averaged_embs
        m.vectors = np.array(list(averaged_embs.values()))
        my_save_word2vec_format(binary=True, fname=path_out,
                                total_vec=len(averaged_embs), vocab=m.vocab,
                                vectors=m.vectors)
        return path_out

    @staticmethod
    def get_averaged_embs(elmo_c_embs: Dict[int, Dict[int, List[np.ndarray]]],
                          term_ids: Set[int],
                          doc_ids: Set[int]
                          ) -> Dict[int, np.ndarray]:
        """Get the average elmo embeddings for given terms and doc-ids.

        Args:
            elmo_c_embs: {term_id: doc_id: [list of embeddings]}
            term_ids: A set of term-ids.
            doc_ids: A set of doc-ids.
        Return:
            averaged_embs: {term_id: average-embedding}
        """
        averaged_embs = {}
        for term_id in term_ids:
            term_embs = []
            for doc_id in elmo_c_embs[term_id]:
                if doc_id in doc_ids:
                    term_embs.extend(elmo_c_embs[term_id][doc_id])
            averaged_embs[term_id] = np.mean(term_embs, axis=0)
        return averaged_embs


class GloVeE(Embeddings):

    @staticmethod
    def train(path_corpus: str,
              fname: str,
              path_out_dir: str,
              term_ids: Set[int],
              doc_ids: Set[int]
              ) -> str:
        """Train GloVe embeddings.

        All setting are default (as in demo-script) except for the
        min-count which is set to one, such that all terms get
        embeddings.

        Args:
            path_corpus: The path to the text file used for training.
            fname: The filename for the embedding file.
            path_out_dir: The path to the output directory.
            term_ids: The set of current term-ids.
            doc_ids: The set of doc-ids making up the current subcorpus.
        Return:
            The path to the embedding file.
        """
        # path_glove = './glove_github/glove/build/'
        path_glove = './glove/'
        raw_fout_glove = 'embeddings/glove_format_'+fname
        raw_fout_w2v = 'embeddings/' + fname
        path_glove_format = os.path.join(path_out_dir, raw_fout_glove)
        path_w2v_format = os.path.join(path_out_dir, raw_fout_w2v)

        call_vocab_count = ('{}vocab_count -min-count 1 -verbose 2 < {} > '
                            'vocab.txt')
        call_coocur = ('{}cooccur -memory 4.0 -vocab-file vocab.txt -verbose 2'
                       ' -window-size 15 < {} > cooccurrence.bin')
        call_shuffle = ('{}shuffle -memory 4.0 -verbose 2 < cooccurrence.bin >'
                        ' cooccurrence.shuf.bin')
        call_glove = ('{}glove -save-file {} -threads 10 -input-file '
                      'cooccurrence.shuf.bin -x-max 10 -iter 15 -vector-size '
                      '100 -binary 0 -vocab-file vocab.txt -verbose 2')

        os.system(call_vocab_count.format(path_glove, path_corpus))
        os.system(call_coocur.format(path_glove, path_corpus, path_glove))
        os.system(call_shuffle.format(path_glove, path_glove, path_glove))
        os.system(call_glove.format(path_glove, path_glove_format, path_glove))

        # Turn glove format to w2v format.
        _ = glove2word2vec(path_glove_format+'.txt', path_w2v_format+'.vec')
        model = KeyedVectors.load_word2vec_format(path_w2v_format+'.vec')
        model.save(path_w2v_format+'.vec')
        # if '16945' not in model.wv.vocab and '16945' in

        # Remove GloVe format files.
        cmd = os.path.join(path_out_dir, 'embeddings/glove*')
        os.system('rm {}'.format(cmd))

        return path_w2v_format+'.vec'


class Word2VecE(Embeddings):

    @staticmethod
    def train(path_corpus: str,
              fname: str,
              path_out_dir: str,
              term_ids: Set[int],
              doc_ids: Set[int]
              ) -> str:
        """Train word2vec embeddings.

        Args:
            path_corpus: The path to the text file used for training.
            fname: The filename for the embedding file.
            path_out_dir: The path to the output directory.
            term_ids: The set of current term-ids.
            doc_ids: The set of doc-ids making up the current subcorpus.
        Return:
            The path to the embedding file.
        """
        raw_path = 'embeddings/{}.vec'.format(fname)
        path_out = os.path.join(path_out_dir, raw_path)
        sentences = Sentences(path_corpus)
        model = Word2Vec(sentences, min_count=1, workers=10)
        model.wv.save(path_out)
        return path_out


class Sentences:
    """Class to feed input corpus to gensim.

    Convert corpus from generator to iterator and feed one sentence
    per iteration.
    """

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for doc in get_docs(self.path):
            for sent in doc:
                yield sent


def get_emb(emb_type: str) -> Any:
    """Return an embedding class depending on the embedding type.

    Args:
        emb_type: The type of the embeddings: 'Word2Vec', 'GloVe'
            or 'ELMo'.
    Return:
        An embedding object.
    """
    if emb_type == 'Word2Vec':
        return Word2VecE
    elif emb_type == 'GloVe':
        return GloVeE
    elif emb_type == 'ELMo':
        return ElmoE
    else:
        raise Exception('Error. Embedding type {} not known.'.format(emb_type))


# class CombinedEmbeddings(Embeddings):
#
#     def __init__(self,
#                  model_types: List[str] = ('fasttext', 'elmo'),
#                  model_paths: List[str] = ('', '')
#                  ) -> None:
#         self.model_types = model_types
#         self.model_paths = model_paths
#         self.model_mapping = {
#             'word2vec': Word2VecE,
#             'glove': GloVeE,
#             # 'fasttext': FastTextE,
#             'elmo': ElmoE
#         }
#         self.models = self._get_models()
#
#     def _get_models(self):
#         models = []
#
#         for i in range(len(self.model_types)):
#             mtype = self.model_types[i]
#             mpath = self.model_paths[i]
#             model = self.model_mapping[mtype]()
#             if mpath:
#                 model.load_term_embeddings(mpath)
#             models.append(model)
#
#         return models
#
#     def get_embeddings(self,
#                        sent: List[str]
#                        ) -> embeddings_type:
#         combined_vectors = []
#         model_word_matrix = []  # rows -> models, columns -> words
#         for model in self.models:
#             embeddings = model.get_embeddings(sent, 2)
#             model_word_matrix.append(embeddings)
#
#         # rows -> words, columns -> models
#         word_model_matrix = list(zip(*model_word_matrix))
#
#         for word_tpl in word_model_matrix:
#             word_vec = np.hstack(word_tpl)
#             combined_vectors.append(word_vec)
#
#         return combined_vectors


def my_save_word2vec_format(fname, vocab, vectors, binary=True,
                            total_vec=2):
    """Store the input-hidden weight matrix in the same format used by
    the original C word2vec-tool, for compatibility.

    :: Code copied from here:
    https://stackoverflow.com/questions/45981305/
    convert-python-dictionary-to-word2vec-object
    ::

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else
        it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        # print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(np.float32)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8(
                    "%s %s\n" % (word, ' '.join(repr(val) for val in row))))


if __name__ == '__main__':
    path_corpus = ('/mnt/storage/harlie/users/jgoldz/output/dblp/'
                   'processed_corpus/1.txt')
    fname = '1'
    path_out_dir = '/mnt/storage/harlie/users/jgoldz/output/dblp/'
    term_ids = set()
    doc_ids = set()
    idx_to_term = {}
    ElmoE.train(path_corpus, fname, path_out_dir, term_ids, doc_ids)
    m2 = Word2VecKeyedVectors.load_word2vec_format(
        path_out_dir+'embeddings/1.vec', binary=True)
    # ElmoE.load_term_embeddings(term_ids, path_out_dir+'embeddings/1.vec',
    # idx_to_term)