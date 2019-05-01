import pickle
from typing import *
from collections import defaultdict
import json
import numpy as np
# import fasttext
# import subprocess
import os
from allennlp.commands.elmo import ElmoEmbedder
# from allennlp.modules.similarity_functions.cosine import CosineSimilarity
# For pickle to KeyedVectors-conversion.
from numpy import float32 as REAL
from gensim import utils

import logging
logging.getLogger("gensim.models").setLevel(logging.WARNING)
logging.getLogger("gensim.scripts.glove2word2vec").setLevel(logging.WARNING)
logging.getLogger("gensim").setLevel(logging.WARNING)
from gensim.models import Word2Vec, KeyedVectors
# from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
logging.getLogger("gensim.models").setLevel(logging.WARNING)
logging.getLogger("gensim.scripts.glove2word2vec").setLevel(logging.WARNING)
logging.getLogger("gensim").setLevel(logging.WARNING)
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
                             term_ids_to_embs_global: Dict[int, List[float]]
                             ) -> Dict[int, List[float]]:
        """Get all embeddings for the given terms from the given file.

        Args:
            term_ids: The ids of the input terms.
            emb_path: The path to the given embedding file.
        Return:
            A dictionary of the form: {term_id: embedding}
        """
        logging.getLogger("gensim.models").setLevel(logging.WARNING)
        logging.getLogger("gensim.scripts.glove2word2vec").setLevel(
            logging.WARNING)
        logging.getLogger("gensim").setLevel(logging.WARNING)
        model = KeyedVectors.load(emb_path)
        term_id_to_emb = {}
        for term_id in term_ids:
            try:
                term_id_to_emb[term_id] = model.wv[str(term_id)]
            except KeyError:
                term_id_to_emb[term_id] = term_ids_to_embs_global[term_id]
        return term_id_to_emb


class ElmoE(Embeddings):

    def __init__(self):
        # self._options = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
        # self._weights = 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
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

    def train(self,
              path_corpus: str,
              fname: str,
              path_out_dir: str,
              term_ids: Set[int],
              doc_ids: Set[int]
              ) -> str:
        """'Train' ELMo embeddings. This means averaging for context.

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
        raw_path_elmo_context = 'embeddings/embs_token_ELMo_context.pickle'
        path_elmo_context = os.path.join(path_out_dir, raw_path_elmo_context)
        elmo_c_embs = pickle.load(path_elmo_context)
        averaged_embs = self.get_averaged_embs(elmo_c_embs, term_ids, doc_ids)
        vector_size = len(averaged_embs[list(averaged_embs.values())[0]])
        m = Word2VecKeyedVectors(vector_size=vector_size)
        m.vocab = averaged_embs
        m.vectors = np.array(list(averaged_embs.values()))
        my_save_word2vec_format(binary=True, fname='train.bin',
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
        # path_corpus = 'output/dblp/processed_corpus/pp_lemma_corpus.txt'
        os.system(call_vocab_count.format(path_glove, path_corpus))
        os.system(call_coocur.format(path_glove, path_corpus, path_glove))
        os.system(call_shuffle.format(path_glove, path_glove, path_glove))
        os.system(call_glove.format(path_glove, path_glove_format, path_glove))
        # print('raw_fout_glove:', raw_fout_glove)
        # print('raw_fout_w2v:', raw_fout_w2v)
        # print('path_glove_format:', path_glove_format)
        # print('path_w2v_format:', path_w2v_format)
        # print('path_corpus:', path_corpus)
        # print('fname', fname)
        # print('path_out_dir:', path_out_dir)

        # Turn glove format to w2v format.
        _ = glove2word2vec(path_glove_format+'.txt', path_w2v_format+'.vec')
        model = KeyedVectors.load_word2vec_format(path_w2v_format+'.vec')
        model.save(path_w2v_format+'.vec')
        # if '16945' not in model.wv.vocab and '16945' in

        # Remove GloVe format files.
        cmd = os.path.join(path_out_dir, 'embeddings/glove*')
        os.system('rm {}'.format(cmd))

        return path_w2v_format+'_GloVe.vec'


# class FastTextE(Embeddings):
#
#     def __init__(self):
#         self.mpath = 'fasttext_model.bin'
#         self.model = None
#
#     def train(self, input_data: str, path_model: str) -> None:
#         """Train a fasttext model.
#
#         Args:
#             input_data: The path to the text file used for training.
#             path_model: Name under which the model is saved.
#         Output:
#             The model is saved in self.model.
#             The model is saved as a binary file in <model_name>.bin.
#             The model is saved as a vector text file in <model_name>.vec.
#         """
#         self.model = fasttext.skipgram(input_data, path_model)
#
#     def load_model(self, fpath: Union[None, str] = None) -> None:
#         if fpath:
#             self.mpath = fpath
#         self.model = fasttext.load_model(self.mpath)
#
#     def get_embeddings(self, sent: List[str]) -> embeddings_type:
#         return [self.model[word] for word in sent]
#
#     def get_embedding(self, word: str):
#         return self.model[word]


class Word2VecE(Embeddings):

    # def __init__(self):
    #     self._mpath = 'GoogleNews-vectors-negative300.bin'
    #     self._model = gensim.models.KeyedVectors.load_word2vec_format(
    #         self._mpath, binary=True)
    #
    # def get_embedding(self, word: str) -> Iterator[float]:
    #     """Get the word2vec embeddings for all tokens in <sent>."""
    #     return self._model.wv[word]
    #
    # def get_embeddings(self, sent: List[str]) -> embeddings_type:
    #     """Get the word2vec embeddings for all tokens in <sent>."""
    #     return [self._model.wv[word] for word in sent]

    # @staticmethod
    # def load_term_embeddings(term_ids: Set[str],
    #                          emb_path: str
    #                          )-> Dict[str, List[float]]:
    #     """Get all embeddings for the given terms from the given file.
    #
    #     Args:
    #         term_ids: The ids of the input terms.
    #         emb_path: The path to the given embedding file.
    #     Return:
    #         A dictionary of the form: {term_id: embedding}
    #     """
    #     # model = gensim.models.KeyedVectors.load_word2vec_format(
    #     #     emb_path, binary=True)
    #     # model = gensim.models.Word2Vec.load(emb_path)
    #     model = KeyedVectors.load(emb_path)
    #     term_id_to_emb = {}
    #     for term_id in term_ids:
    #         term_id_to_emb[term_id] = model.wv[term_id]
    #     return term_id_to_emb

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
        raise NotImplementedError
    else:
        raise Exception('Error. Embedding type {} not known.'.format(emb_type))


class CombinedEmbeddings(Embeddings):

    def __init__(self,
                 model_types: List[str] = ('fasttext', 'elmo'),
                 model_paths: List[str] = ('', '')
                 ) -> None:
        self.model_types = model_types
        self.model_paths = model_paths
        self.model_mapping = {
            'word2vec': Word2VecE,
            'glove': GloVeE,
            # 'fasttext': FastTextE,
            'elmo': ElmoE
        }
        self.models = self._get_models()

    def _get_models(self):
        models = []

        for i in range(len(self.model_types)):
            mtype = self.model_types[i]
            mpath = self.model_paths[i]
            model = self.model_mapping[mtype]()
            if mpath:
                model.load_model(mpath)
            models.append(model)

        return models

    def get_embeddings(self,
                       sent: List[str]
                       ) -> embeddings_type:
        combined_vectors = []
        model_word_matrix = []  # rows -> models, columns -> words
        for model in self.models:
            embeddings = model.get_embeddings(sent)
            model_word_matrix.append(embeddings)

        # rows -> words, columns -> models
        word_model_matrix = list(zip(*model_word_matrix))

        for word_tpl in word_model_matrix:
            word_vec = np.hstack(word_tpl)
            combined_vectors.append(word_vec)

        return combined_vectors


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
        print(total_vec, vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8(
                    "%s %s\n" % (word, ' '.join(repr(val) for val in row))))


    # def get_avg_emb(self, term_id: str, corpus: List[int]) -> Iterator[float]:
    #     """Get the average Embedding for all occurences of a term.
    #
    #     Args:
    #         term_id: The term for which the avg embeddings is computed.
    #         corpus: A list of document indices which make up the corpus.
    #     Return:
    #         The average term embedding as a numpy array.
    #     """
    #     sents, indices = get_term_sents(term_id, corpus)
    #     # returns list of tuples (lemmatized sentences, index at which term is)
    #     occurence_embeddings = []  # List of embeddings of term
    #     for i in range(len(sents)):
    #         sent, idx = sents[i], indices[i]
    #         term_embedding = self.get_embeddings(sent)[idx]
    #         occurence_embeddings.append(term_embedding)
    #     # avg the occurence embeddings
    #     avg_embeddings = np.mean(occurence_embeddings, axis=1)
    #     return avg_embeddings


# def train_fasttext(path: str, level: str) -> None:
#     """Train fasttext models for tokens and lemmas.
#
#     Args:
#         path: Path to the output directory.
#         level: 't' if token, 'l' if lemma.
#     """
#     if level == 't':
#         path_corpus = 'processed_corpus/pp_token_corpus_1000.txt'
#         path_model = 'embeddings/fasttext_token_embeddings'
#     elif level == 'l':
#         path_corpus = 'processed_corpus/pp_lemma_corpus_1000.txt'
#         path_model = 'embeddings/fasttext_lemma_embeddings'
#
#     path_in = os.path.join(
#         path, path_corpus)
#     path_out = os.path.join(
#         path, path_model)
#     embedder = FastTextE()
#     embedder.train(path_in, path_out)


# def train_w2v(path: str) -> None:
#     """Train w2v models for tokens and lemmas.
#
#     Args:
#         path: Path to the output directory.
#     """
#     # Train fasttext on tokens.
#     path_in = os.path.join(
#         path, 'output/dblp/processed_corpus/pp_token_corpus_1000.txt')
#     path_out = os.path.join(
#         path, 'output/dblp/embeddings/w2v_token_embeddings')
#     embedder = Word2VecE()
#     embedder.train(path_in, path_out)
#
#     # Train fasttext on lemmas.
#     path_in = os.path.join(
#         path, 'output/dblp/processed_corpus/pp_lemma_corpus_1000.txt')
#     path_out = os.path.join(
#         path, 'output/dblp/embeddings/w2v_lemma_embeddings')
#     embedder = Word2VecE()
#     embedder.train(path_in, path_out)


# if __name__ == '__main__':
#     elmo = ElmoE()
#     this_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[0]
#     is_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[1]
#     man_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[3]
#     woman_embedding = elmo.get_embeddings(['This', 'is', 'a', 'woman', '.'])[3]
#     sim_man_woman = CosineSimilarity().forward(man_embedding, woman_embedding)
#     sim_this_woman = CosineSimilarity().forward(
#         this_embedding, woman_embedding)
#     sim_is_woman = CosineSimilarity().forward(is_embedding, woman_embedding)
#     sim_this_is = CosineSimilarity().forward(this_embedding, is_embedding)
#     print('man vs woman:', sim_man_woman)
#     print('this vs woman:', sim_this_woman)
#     print('is vs woman:', sim_is_woman)
#     print('this vs is:', sim_this_is)
#     print('--------')
#     this_embedding = elmo.get_embeddings(['This', 'is', 'a', 'cat', '.'])[0]
#     is_embedding = elmo.get_embeddings(['This', 'is', 'a', 'dog', '.'])[1]
#     cat_embedding = elmo.get_embeddings(['This', 'is', 'a', 'cat', '.'])[3]
#     dog_embedding = elmo.get_embeddings(['This', 'is', 'a', 'dog', '.'])[3]
#     sim_cat_dog = CosineSimilarity().forward(cat_embedding, dog_embedding)
#     sim_this_dog = CosineSimilarity().forward(this_embedding, dog_embedding)
#     sim_is_dog = CosineSimilarity().forward(is_embedding, dog_embedding)
#     print('cat vs dog:', sim_cat_dog)
#     print('this vs dog:', sim_this_dog)
#     print('is vs dog:', sim_is_dog)
#     print('--------')
#     w2v = Word2VecE()
#     sent1 = ['woman']
#     sent2 = ['man']
#     woman_embedding = w2v.get_embeddings(sent1)[0]
#     man_embedding = w2v.get_embeddings(sent2)[0]
#     sim_man_woman = w2v._model.similarity('man', 'woman')
#     print('man vs woman:', sim_man_woman)
#     print(man_embedding)
#     print(len(man_embedding))
#     ft = FastTextE()
#     print(ft.get_embeddings(sent1)[0])
#     print(ft.get_embeddings(sent2)[0])
#     print(len(ft.get_embeddings(sent2)[0]))
