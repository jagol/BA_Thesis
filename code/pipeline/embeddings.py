from typing import *
from collections import defaultdict
import json
import torch
import numpy as np
import fasttext
import os
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.similarity_functions.cosine import CosineSimilarity
import gensim
from utility_functions import get_docs


embeddings_type = List[Iterator[float]]


class Embeddings:
    """Interface to all embeddings used in the pipeline."""

    def get_embeddings(self, sent: List[str]) -> List[torch.Tensor]:
        raise NotImplementedError


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
                if 2: The concatenation of option 0 and 1 is returned.
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
            concatenation = [torch.cat((tpl[0], tpl[1]), 0)
                             for tpl in zip(embeddings[0], embeddings[2])]
            return concatenation

    # def calc_avg_embs_per_doc(self,
    #                           path: str,
    #                           level: str
    #                           ) -> None:
    #     """Compute the average ELMo embedding per document.
    #
    #     Produce a dict of the form:
    #     {term_id: {doc_id: [avg<float>, weight<int>]}
    #
    #     Save it under '/embeddings/avg_embs_per_doc.json'.
    #
    #     Args:
    #         cpath: Path to the output directory.
    #         level: 't' if token, 'l' if lemma.
    #     """
    #     if level == 't':
    #         cpath = os.path.join(
    #             path, 'processed_corpus/pp_token_corpus.txt')
    #         cpath_idx = os.path.join(
    #             path, 'processed_corpus/token_idx_corpus.txt')
    #         path_terms = os.path.join(
    #             path, 'processed_corpus/token_terms_idxs.txt')
    #         path_out = os.path.join(
    #             path, 'embeddings/elmo_avg_embs_per_doc_tokens.json')
    #     elif level == 'l':
    #         cpath = os.path.join(
    #             path, 'processed_corpus/pp_lemma_corpus.txt')
    #         cpath_idx = os.path.join(
    #             path, 'processed_corpus/lemma_idx_corpus.txt')
    #         path_terms = os.path.join(
    #             path, 'processed_corpus/lemma_terms_idxs.txt')
    #         path_out = os.path.join(
    #             path, 'embeddings/elmo_avg_embs_per_doc_lemmas.json')
    #
    #     terms = set()
    #     with open(path_terms, 'r', encoding='utf8') as fin:
    #         for line in fin:
    #             terms.add(line.strip('\n'))
    #
    #     avg_embs = defaultdict(lambda: defaultdict(int))
    #     # Loop though both corpus files at once. Get term embeddings.
    #     doc_counter = -1
    #     # for word_doc, idx_doc in zip(get_docs(cpath), get_docs(cpath_idx)):
    #     #     doc_counter += 1
    #     #     print('processing {}'.format(doc_counter))
    #     #     doc_embs = defaultdict(list)   #  {term_idx: [emb1, emb2, ...]}
    #     #     for i in range(len(idx_doc)):
    #     #         idx_sent = idx_doc[i]
    #     #         word_sent = word_doc[i]
    #     #         for j in range(len(idx_sent)):
    #     #             idx = idx_sent[j]
    #     #             if idx in terms:
    #     #                 emb = self.get_embeddings(word_sent)[j]
    #     #                 doc_embs[idx].append(emb)
    #     for doc in get_docs(cpath):
    #         doc_counter += 1
    #         print('processing {}'.format(doc_counter))
    #         for i in range(len(doc)):
    #             word_sent = doc[i]
    #             emb = self.get_embeddings(word_sent)
    #
    #         # # Calculate average term embeddings per document
    #         # # and it's weight (=number of term occurences).
    #         # for idx in doc_embs:
    #         #     embs = doc_embs[idx]
    #         #     weight = len(embs)
    #         #     avg_emb = np.mean(embs)
    #         #     avg_embs[idx][doc_counter] = [avg_emb.tolist(), weight]
    #
    #         if doc_counter >= 10:
    #             break
    #
    #     with open(path_out, 'w', encoding='utf8') as f:
    #         json.dump(avg_embs, f)


class GloveE(Embeddings):
    pass


class FastTextE(Embeddings):

    def __init__(self):
        self.mpath = 'fasttext_model.bin'
        self.model = None

    def train(self, input_data: str, path_model: str) -> None:
        """Train a fasttext model.

        Args:
            input_data: The path to the text file used for training.
            path_model: Name under which the model is saved.
        Output:
            The model is saved in self.model.
            The model is saved as a binary file in <model_name>.bin.
            The model is saved as a vector text file in <model_name>.vec.
        """
        self.model = fasttext.skipgram(input_data, path_model)

    def load_model(self, fpath: Union[None, str] = None) -> None:
        if fpath:
            self.mpath = fpath
        self.model = fasttext.load_model(self.mpath)

    def get_embeddings(self, sent: List[str]) -> embeddings_type:
        return [self.model[word] for word in sent]

    def get_embedding(self, word: str):
        return self.model[word]


class Word2VecE(Embeddings):

    # def __init__(self):
    #     self._mpath = 'GoogleNews-vectors-negative300.bin'
    #     self._model = gensim.models.KeyedVectors.load_word2vec_format(
    #         self._mpath, binary=True)

    def get_embedding(self, word: str) -> Iterator[float]:
        """Get the word2vec embeddings for all tokens in <sent>."""
        return self._model.wv[word]

    def get_embeddings(self, sent: List[str]) -> embeddings_type:
        """Get the word2vec embeddings for all tokens in <sent>."""
        return [self._model.wv[word] for word in sent]

    def train(self, input_data: str, path_model: str) -> None:
        """Train word2vec embeddings.

        Args:
            input_data: The path to the text file used for training.
            path_model: Name/path under which the model is saved.
        """
        sentences = get_docs(input_data)
        model = gensim.models.Word2Vec(sentences, min_count=1, workers = 4)
        model.wv.save(path_model)


class CombinedEmbeddings(Embeddings):

    def __init__(self,
                 model_types: List[str] = ('fasttext', 'elmo'),
                 model_paths: List[str] = ('', '')
                 ) -> None:
        self.model_types = model_types
        self.model_paths = model_paths
        self.model_mapping = {
            'word2vec': Word2VecE,
            'glove': GloveE,
            'fasttext': FastTextE,
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


def train_fasttext(path: str, level: str) -> None:
    """Train fasttext models for tokens and lemmas.

    Args:
        path: Path to the output directory.
        level: 't' if token, 'l' if lemma.
    """
    if level == 't':
        path_corpus = 'processed_corpus/pp_token_corpus_1000.txt'
        path_model = 'embeddings/fasttext_token_embeddings'
    elif level == 'l':
        path_corpus = 'processed_corpus/pp_lemma_corpus_1000.txt'
        path_model = 'embeddings/fasttext_lemma_embeddings'

    path_in = os.path.join(
        path, path_corpus)
    path_out = os.path.join(
        path, path_model)
    embedder = FastTextE()
    embedder.train(path_in, path_out)


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


if __name__ == '__main__':
    elmo = ElmoE()
    this_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[0]
    is_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[1]
    man_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[3]
    woman_embedding = elmo.get_embeddings(['This', 'is', 'a', 'woman', '.'])[3]
    sim_man_woman = CosineSimilarity().forward(man_embedding, woman_embedding)
    sim_this_woman = CosineSimilarity().forward(
        this_embedding, woman_embedding)
    sim_is_woman = CosineSimilarity().forward(is_embedding, woman_embedding)
    sim_this_is = CosineSimilarity().forward(this_embedding, is_embedding)
    print('man vs woman:', sim_man_woman)
    print('this vs woman:', sim_this_woman)
    print('is vs woman:', sim_is_woman)
    print('this vs is:', sim_this_is)
    print('--------')
    this_embedding = elmo.get_embeddings(['This', 'is', 'a', 'cat', '.'])[0]
    is_embedding = elmo.get_embeddings(['This', 'is', 'a', 'dog', '.'])[1]
    cat_embedding = elmo.get_embeddings(['This', 'is', 'a', 'cat', '.'])[3]
    dog_embedding = elmo.get_embeddings(['This', 'is', 'a', 'dog', '.'])[3]
    sim_cat_dog = CosineSimilarity().forward(cat_embedding, dog_embedding)
    sim_this_dog = CosineSimilarity().forward(this_embedding, dog_embedding)
    sim_is_dog = CosineSimilarity().forward(is_embedding, dog_embedding)
    print('cat vs dog:', sim_cat_dog)
    print('this vs dog:', sim_this_dog)
    print('is vs dog:', sim_is_dog)
    print('--------')
    w2v = Word2VecE()
    sent1 = ['woman']
    sent2 = ['man']
    woman_embedding = w2v.get_embeddings(sent1)[0]
    man_embedding = w2v.get_embeddings(sent2)[0]
    sim_man_woman = w2v._model.similarity('man', 'woman')
    print('man vs woman:', sim_man_woman)
    print(man_embedding)
    print(len(man_embedding))
    ft = FastTextE()
    print(ft.get_embeddings(sent1)[0])
    print(ft.get_embeddings(sent2)[0])
    print(len(ft.get_embeddings(sent2)[0]))