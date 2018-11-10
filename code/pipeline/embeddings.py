from typing import *
import torch
import numpy as np
import fasttext
# from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.similarity_functions.cosine import CosineSimilarity
import gensim


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

    def get_embeddings(self, sent: List[str]) -> embeddings_type:
        """Get embeddings for all tokens in <sent>.

        Args:
            sent: The sentence the word appears in.
        Return:
            The concatenation of the two hidden layer embeddings.
        """
        # character_ids = batch_to_ids([sent])
        # embeddings = self._elmo(character_ids)
        # layer1 = embeddings['elmo_representations'][0][0]
        # layer2 = embeddings['elmo_representations'][1][0]
        # concatenation = [torch.cat((tpl[0], tpl[1]), 0)
        #                  for tpl in zip(layer1, layer2)]
        # return concatenation
        embeddings = self.elmo.embed_sentence(sent)
        vectors_layer3 = embeddings[2]
        return vectors_layer3


class GloveE(Embeddings):
    pass


class FastTextE(Embeddings):

    def __init__(self):
        self.mpath = 'fasttext_model.bin'
        self.model = None

    def train(self, input_data: str, model_name: str) -> None:
        """Train a fasttext model.

        Args:
            input_data: The path to the text file used for training.
            model_name: Name under which the model is saved.
        Output:
            The model is saved in self.model.
            The model is saved as a binary file in <model_name>.bin.
            The model is saved as a vector text file in <model_name>.vec.
        """
        self.model = fasttext.skipgram(input_data, model_name)

    def load_model(self, fpath: Union[None, str] = None) -> None:
        if fpath:
            self.mpath = fpath
        self.model = fasttext.load_model(self.mpath)

    def get_embeddings(self, sent: List[str]) -> embeddings_type:
        return [self.model[word] for word in sent]

    def get_embedding(self, word: str):
        return self.model[word]


class Word2VecE(Embeddings):

    def __init__(self):
        self._mpath = 'GoogleNews-vectors-negative300.bin'
        self._model = gensim.models.KeyedVectors.load_word2vec_format(
            self._mpath, binary=True)

    def get_embeddings(self, sent: List[str]) -> embeddings_type:
        """Get the word2vec embeddings for all tokens in <sent>."""
        return [self._model.wv[token] for token in sent]


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