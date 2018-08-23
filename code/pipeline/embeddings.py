from typing import *
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.similarity_functions.cosine import CosineSimilarity
options_file = "/home/pogo/Dropbox/UZH/BA_Thesis/code/other_scripts/test_elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "/home/pogo/Dropbox/UZH/BA_Thesis/code/other_scripts/test_elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0.5)


class Embeddings:
    """Interface to all embeddings used in the pipeline."""

    def get_embeddings(self, sent: List[str]) -> List[torch.Tensor]:
        pass


class ElmoE(Embeddings):

    def __init__(self):
        self._options_file = "./elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        self._weight_file = "./elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
        self._elmo = Elmo(self._options_file, self._weight_file, 2, dropout=0.5)

    def get_embeddings(self, sent: List[str]) -> List[torch.Tensor]:
        """Get embeddings for all tokens in <sent>.

        Args:
            sent: The sentence the word appears in.
        Return:
            The concatenation of the two hidden layer embeddings.
        """
        character_ids = batch_to_ids([sent])
        embeddings = self._elmo(character_ids)
        layer1 = embeddings['elmo_representations'][0][0]
        layer2 = embeddings['elmo_representations'][1][0]
        concatenation = [torch.cat((tpl[0], tpl[1]), 0)
                         for tpl in zip(layer1, layer2)]
        return  concatenation


class GloveE(Embeddings):
    pass


class FastTextE(Embeddings):
    pass


class Word2VecE(Embeddings):
    pass


if __name__ == '__main__':
    elmo = ElmoE()
    this_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[0]
    is_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[1]
    man_embedding = elmo.get_embeddings(['This', 'is', 'a', 'man', '.'])[3]
    woman_embedding = elmo.get_embeddings(['This', 'is', 'a', 'woman', '.'])[3]
    sim_man_woman = CosineSimilarity().forward(man_embedding, woman_embedding)
    sim_this_woman = CosineSimilarity().forward(this_embedding, woman_embedding)
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
    sim_this_dog = CosineSimilarity().forward(this_embedding,
                                                dog_embedding)
    sim_is_dog = CosineSimilarity().forward(is_embedding, dog_embedding)
    print('cat vs dog:', sim_cat_dog)
    print('this vs dog:', sim_this_dog)
    print('is vs dog:', sim_is_dog)
