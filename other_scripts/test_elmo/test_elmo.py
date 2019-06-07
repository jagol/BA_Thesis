from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "/home/pogo/Dropbox/UZH/BA_Thesis/code/other_scripts/test_elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "/home/pogo/Dropbox/UZH/BA_Thesis/code/other_scripts/test_elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0.5)
# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', 'is', 'hard', '.'], ['Another', '.'], ['Hey']]
character_ids = batch_to_ids(sentences)
embeddings = elmo(character_ids)
print(len(sentences[0]))
print(len(character_ids[0]))
print(len(embeddings['elmo_representations']))
print(embeddings)
print(embeddings['elmo_representations'][0])
print(embeddings['elmo_representations'][1])

# list[list[list[list[float]]]]
# layer-sents-sent-token-weight
# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector


"""
PLAN:
- write class as interface for elmo
- for each word: 
    get glove/word2vec-embedding
    get elmo-ebedding
    concatenate
- do similarity computation etc
"""