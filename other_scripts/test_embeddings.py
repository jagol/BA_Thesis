# test FastTextE
from embeddings import FastTextE

fpath = 'preprocessed_corpora/dblp/pp_corpus.txt'
# ft = FastTextE().train(fpath)
# print(ft.model.wv['machine'])
# print(ft.model.wv['vector'])

# model_path = 'fasttext_model.bin'
# print('loading model...')
# ft = FastTextE()
# ft.load_model(model_path)
# print('get vectors...')
# print('king:', ft.get_embedding('king'))
# sent = ['This', 'is' 'a', 'test']
# print('This is a test:', ft.get_embeddings(sent))

# Test ElmoE
# from embeddings import ElmoE
#
test_sent = ['this', 'is', 'a', 'dinosaur']
# elmo = ElmoE()
# embs = elmo.get_embeddings(test_sent)
# print(embs)
# print(len(embs))
# print(type(embs))
# print(len(embs[0]))
# print(type(embs[0]))
# print(type(embs[0][0]))

from embeddings import CombinedEmbeddings

cemb = CombinedEmbeddings(model_paths = ['fasttext_model.bin', ''])
embs = cemb.get_embeddings(test_sent)
print(len(embs))
for e in embs:
    print(len(e))