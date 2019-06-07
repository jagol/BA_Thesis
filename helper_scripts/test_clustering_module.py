from clustering import Clustering
# from embeddings import *

words = ['computer', 'algorithm', 'program', 'bear', 'cat', 'snake',
         'fish', 'tree', 'flower', 'gras', 'tea', 'water', 'milk']

# embedder = FastTextE()
# embedder.load_model()
emb_dict = {}
with open('fasttext-wiki-news-300d-1M.vec', 'r', encoding='utf8') as f:
    for line in f:
        sp_line = line.split(' ')
        token, vector = sp_line[0], sp_line[1:]
        emb_dict[token] = vector
word_embeddings = [emb_dict[word] for word in words]
clus = Clustering()
print(clus)
print(clus.clus_type)
print(clus.affinity)
print(clus.fit(word_embeddings))