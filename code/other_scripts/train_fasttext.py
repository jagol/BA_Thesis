from embeddings import FastTextE

# path_in = '/mnt/storage/harlie/preprocessed_corpora/dblp/pp_corpus.txt'
path_in = './output/dblp/processed_corpus/pp_lemma_corpus.txt'
path_out = './output/dblp/embeddings/fasttext_embeddings.txt'
embedder = FastTextE()
embedder.train(path_in, path_out)