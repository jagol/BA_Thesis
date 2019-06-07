from corpus import Corpus

c = Corpus(set([10, 205, 123354]), 'preprocessed_corpora/dblp/pp_corpus.txt')
i = 0
for d in c.get_corpus_docs():
	print(i, d)
	i += 1

print(c.get_orig_corpus_len())
