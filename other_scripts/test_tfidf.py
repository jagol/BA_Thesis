from corpus import *

corpus = set([1, 6, 9])
term_ids = set(['13', '7', '40'])

tf, df, idf, tfidf = get_tfidf(term_ids, corpus, key='doc')
print('term_ids:', term_ids)
print('tf:', tf)
print('df:', df)
print('idf:', idf)
print('tfidf:',  tfidf)
