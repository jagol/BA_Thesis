from math import log
from typing import Generator, List, Set, Tuple, Dict
from collections import defaultdict


class Corpus:

    """Class to represent corpora and pseudo corpora as a collection of
    indices to the documents in the original corpus file.
    """

    def __init__(self, docs: Set[int], path: str) -> None:
        """Initialize the corpus.

        Args:
            docs: A list of indices to the docs belonging to the corpus.
            path: The path to the original corpus file
        """
        self.docs = docs
        self.num_docs = len(self.docs)
        self.docs_read = set()
        self.path = path

    def get_corpus_docs(self) -> Generator[Tuple[int, List[List[str]]], None, None]:
        """Get all the documents belonging to the corpus.

        Yield a generator of documents. Each document is a list of
        sentences and each sentence a list of words.
        """
        for i, doc in enumerate(self.get_docs()):
            if i in self.docs:
                doc = [line.strip('\n').split(' ') for line in doc]
                self.docs_read.add(i)
                yield i, doc
            # stop iterating if all docs were fetched
            if len(self.docs_read) == self.num_docs:
                break

        # check if all docs were fetched
        not_extracted = []
        for i in self.docs:
            if i not in self.docs_read:
                not_extracted.append(i)

        # throw exception if not all documents were fetched
        if not_extracted:
            doc_ids = ', '.join([str(i) for i in not_extracted])
            msg = 'Not all documents were extracted. DocIDs: {}'
            raise Exception(msg.format(doc_ids))

    def get_docs(self) -> Generator[List[str], None, None]:
        """Yield documents from given file.

        Each document is a list of lines.
        """
        doc = []
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                if line == '\n':
                    yield doc
                    doc = []
                else:
                    doc.append(line)

    def get_orig_corpus_len(self) -> int:
        i = 0
        for _ in self.get_docs():
            i += 1
        return i

    @staticmethod
    def flatten_doc(doc: List[List[str]])-> List[str]:
        """Convert a list of sentences to a flat list of tokens.

        Args:
            doc: A list of sentences.
        Return:
            A list of tokens concatenated from the sentences.
        """
        return [token for sent in doc for token in sent]


def get_pseudo_corpus(term_ids: Set[str],
                      base_corpus: Set[int],
                      n: int
                      ) -> Set[int]:
    """Generate a pseudo corpus for a given set of terms.

    Find the documents of the corpus using tfidf. The general idea is:
    Those documents, for which the given terms are important, belong
    to the corpus. Thus, for each of the given terms, get the tfidf
    score in the base corpus (corpus from which the most important
    documents for the pseudo corpus are selected). Then get the top n
    documents, for which the terms are most important. The importance
    score for a document d is: score(d) = sum(tfidf(t1...tn, d))
    where t1...tn denotes the set of given terms.

    Args:
        term_ids: The lemma-ids of the terms that define the corpus.
        base_corpus: The document ids, that form the document collection
            from which to choose from.
        n: The top n scored documents are chosen for the pseudo corpus.
            n should be chosen as num_docs / n_clus
            where num_docs denotes the number of documents in the base
            corpus and n_clus denotes the number of clusters (or just
            the number of parts) the base corpus is divided into.
    Return:
        Set of indices which denote the documents beloning to the pseudo
        corpus.
    """
    # tfidf_term: Dict[term_id, Dict[doc_id, tfidf]]]
    # -> Dict[doc_id, List[tfidf_term1, tfidf_term2, ...]]
    tfidf_doc = get_tfidf(term_ids, base_corpus, key='doc')

    for doc_id in tfidf_doc:
        tfidf_doc[doc_id] = sum(tfidf_dict[doc_id])
    ranked_docs = sorted(
        tfidf_doc.items(), key=lambda tpl: tpl[1], reverse=True)
    return ranked_docs[:n]

def get_tfidf(term_ids: Set[str],
              corpus: Set[int],
              key: str = 'term'
              ) -> Defaultdict[int, Dict[str, float]]:
    """Compute the tfidf score for the given terms in the given corpus.

    Args:
        term_ids: The lemma-ids of the lemmas for which tfidf is computed.
        corpus: The document-ids of the documents which make up the
            corpus.
        key: either 'term' or 'doc'.
            If 'term', then the tfidf is returned in the
            following structure: Dict[term_id, List[tfidf_doc1, ...]
            If 'doc', then the tfidf is returned in the following
            structure: Dict[doc_id, List[term1, ...]]
            NOTE: at the moment the method always returns as if key='doc'
    """
    fpath = 'preprocessed_corpora/dblp/lemma_idx_corpus.txt'
    c = Corpus(corpus, fpath)
    docs = c.get_corpus_docs()
    df = defaultdict(int)  # Dict[term_id, document-frequency]
    tf = {}                # Dict[doc_id, Dict[term_id, frequency]]
    for doc_id, doc in docs:
        # calc tf for a doc
        tf_doc = defaultdict(int)
        doc = c.flatten_doc(doc)
        num_tokens = len(doc)
        print(doc_id, doc)
        for term_id in term_ids:
            for lemma_id in doc:
                if lemma_id == term_id:
                    tf_doc[term_id] += 1

        # normalize for document length
        for term_id in tf_doc:
            tf_doc[term_id] = tf_doc[term_id]/num_tokens

        tf[doc_id] = tf_doc

        # update df
        for term_id in term_ids:
            if term_id in tf_doc:
                df[term_id] += 1

    # calc idf
    idf = {}
    for term_id in df:
        idf[term_id] = log(c.num_docs/(1+df[term_id]))

    # calc tfidf
    tfidf = defaultdict(dict)
    for doc_id in tf:
        for term_id in tf[doc_id]:
            tf_term_doc = tf[doc_id][term_id]
            idf_term = idf[term_id]
            tfidf[doc_id][term_id] = tf_term_doc*idf_term

    return tfidf