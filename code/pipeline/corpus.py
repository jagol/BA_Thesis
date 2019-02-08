from math import log
import json
from typing import *
from collections import defaultdict
from utility_functions import get_sim
from numpy import mean


class Corpus:

    """Class to represent corpora and pseudo corpora as a collection of
    indices to the documents in the original corpus file.
    """

    def __init__(self, doc_ids: Set[int], path: str) -> None:
        """Initialize the corpus.

        Args:
            doc_ids: A list of indices to the docs belonging to the corpus.
            path: The path to the original corpus file
        """
        self.doc_ids = doc_ids
        self.num_docs = len(self.doc_ids)
        self.docs_read = set()
        self.path = path
        self.docs = []  # List of tuples (doc_id, doc)

    def get_corpus_docs(self,
                        save_inside: bool = True,
                        ) -> Generator[Tuple[int, List[List[str]]],
                                       None, None]:
        """Get all the documents belonging to the corpus.

        Yield a generator of documents. Each document is a list of
        sentences and each sentence a list of words.

        Args:
            save_inside: If save_inside, then documents are not yielded
                but instead saved in self.docs.
        """
        for i, doc in enumerate(self.get_docs()):
            if i in self.doc_ids:
                doc = [line.strip('\n').split(' ') for line in doc]
                self.docs_read.add(i)
                if save_inside:
                    self.docs.append((i, doc))
                else:
                    yield i, doc
            # stop iterating if all docs were fetched
            if len(self.docs_read) == self.num_docs:
                break

        # check if all docs were fetched
        not_extracted = []
        for i in self.doc_ids:
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


def get_relevant_docs(term_ids: Set[str],
                      base_corpus: Set[str],
                      n: int,
                      tfidf_base: Dict[str, Dict[str, float]],
                      doc_embeddings: Dict[str, List[float]],
                      topic_embedding: List[float],
                      only_tfidf: bool=False
                      ) -> Set[str]:
    """Generate a pseudo corpus (relevant_docs) for given set of terms.

    Find the documents for the pseudo-corpus by calculating a document
    embedding. The doc-embedding is TFIDF-weighted average of all its
    terms. Use the average cluster-direction as a query vector. Find
    the n most closest documents to the query vector. n is determined as
    the |D|/5*level where |D| is the number of all documents and level
    is the depth of the taxonomy tree. Multiplicated with 5 since at
    every level the every branch gets 5 subbranches.

    Args:
        term_ids: The lemma-ids of the terms that define the corpus.
        base_corpus: The document ids, that form the document collection
            from which to choose from.
        n: The top n scored documents are chosen for the pseudo corpus.
            n should be chosen as num_docs / n_clus
            where num_docs denotes the number of documents in the base
            corpus and n_clus denotes the number of clusters (or just
            the number of parts) the base corpus is divided into.
        tfidf_base: The tfidf values for the terms in the entire
            base_corpus.
        doc_embeddings: The document embeddings.
        topic_embedding: The topic embedding.
        only_tfidf: Don't use embeddings but only the summed tfidf
            scores for documents to rank them.
    Return:
        Set of indices which denote the documents beloning to the pseudo
        corpus.
    """
    if only_tfidf:
        return get_relevant_docs_only_tfidf(
            term_ids, base_corpus, n, tfidf_base)

    doc_sims = []
    for doc_id, doc_emb in doc_embeddings.items():
        doc_sims.append((doc_id, get_sim(topic_embedding, doc_emb)))

    top_n = sorted(doc_sims, key=lambda t: t[1], reverse=True)[:n]
    return set([t[0] for t in top_n])


def get_doc_embeddings(tfidf_base: Dict[str, Dict[str, float]],
                       term_ids_to_embs: Dict[str, List[float]]
                       ) -> Dict[str, Any]:
    """Compute document embeddings using term-embeddings and tfidf.

    Compute document embeddings though average of tfidf weighted term
    embeddings.

    The embedding for each document d_e is computed as:
    d_e = avg(tfidf(t1..tn)*emb(t1..tn))
    where t is a term in d.

    Args:
        tfidf_base: The tfidf values for the terms in the entire
            base_corpus (global).
        term_ids_to_embs: Maps term-ids to their global embeddings.
    Return:
        doc_embeddings: {doc-id: embedding}
    """
    doc_embeddings = {}
    for doc_id in tfidf_base:
        doc_emb = []
        tfidf_doc = tfidf_base[doc_id]
        for term_id in tfidf_doc:
            term_emb = term_ids_to_embs[term_id]
            tfidf = tfidf_base[doc_id][term_id]
            term_emb = [tfidf*d for d in term_emb]
            doc_emb.append(term_emb)
        doc_embeddings[doc_id] = mean(doc_emb, axis=0)
    return doc_embeddings


def get_topic_embeddings(clusters: Dict[int, Set[str]],
                         term_ids_to_embs: Dict[str, List[float]]
                         )-> Dict[int, Any]:
    """Compute embeddings for topics/clusters by averaging terms-embs.

    Args:
        clusters: Each cluster is a set of terms.
        term_ids_to_embs: Maps terms to their global embeddings.
    Return:
        topic_embeddings: {cluster-label: embedding}
    """
    topic_embeddings = {}
    for label, clus in clusters.items():
        embs = [term_ids_to_embs[term_id] for term_id in clus]
        topic_embeddings[label] = mean(embs, axis=0)
    return topic_embeddings


def get_relevant_docs_only_tfidf(term_ids: Set[str],
                                 base_corpus: Set[str],
                                 n: int,
                                 tfidf_base: Dict[str, Dict[str, float]]
                                 ) -> Set[str]:
    """Generate a pseudo corpus (relevant_docs) for given set of terms.

    This is the tfidf-only version without usage of embeddings.

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
        tfidf_base: The tfidf values for the terms in the entire
            base_corpus.
    Return:
        Set of indices which denote the documents beloning to the pseudo
        corpus.
    """
    tfidf_doc = {}
    # Calculate document scores.
    for doc_id in base_corpus:
        vals = []  # The importances of a (cluster)terms for a document.
        for term_id in term_ids:
            try:
                vals.append(tfidf_base[doc_id][term_id])
            except KeyError:
                pass
        tfidf_doc[doc_id] = sum(vals)

    # Rank documents by score.
    ranked_docs = sorted(
        tfidf_doc.items(), key=lambda tpl: tpl[1], reverse=True)

    # Return only the ids of the n highest scored documents.
    return set(d[0] for d in ranked_docs[:n])


def get_doc_topic_sims(doc_embeddings: Dict[str, List[float]],
                       topic_embeddings: Dict[int, List[float]]
                       ) -> Dict[str, Dict[int, float]]:
    """Get the similarities between topic and document vectors.

    Args:
        doc_embeddings: Maps doc-ids to their embeddings.
        topic_embeddings: Maps topic-ids to their embeddings.
    Return:
        A dict of the form: {doc-id: {topic/cluster-label: similarity}}
    """
    doc_topic_sims = defaultdict(dict)
    for topic, temb in topic_embeddings.items():
        for doc, demb in doc_embeddings.items():
            doc_topic_sims[doc][topic] = get_sim(temb, demb)
    return doc_topic_sims


def get_topic_docs(doc_topic_sims: Dict[str, Dict[int, float]],
                   m: Union[int, None]=None
                   ) -> Dict[int, Set[str]]:
    """Get the topic/cluster for each document.

    Match all documents to a topic by choosing the max similarity.

    Args:
        doc_topic_sims: Maps doc-ids to a dict of the form:
        {cluster-label: similarity}
        m: If not None, only the m most similar docs are returned.
    Return:
        A dict of the form: {cluster-label: {A set of doc-ids}}
    """
    td_sc = defaultdict(list)  # {topic: [(doc_id, score), ...]}
    for doc in doc_topic_sims:
        topic_sims = doc_topic_sims[doc]
        topic = max(topic_sims.keys(), key=(lambda key: topic_sims[key]))
        td_sc[topic].append((doc, topic_sims[topic]))
    topic_docs = defaultdict(set)

    # If m is defined only use the top m docs.
    if m:
        for tc in td_sc:
            docs_sort = sorted(td_sc[tc], key=(lambda t: t[1]), reverse=True)
            topic_docs[tc] = set([tpl[0] for tpl in docs_sort[:m]])
    else:
        for tc in td_sc:
            doc_ids = [doc_id for doc_id, score in td_sc[tc]]
            topic_docs[tc] = set(doc_ids)
    return topic_docs


def get_tf_corpus(corpus: Set[str],
                  path_tf: str
                  ) -> Dict[int, Dict[int, int]]:
    """Get the term frequencies of a corpus.

    Args:
        corpus: A set of document indices.
        path_tf: The path to the term frequencies per document of the
            base corpus.
    Return:
        {doc_id: {term_id: frequency}}
    """
    tf_corpus = defaultdict(dict)
    with open(path_tf, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i in corpus:
                tf_corpus[i] = json.load(line.strip('\n'))
    return tf_corpus


def get_df_corpus(term_ids: Set[str],
                  corpus: Set[str],
                  path_df: str
                  ) -> DefaultDict[int, int]:
    """Get the document frequencies of a corpus.

    Args:
        term_ids: A set of indices of terms.
        corpus: A set of document indices.
        path_df: The path to the document frequencies per document of
        the base corpus. This is a json file of the form:
        {term_id: Set of document indices}
        This means that |Set of document indices| is the df of term_id.
    Return:
        {term_id: frequency}
    """
    with open(path_df, 'r', encoding='utf8') as f:
        df_total = json.load(f)
    df_corpus = defaultdict(int)
    for term_id in term_ids:
        for doc_id in df_total:
            if doc_id in corpus:
                df_corpus[term_id] += 1
    return df_corpus


def get_tfidf(term_ids: Set[str],
              corpus: Set[str],
              path_tf: str,
              path_df: str,
              key: str = 'doc',
              ) -> DefaultDict[int, Dict[str, float]]:
    """Compute the tfidf score for the given terms in the given corpus.

    Args:
        term_ids: The lemma-ids of the lemmas for which tfidf is computed.
        corpus: The document-ids of the documents which make up the
            corpus.
        path_tf: Path to the frequencies of terms per document.
        path_df: Path to json-file of the form:
        {term_id: [doc term appears in]}
        key: either 'term' or 'doc'.
            If 'term', then the tfidf is returned in the
            following structure: Dict[term_id, Dict[term_id, tfidf]
            If 'doc', then the tfidf is returned in the following
            structure: Dict[doc_id, Dict[term_id, tfidf]]
            NOTE: at the moment the method always returns as if key='doc'
    """
    df = get_df_corpus(term_ids, corpus, path_df)  # {term_id: doc-freq}
    tf = get_tf_corpus(corpus, path_tf)  # {doc_id: term_id: frequency}
    n = len(tf)  # number of documents

    tfidf = defaultdict(dict)
    for doc_id in tf:
        tf_doc = tf[doc_id]
        for term_id in tf_doc:
            tf_word = tf_doc[term_id]
            df_word = df[term_id]
            tfidf[doc_id][term_id] = tf_word*log(n/df_word)
    return tfidf
