from typing import *
from math import sqrt, exp, log
from collections import defaultdict
from numpy import mean

from scipy.spatial.distance import cosine

from corpus import *

"""Compute term-candidate scores."""
"""
- prepare pseudo docs for given terms using tfidf scores
- code formulas:
    - score(t, S_k) = sqrt(pop(t, S_k)*con(t, S_k)*hyp(t, S_k))
    - pop(t, S_k) = log(tf(t, D_k)+1)/log(tf(D_k))
        -> D_k = total number of tokens in D_k
    - con(t, S_k) = exp(rel(t, D_k))/(1+sum(exp(rel(t, D_1...D_n))))
        -> use gensim's bm25 scorer to calculate rel(t, D_k)
    - hyp(t, S_k) = sum(sim(t, projection(z1...zn)*score(z1...zn)))
        -> z1...zn: labels of the parent cluster
"""

# ----------------------------------------------------------------------
# type definitions

# Type of a cluster as a set of term-ids.
cluster_type = Set[str]
# Type of a corpus as a list of tuples (doc_id, doc) where the document
# is a list of sentences which is a list of words.
corpus_type = List[Tuple[int, List[List[str]]]]

# ----------------------------------------------------------------------


class Scorer:

    def __init__(self,
                 clusters: Dict[int, cluster_type],
                 subcorpora: Dict[int, Set[str]],
                 # parent_corpus: Set[int],
                 # path_base_corpus: str,
                 level: int
                 ) -> None:
        """Initialize a Scorer object. Precompute all term scores.

        Args:
            clusters: A list of clusters. Each cluster is a set of
                term-ids.
            subcorpora: Maps each cluster label to the relevant doc-ids.
        (Old Args:
            parent_corpus: The doc-ids of the document, that make up the
                parent corpus (the corpus, to which the cluster belongs).
            path_base_corpus: The path to the original corpus file.)
        """
        self.clusters = clusters
        self.subcorpora = subcorpora
        # self.parent_corpus = Corpus(parent_corpus, self.path_base_corpus)
        # self.parent_corpus.get_corpus_docs()
        # self.path_base_corpus = path_base_corpus
        # self.topic_corpus = Corpus(self.cluster, self.path_base_corpus)
        # self.topic_corpus.get_corpus_docs()
        self.level = level

        # precompute term frequencies for all given terms
        # self.tf_topic_corpus = self.get_tf(cluster, self.topic_corpus.docs)
        # self.tf_parent_corpus = self.get_tf(cluster, self.parent_corpus.docs)

    def get_term_scores(self,
                        # df_base: Dict[str, int],
                        tf: Dict[str, Dict[str, int]],
                        tf_base: Dict[str, Dict[str, int]],
                        dl: Dict[str, Union[int, float]]
                        ) -> Dict[str, Tuple[float, float]]:
        """For all terms, compute and get popularity and concentration.

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
            tf_base:
            dl: The document lengts. Form: {doc-id: length}

        Return:
            A dictionary mapping each term-id a tuple containing the
            terms popularity and concentration. Form:
            {term-id: (popularity, concentration)}
        (Old Args:
            df_base: The document frequencies of the terms in parent-corpus.
                Form: {term-id: frequency})
        """
        pop_scores = self.get_pop_scores(tf, dl)  # {term-id: popularity}
        con_scores = self.get_con_scores(tf, tf_base, dl)
        # {term-id:concentration}

        term_scores = {}
        for term_id in pop_scores:
            pop = pop_scores[term_id]
            con = con_scores[term_id]
            term_scores[term_id] = (pop, con)

        return term_scores

    def get_pop_scores(self,
                       tf: Dict[str, Dict[str, int]],
                       dl: Dict[str, Union[int, float]]
                       ) -> Dict[str, float]:
        """Get the popularity scores for all terms in clusters.

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
            dl: The document lengths. Form: {doc-id: length}

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: (popularity, concentration)}
        """
        pop_scores = {}
        for label, clus in self.clusters.items():
            subcorp = self.subcorpora[label]
            num_tokens = sum([dl[doc_id] for doc_id in subcorp])
            for term_id in clus:
                num_occurences = sum([tf[doc_id][term_id] for doc_id in subcorp
                                      if doc_id in tf
                                      and term_id in tf[doc_id]])
                pop_score = (log(num_occurences+1) + 1) / (log(num_tokens+1)+1)
                pop_scores[term_id] = pop_score
        return pop_scores

    def get_con_scores(self,
                       # df_base: Dict[str, List[str]],
                       tf: Dict[str, Dict[str, int]],
                       tf_base: Dict[str, Dict[str, int]],
                       dl: Dict[str, Union[int, float]]
                       ) -> Dict[str, float]:
        """Get the concentration scores for all terms in clusters.

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
            tf_base:
            dl: The document lengts. Form: {doc-id: length}

        Return:
            A dictionary mapping each term-id the terms concentration.
            Form: {term-id: concentration}
        (Old Args:
            df_base: The document frequencies of the terms in parent-
                corpus. Form: {term-id: frequency})
        """
        bm25_scores = self.get_bm25_scores(tf, tf_base, dl)
        # {term-id: {label: bm25-score}}
        bm25_scores_sum = self.sum_bm25_scores(bm25_scores)
        # {term-id: sum_bm_25_scores}

        con_scores = {}  # {term_id: concentration}
        for label, clus in self.clusters.items():
            for term_id in clus:
                numerator = exp(bm25_scores[term_id][label])
                denominator = 1+exp(bm25_scores_sum[term_id])
                con_score = numerator/denominator
                con_scores[term_id] = con_score
        return con_scores

    def get_bm25_scores(self,
                        tf: Dict[str, Dict[str, int]],
                        tf_base: Dict[str, Dict[str, int]],
                        dl: Dict[str, Union[int, float]]
                        ) -> Dict[str, Dict[int, float]]:
        """Get the bm25 scores for all terms in clusters.

        The score is calculated as described in:
        https://en.wikipedia.org/wiki/Okapi_BM25

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
            tf_base:
            dl: The document lengts. Form: {doc-id: length}

        Return:
            A mapping of term's ids to their bm25 score in the form:
            {term-id: {label: bm25-score}}
        """
        bm25_scores = defaultdict(dict)
        idf = self.get_idf_scores(tf)  # {term-id: idf}
        k1 = 1.6
        b = 0.75
        len_pseudo_docs = self.get_len_pseudo_docs(dl)  # {label: len}
        # tf = self.get_tf(tf)  # {term-id: {pseudo_doc-id/label: tf}}
        avgdl = mean(list(len_pseudo_docs.values()))
        for label, clus in self.clusters.items():
            pseudo_doc = self.subcorpora[label]
            tf_pseudo_doc = self.get_tf_pseudo_doc(pseudo_doc, tf_base)
            # {term-id: tf}
            len_pseudo_doc = len_pseudo_docs[label]
            for term_id in clus:
                tf_td = tf_pseudo_doc[term_id]  # tf_term_doc
                numerator = idf[term_id]*(tf_td*(k1+1))
                denominator = (tf_td+k1*(1-b+b*(len_pseudo_doc/avgdl)))
                bm25_score = numerator/denominator
                bm25_scores[term_id][label] = bm25_score
        return bm25_scores

    def get_idf_scores(self,
                       tf: Dict[str, Dict[str, int]]
                       ) -> Dict[str, float]:
        """Get idf scores for terms in clusters and pseudo docs.

        idf(t, D) = log(N/|d e D: t e d|)
        There are 5 pseudo-docs, so N = |D| = 5.

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
        Return:
            {term_id: idf}
        """
        pd_bof = self.form_pseudo_docs_bag_of_words(tf)
        # {label: set(term-ids)}
        idf = defaultdict(float)
        for label, clus in self.clusters.items():
            for term_id in clus:
                df_t = len([1 for label in pd_bof if term_id in pd_bof[label]])
                idf[term_id] = log(5/(df_t+1))
        return idf

    def form_pseudo_docs_bag_of_words(self,
                                      tf: Dict[str, Dict[str, int]]
                                      ) -> Dict[int, Set[str]]:
        """Form bag of words pseudo docs.

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
        Return:
            {label: Bag of document words}
        """
        pd_bof = {}  # pseudo-docs-bag-of-words
        for label, sc in self.subcorpora.items():
            pd_bof[label] = set()
            for doc_id in sc:
                if doc_id in tf:
                    s = set([term_id for term_id in tf[doc_id]])
                    pd_bof[label].union(s)
        return pd_bof

    @staticmethod
    def sum_bm25_scores(bm25_scores: Dict[str, Dict[int, float]]
                        ) -> Dict[str, float]:
        """Get the summed bm25-score for all terms over subcorpora.

        Args:
            bm25_scores: The bm25-scores of the terms.
        Return:
            {term-id: sum_bm_25_scores}
        """
        bm25_scores_sum = {}
        for term_id in bm25_scores:
            bm25_scores_sum[term_id] = sum(bm25_scores[term_id].values())
        return bm25_scores_sum

    def get_len_pseudo_docs(self,
                            dl: Dict[str, Union[int, float]]
                            ) -> Dict[int, int]:
        """Get the length of pseudo-docs subcorpora.

        Return:
            {label: length}
        """
        len_pseudo_docs = {}
        for label, sc in self.subcorpora.items():
            len_pseudo_docs[label] = sum(dl[doc_id] for doc_id in sc)
        return len_pseudo_docs

    @staticmethod
    def get_tf_pseudo_doc(pseudo_doc: Set[str],
                          tf_base: Dict[str, Dict[str, int]]
                          )-> Dict[str, int]:
        """Get term frequencies for the given pseudo-document.

        Args:
            pseudo_doc: A set of document ids.
            tf_base: The term frequencies of the terms in the base-corpus.
                Form: {doc-id: {term-id: frequency}}
        Return:
            {term_id: frequency}
        """
        tf_pseudo_doc = defaultdict(int)
        for doc_id in pseudo_doc:
            for term_id in tf_base[doc_id]:
                tf_pseudo_doc[term_id] += tf_base[doc_id][term_id]
        return tf_pseudo_doc

    # def get_tf(self,
    #            cluster: cluster_type,
    #            corpus: corpus_type
    #            ) -> Dict[str, int]:
    #     """Get term frequency of terms in the cluster for the corpus.
    #
    #     Args:
    #         cluster: A set of term-ids.
    #         corpus: A list of document-ids and their documents.
    #     Return:
    #         A dictionary mapping each term-id in the cluster to it's
    #         frequency in the corpus.
    #     """
    #     tf_dict = defaultdict(int)
    #     for id_, doc in corpus:
    #         for sent in doc:
    #             for word in sent:
    #                 for term_id in cluster:
    #                     if term_id == word:
    #                         tf_dict[term_id] += 1
    #     return tf_dict

    def calc_term_score(self,
                        term_id: str,
                        cluster_id: int,
                        top_level=False
                        ) -> float:
        if top_level:
            score = self.calc_top_level_score(term_id, cluster_id)
        else:
            score = self.calc_gen_score(term_id, cluster_id)

        return score

    def calc_top_level_score(self, term_id: str, cluster_id: int) -> float:
        """Calculate the term score on the top level of the taxonomy.

        Args:
            term_id: The id of the term for which the score is
                calculated.
            cluster_id: The id of the cluster for which the term score
                is calculated.
        Return:
            The score.
        """
        pop = self.get_pop(term_id)
        con = self.get_con(term_id, cluster_id)
        return sqrt(pop*con)

    def calc_gen_score(self, term_id: str, cluster_id: int) -> float:
        """Calculate the term score in the general case (not top level).

        The term score is calculated as following:
        score(t, D_k) = sqrt(pop(t, D_k)*con(t, D_k)*hyp(t, D_k))

        Args:
            term_id: The id of the term for which the score is
                calculated.
            cluster_id: The id of the cluster for which the term score
                is calculated.
        Return:
            The score.
        """
        pop = self.get_pop(term_id)
        con = self.get_con(term_id, cluster_id)
        hyp = self.get_hyp(term_id, cluster_id)
        return sqrt(pop*con*hyp)

    def get_pop(self, term_id: str) -> float:
        """Get the popularity of the term in the cluster corpus.

        For the set of documents belonging to the cluster-id, calculate:
        pop(t, S_k) = log(tf(t, D_k)+1)/log(tf(D_k))
        where tf(D_k) denotes the total number of tokens in the cluster
        corpus D_k and tf(t, D_k) denotes the term frequency of term t
        in the cluster corpus D_k.

        Args:
            term_id: The id of the term to be scored.
            cluster_id: The id of the cluster for which the term is
            to be scored.
        Return:
            The popularity score.
        """
        numerator = log(self.tf_topic_corpus[term_id]+1)
        denominator = log(self.tf_parent_corpus[term_id])
        return numerator/denominator

    def get_con(self, term_id: str, cluster_id: int) -> float:
        """Get the concentration of the term in the cluster corpus.

        For the set of documents belonging to the cluster-id calculate:
        con(t, S_k) = exp(rel(t, D_k))/(1+sum(exp(rel(t, D_1...D_n))))
        where rel(t, D_k) denotes the bm25-relevancy of term t for the
        cluster corpus D_k.
        """
        numerator = exp(self.get_bm25(term_id, cluster_id))
        rel_scores = [exp(self.get_bm25(term_id, id_))
                      for id_ in self.cluster_ids]
        denominator = 1+sum(rel_scores)
        return numerator/denominator

    def get_hyp(self, term_id: str, cluster_id: int) -> float:
        """Get the hyponym score of the term for the cluster.

        Given the labels of the parent/hypernym-topics calculate how
        close the term is to the hyponym projection of the hypernyms.
        The closeness projection is weighted with the score of the
        corresponding hypernym.
        """
        parent_clus_id = self.get_parent_cluster(cluster_id)
        parent_label_ids, parent_label_scores = self.get_labels(parent_clus_id)
        projections = [self.get_projection(idx) for idx in parent_label_ids]
        term_vec = self.get_vec(term_id)
        similarities = [self.get_sim(pj, term_vec) for pj in projections]
        sim_scores = []
        for i in range(len(similarities)):
            sim_scores.append(similarities[i]*parent_label_scores[i])
        return sum(sim_scores)

    @staticmethod
    def get_sim(v1: Iterator[float], v2: Iterator[float]) -> float:
        """Calcualte the consine similarity between vectors v1 and v2.

        Args:
            v1: vector 1
            v2: vector 2
        Return:
            The cosine similarity.
        """
        return 1-cosine(v1, v2)

    # def get_term_scores(self) -> Dict[str, Tuple[float, float]]:
    #     """Get the term scores for all the terms in the given clusters.
    #
    #     Return:
    #         A dictionary mapping each term-id to a tuple of the form:
    #         (popularity, concentration)
    #     """
    #     pass
