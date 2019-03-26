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
# {doc-id: {word-id: (term-freq, tfidf)}} doc-length is at word-id -1
word_distr_type = DefaultDict[int, DefaultDict[int, Union[Tuple[int, int], int]]]

# ----------------------------------------------------------------------


class Scorer:

    def __init__(self,
                 clusters: Dict[int, cluster_type],
                 cluster_centers: Dict[int, List[float]],
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
        self.cluster_centers = cluster_centers
        self.subcorpora = subcorpora
        self.level = level

    def get_term_scores(self,
                        word_distr: word_distr_type,
                        df: Dict[int, List[int]]
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
        """
        print('  Get popularity scores...')
        pop_scores = self.get_pop_scores(word_distr, df)
        print('  Get concentration scores...')
        con_scores = self.get_con_scores(word_distr)

        term_scores = {}
        for term_id in pop_scores:
            pop = pop_scores[term_id]
            con = con_scores[term_id]
            term_scores[term_id] = (pop, con)

        return term_scores

    def get_pop_scores(self,
                       word_distr: Dict[int, Dict[int, Union[Tuple[int, int], int]]],
                       df: Dict[int, List[int]]
                       ) -> Dict[str, float]:
        """Get the popularity scores for all terms in clusters.

        For a given cluster (init) and given subcorpora for each cluster
        (init) calculate the popularity.

        for label, clus in self.clusters.items():
            subcorp = self.subcorpora[label]
            num_tokens = sum([word_distr[doc_id][-1] for doc_id in subcorp])
            for term_id in clus:
                num_occurences = sum([word_distr[doc_id][term_id] for doc_id in subcorp if term_id in word_distr[doc_id])

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

            # Calc tf_Dk.
            tf_Dk = 0
            for doc_id in subcorp:
                tf_Dk += word_distr[doc_id][-1]

            # Calc tf_t_Dk.
            for term_id in clus:
                tf_t_Dk = 0
                for doc_id in df[term_id]:
                    tf_t_Dk += word_distr[doc_id][term_id][0]

                # Calc pop-score.
                pop_score = log(tf_t_Dk + 1) / log(tf_Dk)
                pop_scores[term_id] = pop_score

        return pop_scores

    def get_con_scores(self,
                       word_distr: word_distr_type
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
        bm25_scores = self.get_bm25_scores(word_distr)
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
                        word_distr: word_distr_type
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
        idf = self.get_idf_scores(word_distr)  # {term-id: idf}
        k1 = 1.2
        b = 0.5
        multiplier = 3
        len_pseudo_docs = self.get_len_pseudo_docs(word_distr)  # {label: len}
        avgdl = mean(list(len_pseudo_docs.values()))

        for label, clus in self.clusters.items():
            pseudo_doc = self.subcorpora[label]
            tf_pseudo_doc = self.get_tf_pseudo_doc(pseudo_doc, clus, word_distr)
            # {term-id: tf}
            len_pseudo_doc = len_pseudo_docs[label]

            for term_id in clus:
                tf_td = tf_pseudo_doc[term_id]  # tf_term_doc
                numerator = idf[term_id]*(tf_td*(k1+1))
                denominator = (tf_td+k1*(1-b+b*(len_pseudo_doc/avgdl)))
                bm25_score = numerator/denominator
                bm25_scores[term_id][label] = bm25_score*multiplier

        return bm25_scores

    def get_idf_scores(self,
                       # tf: Dict[str, Dict[str, int]]
                       word_distr: word_distr_type
                       ) -> Dict[str, float]:
        """Get idf scores for terms in clusters and pseudo docs.

        idf(t, D) = log(N/|d e D: t e d|)
        There are 5 pseudo-docs, so N = |D| = 5.

        Method:
        Form a bag of words representation for each pseudo-document ->
            {pseudo-doc-label: set of all term-ids in the pseudo-doc}
        Calc df by counting in how many pseudo-docs a term appears.
        Calc idf.

        Args:
            tf: The term frequencies of the terms in parent-corpus.
                Form: {doc-id: {term-id: frequency}}
        Return:
            {term_id: idf_score}
        """
        pd_bof = self.form_pseudo_docs_bag_of_words(word_distr)
        # {label: set(term-ids)}
        idf = defaultdict(float)
        for label, clus in self.clusters.items():
            for term_id in clus:
                df_t = len([1 for label in pd_bof if term_id in pd_bof[label]])
                idf[term_id] = log(5/(df_t+1))
        return idf

    def form_pseudo_docs_bag_of_words(self,
                                      word_distr: word_distr_type
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
                s = set([term_id for term_id in word_distr[doc_id] if term_id != -1])
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
                            # dl: Dict[str, Union[int, float]]
                            word_distr: word_distr_type
                            ) -> Dict[int, int]:
        """Get the length of pseudo-docs subcorpora.

        Return:
            {label: length}
        """
        len_pseudo_docs = {}
        for label, sc in self.subcorpora.items():
            len_pseudo_docs[label] = sum(word_distr[doc_id][-1] for doc_id in sc)
        return len_pseudo_docs

    @staticmethod
    def get_tf_pseudo_doc(pseudo_doc: Set[str],
                          cluster: Set[str],
                          word_distr: word_distr_type
                          )-> Dict[str, int]:
        """Get term frequencies for the given pseudo-document.

        Get: how often terms appears in given pseudo-doc-
        Needed: doc-ids of pseudo-doc, set of terms, term-frequency per doc
        Which terms? All term in one of the current clusters.

        Args:
            pseudo_doc: A set of document ids.
            tf_base: The term frequencies of the terms in the base-corpus.
                Form: {doc-id: {term-id: frequency}}
        Return:
            {term_id: frequency}
        """
        tf_pseudo_doc = defaultdict(int)
        for doc_id in pseudo_doc:
            for term_id in word_distr[doc_id]:
                if term_id in cluster:
                    tf_pseudo_doc[term_id] += word_distr[doc_id][term_id][0]
        return tf_pseudo_doc

    def get_term_scores_efficient(self,
                                  word_distr: word_distr_type
                                  ) -> Dict[str, Tuple[float, float]]:
        """More efficient version of get_term_scores.

        Return:
            A dict mapping the term-id to concentration and popularity.
        """
        df = {}  # {term-id: list of doc-ids}
        term_scores = defaultdict(list)
        for label, clus in self.clusters.items():
            subcorp = self.subcorpora[label]
            tf_Dk = sum([word_distr[doc_id][-1] for doc_id in subcorp])
            for term_id in clus:
                tf_t_Dk = 0
                for doc_id in subcorp:
                    if term_id in word_distr[doc_id]:
                        tf_t_Dk += word_distr[doc_id][term_id][0]

                pop_score = log(tf_t_Dk + 1) / log(tf_Dk)
                term_scores[term_id].append(pop_score)

        return term_scores

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