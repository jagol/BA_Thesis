from typing import *
from math import sqrt, exp, log
from collections import defaultdict
import numpy as np
from utility_functions import get_config


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
cluster_type = Set[int]
cluster_centers_type = Dict[int, List[float]]
# {doc-id: {word-id: (term-freq, tfidf)}} doc-length is at word-id -1
term_distr_type = DefaultDict[int, DefaultDict[int, Union[Tuple[int, int],
                                                          int]]]

# ----------------------------------------------------------------------


class Scorer:

    def __init__(self,
                 clusters: Dict[int, cluster_type],
                 cluster_centers: Dict[int, List[float]],
                 subcorpora: Dict[int, Set[int]],
                 level: int
                 ) -> None:
        """Initialize a Scorer object.

        Args:
            clusters: A list of clusters. Each cluster is a set of
                term-ids.
            cluster_centers: A dict mapping the cluster-id to it's
                cluster-center.
            subcorpora: Maps each cluster label to the relevant doc-ids.
            level: Level in the taxonomy. Root is level 0.
        """
        self.clusters = clusters
        self.clusters_inv = self.inverse_cluster(clusters)
        self.cluster_centers = cluster_centers
        self.subcorpora = subcorpora
        self.level = level
        self.config = get_config()
        self.pop_df_version = self.config['pop_df_version']
        self.pop_sum_version = self.config['pop_sum_version']

    @staticmethod
    def inverse_cluster(clusters: Dict[int, cluster_type]
                        ) -> Dict[int, int]:
        """Inverse a cluster.

        Return:
            {term-id: clus_label}
        """
        inv_clusters = {}
        for label, clus in clusters.items():
            for term_id in clus:
                inv_clusters[term_id] = label
        return inv_clusters

    def get_term_scores(self,
                        term_distr: term_distr_type,
                        df: Dict[int, List[int]],
                        ) -> Dict[int, Tuple[float, float, float]]:
        """For all terms, compute and get popularity and concentration.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id a tuple containing the
            terms popularity and concentration. Form:
            {term-id: (popularity, concentration)}
        """
        print('  Get popularity scores...')
        if self.pop_df_version:
            if self.pop_sum_version:
                pop_scores = self.get_pop_scores_df_sum(df)
            else:
                pop_scores = self.get_pop_scores_df(df)
        else:
            if self.pop_sum_version:
                pop_scores = self.get_pop_scores_sum(term_distr, df)
            else:
                pop_scores = self.get_pop_scores(term_distr, df)

        print('  Get concentration scores...')
        con_scores = self.get_con_scores(term_distr)

        print('  Get repr. scores...')
        term_scores = {}
        for term_id in pop_scores:
            pop = pop_scores[term_id]
            con = con_scores[term_id]
            total = sqrt(pop*con)
            term_scores[term_id] = (pop, con, total)

        return term_scores

    def get_pop_scores(self,
                       term_distr: term_distr_type,
                       df: Dict[int, List[int]]
                       ) -> Dict[int, float]:
        """Get the popularity scores for all terms in clusters.

        For a given cluster (init) and given subcorpora for each cluster
        (init) calculate the popularity.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: popularity}
        """
        pop_scores = {}
        for label, clus in self.clusters.items():
            subcorp = self.subcorpora[label]

            # Calc tf_Dk.
            tf_Dk = 0
            for doc_id in subcorp:
                tf_Dk += term_distr[doc_id][-1]

            # Calc tf_t_Dk.
            for term_id in clus:
                tf_t_Dk = 0
                for doc_id in df[term_id]:
                    if doc_id in subcorp:
                        tf_t_Dk += term_distr[doc_id][term_id][0]

                # Calc pop-score.
                pop_score = log(tf_t_Dk + 1) / log(tf_Dk)
                pop_scores[term_id] = pop_score

        return pop_scores

    def get_pop_scores_df(self,
                          df: Dict[int, List[int]]
                          ) -> Dict[int, float]:
        """Get the df-popularity scores for all terms in clusters.

        This Method computes the popularity scores using the document
        frequency instead of the term frequency.

        Args:
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: popularity}
        """
        pop_scores = {}
        for label, clus in self.clusters.items():
            subcorp = self.subcorpora[label]

            # Calc tf_Dk.
            df_Dk = 0
            for term_id in df:
                df_Dk += len(df[term_id])

            # Calc tf_t_Dk.
            for term_id in clus:
                df_t_Dk = 0
                for doc_id in df[term_id]:
                    if doc_id in subcorp:
                        df_t_Dk += 1

                # Calc pop-score.
                pop_score = log(df_t_Dk + 1) / log(df_Dk)
                pop_scores[term_id] = pop_score

        return pop_scores

    def get_pop_scores_sum(self,
                           term_distr: term_distr_type,
                           df: Dict[int, List[int]]
                           ) -> Dict[int, float]:
        """Get the popularity scores for all terms in clusters.

        For a given cluster (init) and given subcorpora for each cluster
        (init) calculate the popularity.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: popularity}
        """
        pop_scores_raw = {}  # {term_id: [pop1, pop2, pop3, pop4, pop5]}
        for term_id in self.clusters_inv:
            tf_t_Dk_clus = np.zeros(len(self.clusters))
            # [pop1, pop2, pop3, pop4, pop5]
            for label in self.clusters:
                subcorp = self.subcorpora[label]
                for doc_id in df[term_id]:
                    if doc_id in subcorp:
                        tf_t_Dk_clus[label] += term_distr[doc_id][term_id][0]

                # Calc pop-score.
                pop_score = np.log2(tf_t_Dk_clus + 1)  # / log(tf_Dk)
                pop_scores_raw[term_id] = pop_score

        pop_scores = {}
        for term_id in pop_scores_raw:
            label = self.clusters_inv[term_id]
            scores = pop_scores_raw[term_id]
            pop_scores[term_id] = scores[label] / sum(scores)

        return pop_scores

    def get_pop_scores_df_sum(self,
                              df: Dict[int, List[int]]
                              ) -> Dict[int, float]:
        """Get the popularity scores for all terms in clusters.

        For a given cluster (init) and given subcorpora for each cluster
        (init) calculate the popularity.

        Args:
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: popularity}
        """
        pop_scores_raw = {}  # {term_id: [pop1, pop2, pop3, pop4, pop5]}
        for term_id in self.clusters_inv:
            df_t_Dk_clus = np.zeros(len(self.clusters))
            # [pop1, pop2, pop3, pop4, pop5]
            for label in self.clusters:
                subcorp = self.subcorpora[label]
                for doc_id in df[term_id]:
                    if doc_id in subcorp:
                        df_t_Dk_clus[label] += 1

                # Calc pop-score.
                pop_score = np.log2(df_t_Dk_clus + 1)  # / log(tf_Dk)
                pop_scores_raw[term_id] = pop_score

        pop_scores = {}
        for term_id in pop_scores_raw:
            label = self.clusters_inv[term_id]
            scores = pop_scores_raw[term_id]
            pop_scores[term_id] = scores[label] / sum(scores)

        return pop_scores

    def get_con_scores(self,
                       term_distr: term_distr_type,
                       ) -> Dict[int, float]:
        """Get the concentration scores for all terms in clusters.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.

        Return:
            A dictionary mapping each term-id the terms concentration.
            Form: {term-id: concentration}
        """
        bm25_scores = self.get_bm25_scores(term_distr)
        # {term-id: {label: bm25-score}}
        bm25_scores_sum = self.sum_bm25_scores(bm25_scores)
        # {term-id: sum_bm_25_scores}

        con_scores = {}  # {term_id: concentration}
        for label, clus in self.clusters.items():
            for term_id in clus:
                numerator = exp(bm25_scores[term_id][label])
                denominator = 1 + exp(bm25_scores_sum[term_id])
                con_score = numerator/denominator
                con_scores[term_id] = con_score
        return con_scores

    def get_bm25_scores(self,
                        term_distr: term_distr_type,
                        ) -> DefaultDict[int, Dict[int, float]]:
        """Get the bm25 scores for all terms in clusters.

        The score is calculated as described in:
        https://en.wikipedia.org/wiki/Okapi_BM25

        Args:
            term_distr: Description at the top of the file in
                type-definitions.

        Return:
            A mapping of term's ids to their bm25 score in the form:
            {term-id: {clus-label: bm25-score}}
        """
        bm25_scores = defaultdict(dict)
        k1 = 1.2
        b = 0.5
        multiplier = 3
        df_scores_pd = self.get_df_scores_pd(term_distr)
        # {term-id: df_pd}
        idf_scores_pd = self.get_idf_scores_pd(df_scores_pd)
        # {term-id: idf_pd}
        len_pds = self.get_len_pseudo_docs(term_distr)
        # {label: len_pd}
        tf_scores_pd = self.get_tf_scores_pd(term_distr)
        # {term-id: [tf_pd0, ...tf_pd4]}
        avgdl = np.mean(list(len_pds.values()))
        max_df = max(df_scores_pd.values())
        # max_df is the max number of pseudo-docs a term appears in.

        for term_id in self.clusters_inv:
            for label in self.clusters:
                tf = tf_scores_pd[term_id][label]
                df = df_scores_pd[term_id]
                idf = idf_scores_pd[term_id]
                df_factor = log(1 + df, 2) / log(1 + max_df, 2)
                len_pd = len_pds[label]
                bm25 = idf*(tf*(k1 + 1)) / (tf + k1*(1 - b + b*(len_pd/avgdl)))
                bm25 *= df_factor
                bm25 *= multiplier
                bm25_scores[term_id][label] = bm25

        return bm25_scores

    def get_df_scores_pd(self,
                         term_distr: term_distr_type
                         ) -> Dict[int, int]:
        """Get the documents frequencies for the terms in pseudo-docs.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
        Return:
            The document frequencies for terms in pseudo-docs.
            {term-id: df}
        """
        df_scores_pd = {}
        pd_bow = self.form_pseudo_docs_bag_of_words(term_distr)
        for term_id in self.clusters_inv:
            count = 0
            for label in self.clusters:
                if term_id in pd_bow[label]:
                    count += 1
            df_scores_pd[term_id] = count
        return df_scores_pd

    def get_idf_scores_pd(self,
                          df_scores_pd: Dict[int, int]
                          ) -> Dict[str, float]:
        """Get idf scores for terms in clusters and pseudo docs.

        idf(t, D) = log(N/|d e D: t e d|)
        There are 5 pseudo-docs, so N = |D| = 5.

        Args:
            df_scores_pd: {term_id: pseudo-document frequency}
        Return:
            {term_id: idf_score}
        """
        idf = defaultdict(float)
        for term_id in df_scores_pd:
            df = df_scores_pd[term_id]
            idf[term_id] = log(len(self.clusters)/(df+1))
        return idf

    def form_pseudo_docs_bag_of_words(self,
                                      term_distr: term_distr_type
                                      ) -> Dict[int, Set[int]]:
        """Form bag of words pseudo docs.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
        Return:
            {label: Bag of document words}
        """
        pd_bof = {}  # pseudo-docs-bag-of-words
        for label, sc in self.subcorpora.items():
            vocab = []
            for doc_id in sc:
                s = [t_id for t_id in term_distr[doc_id] if t_id != -1]
                vocab.extend(s)
            pd_bof[label] = set(vocab)
        return pd_bof

    @staticmethod
    def sum_bm25_scores(bm25_scores: Dict[int, Dict[int, float]]
                        ) -> Dict[int, float]:
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
                            term_distr: term_distr_type
                            ) -> Dict[int, int]:
        """Get the length of pseudo-docs subcorpora.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.

        Return:
            {label: length}
        """
        len_pseudo_docs = {}
        for label, sc in self.subcorpora.items():
            len_pseudo_doc = sum(term_distr[doc_id][-1] for doc_id in sc)
            len_pseudo_docs[label] = len_pseudo_doc
        return len_pseudo_docs

    def get_tf_scores_pd(self,
                         term_distr: term_distr_type
                         ) -> Dict[int, Dict[int, int]]:
        """Get the term frequencies for the pseudo docs/subcorpora.

        Get: how often terms appears in given pseudo-doc-
        Needed: doc-ids of pseudo-doc, set of terms, term-frequency per doc
        Which terms? All term in one of the current clusters.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
        Return:
            {term_id: {label: tf}}
        """
        tf_pd = {}
        for label, sc in self.subcorpora.items():
            for doc_id in sc:
                for term_id in term_distr[doc_id]:
                    if term_id in self.clusters_inv:
                        if term_id not in tf_pd:
                            tf_pd[term_id] = {0: 0, 1: 0, 2: 0,
                                              3: 0, 4: 0}
                        tf_pd[term_id][label] += term_distr[doc_id][term_id][0]
        return tf_pd
