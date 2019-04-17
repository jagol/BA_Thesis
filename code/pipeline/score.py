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
        self.kl_divergence = self.config['kl_divergence']

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

        print('  Compute concentration scores...')
        con_scores = self.get_con_scores(term_distr)

        print('  Compute repr. scores...')
        term_scores = {}
        for term_id in pop_scores:
            pop = pop_scores[term_id]
            con = con_scores[term_id]
            total = np.array([sqrt(pop[i]*con[i]) for i in range(len(pop))])
            term_scores[term_id] = (pop, con, total)

        print('  Compute the KL-Divergence...')
        if self.kl_divergence:
            term_scores = self.get_kl_divergence(term_scores)
        else:
            term_scores = self.get_one_score(term_scores)

        return term_scores

    # def get_pop_scores(self,  # original pop_scores-method
    #                    term_distr: term_distr_type,
    #                    df: Dict[int, List[int]]
    #                    ) -> Dict[int, np.ndarray[float]]:
    #     """Get the popularity scores for all terms in clusters.
    #
    #     For a given cluster (init) and given subcorpora for each cluster
    #     (init) calculate the popularity.
    #
    #     Args:
    #         term_distr: Description at the top of the file in
    #             type-definitions.
    #         df: The document frequencies of terms in the base corpus.
    #
    #     Return:
    #         A dictionary mapping each term-id the terms popularity.
    #         Form: {term-id: array of popularity per cluster}
    #     """
    #     pop_scores = {}
    #     for label, clus in self.clusters.items():
    #         subcorp = self.subcorpora[label]
    #
    #         # Calc tf_Dk.
    #         tf_Dk = 0
    #         for doc_id in subcorp:
    #             tf_Dk += term_distr[doc_id][-1]
    #
    #         # Calc tf_t_Dk.
    #         for term_id in clus:
    #             tf_t_Dk = 0
    #             for doc_id in df[term_id]:
    #                 if doc_id in subcorp:
    #                     tf_t_Dk += term_distr[doc_id][term_id][0]
    #
    #             # Calc pop-score.
    #             pop_score = log(tf_t_Dk + 1) / log(tf_Dk)
    #             pop_scores[term_id] = pop_score
    #
    #     return pop_scores

    def get_pop_scores(self,
                       term_distr: term_distr_type,
                       df: Dict[int, List[int]]
                       ) -> Dict[int, np.ndarray]:
        """Get the popularity scores for all terms in clusters.

        For a given cluster (init) and given subcorpora for each cluster
        (init) calculate the popularity.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: array of popularity per cluster}
        """
        pop_scores_raw = {}  # {term_id: [pop1, pop2, pop3, pop4, pop5]}
        tf_Dk_clus = {}  # {label: total number of tokens in Dk}

        # Calc tf_Dk.
        for label in self.clusters:
            subcorp = self.subcorpora[label]
            for doc_id in subcorp:
                tf_Dk_clus[label] += term_distr[doc_id][-1]

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

            tf_Dk_clus[term_id] = tf_Dk_clus

        pop_scores = {}
        for term_id in pop_scores_raw:
            scores = pop_scores_raw[term_id]
            # Divide each raw-pop-score by the total number of
            # occurrences of tokens in Dk (tf_Dk).
            # Equivalent to l1-norm.
            pop_scores[term_id] = np.array([scores[label] / tf_Dk_clus[label]
                                            for label in self.clusters])

        return pop_scores

    def get_pop_scores_df(self,
                          df: Dict[int, List[int]]
                          ) -> Dict[int, np.ndarray]:
        """Get the df-popularity scores for all terms in clusters.

        This Method computes the popularity scores using the document
        frequency instead of the term frequency.

        Args:
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: array of popularity per cluster}
        """
        pop_scores_raw = {}  # {term_id: [pop1, pop2, pop3, pop4, pop5]}
        df_Dk_clus = {}  # {label: total number of documents in Dk}

        # Calc tf_Dk.
        for label in self.clusters:
            subcorp = self.subcorpora[label]
            df_Dk_clus[label] = len(subcorp)

        for term_id in self.clusters_inv:
            df_t_Dk_clus = np.zeros(len(self.clusters))
            # [pop1, pop2, pop3, pop4, pop5]

            for label in self.clusters:
                subcorp = self.subcorpora[label]

                for doc_id in df[term_id]:
                    if doc_id in subcorp:
                        df_t_Dk_clus[label] += 1

                # Calc pop_score_raw.
                pop_scores_raw[term_id] = np.log2(df_t_Dk_clus + 1)

        pop_scores = {}
        for term_id in pop_scores_raw:
            scores = pop_scores_raw[term_id]
            # Divide each raw-pop-score by the total number of
            # occurrences of tokens in Dk (tf_Dk).
            # Equivalent to l1-norm.
            pop_scores[term_id] = np.array([scores[label] / df_Dk_clus[label]
                                            for label in self.clusters])

        return pop_scores

    def get_pop_scores_sum(self,
                           term_distr: term_distr_type,
                           df: Dict[int, List[int]]
                           ) -> Dict[int, np.ndarray]:
        """Get the popularity scores for all terms in clusters.

        This is a sum-version of the method: instead of counting normalizing
        using the total number of tokens in Dk the sum of term occurrences
        in all subcorpora is used.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: array of popularity per cluster}
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
            scores = pop_scores_raw[term_id]
            pop_scores[term_id] = scores / sum(scores)

        return pop_scores

    def get_pop_scores_df_sum(self,
                              df: Dict[int, List[int]]
                              ) -> Dict[int, np.ndarray]:
        """Get the popularity scores for all terms in clusters.

        This method is a combination of the sum-version and the
        df-version of the get_pop_scores_method.

        Args:
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms popularity.
            Form: {term-id: array of popularity per cluster}
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
            scores = pop_scores_raw[term_id]
            pop_scores[term_id] = scores / sum(scores)

        return pop_scores

    def get_con_scores(self,
                       term_distr: term_distr_type,
                       ) -> Dict[int, np.ndarray]:
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

        num_clus = len(self.clusters)
        con_scores = {tid: np.zeros(num_clus) for tid in self.clusters_inv}
        # {term_id: array of concentration scores}

        for term_id in self.clusters_inv:
            for label, clus in self.clusters.items():
                numerator = exp(bm25_scores[term_id][label])
                denominator = 1 + exp(bm25_scores_sum[term_id])
                con_score = numerator/denominator
                con_scores[term_id][label] = con_score
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

    def get_kl_divergence(self,
                          term_scores: Dict[int, Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray]]
                          ) -> Dict[int, Tuple[float, float, float]]:
        """Compute the kl-divergence.

        Written after
        https://github.com/franticnerd/taxogen/blob/master/code/utils.py#L18
        Code is changed to suit into this environment

        Args:
            term_scores: Maps the term-id to an array with the
                term-score for a cluster l at place array[l].
        Return:
            term_scores_clus: Maps the term-id to the term-score of the
                cluster the term belongs to.
        """
        term_scores_clus = {}  # {term-id: score}
        num_clus = len(self.clusters)
        ed = [1.0 / num_clus] * num_clus  # expected distribution
        for term_id in term_scores:
            label = self.clusters_inv[term_id]
            pop = term_scores[term_id][0]
            con = term_scores[term_id][1]
            ad = term_scores[term_id][2]  # actual distribution

            if len(ad) != len(ed):
                print('KL divergence error: p, q have different length')
            c_entropy = 0
            for i in range(len(ad)):
                if ad[i] > 0:
                    c_entropy += ad[i] * log(float(ad[i]) / ed[i])

            term_scores_clus[term_id] = (pop[label], con[label], c_entropy)

        return term_scores_clus

    def get_one_score(self,
                      term_scores: Dict[int, Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]]
                      ) -> Dict[int, Tuple[float, float, float]]:
        """Get the one term score for the cluster the term belongs to.

        Args:
            term_scores: Maps the term-id to an array with the
                term-score for a cluster l at place array[l].
        Return:
            term_scores_clus: Maps the term-id to the term-score of the
                cluster the term belongs to.
        """
        term_scores_clus = {}
        for term_id in term_scores:
            label = self.clusters_inv[term_id]
            pop, con, total = term_scores[term_id]
            term_scores_clus[term_id] = (pop[label], con[label], total[label])
        return term_scores_clus
