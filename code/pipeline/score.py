from typing import *
from math import sqrt, exp, log
from collections import defaultdict
from numpy import mean
from scipy.spatial.distance import cosine
from corpus import *
import pdb

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
# Type of a corpus as a list of tuples (doc_id, doc) where the document
# is a list of sentences which is a list of words.
corpus_type = List[Tuple[int, List[List[str]]]]
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


    def inverse_cluster(self,
                        clusters: Dict[int, cluster_type]
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
                        df: Dict[int, List[int]]
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
        pop_scores = self.get_pop_scores(term_distr, df)
        print('  Get concentration scores...')
        con_scores = self.get_con_scores(term_distr, df)

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

    def get_con_scores(self,
                       term_distr: term_distr_type,
                       df: Dict[int, List[int]]
                       ) -> Dict[int, float]:
        """Get the concentration scores for all terms in clusters.

        Args:
            term_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

        Return:
            A dictionary mapping each term-id the terms concentration.
            Form: {term-id: concentration}
        """
        bm25_scores = self.get_bm25_scores(term_distr, df)
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

    # def get_bm25_scores(self,
    #                     word_distr: term_distr_type,
    #                     df: Dict[int, List[int]]
    #                     ) -> Dict[int, Dict[int, float]]:
    #     """Get the bm25 scores for all terms in clusters.
    #
    #     The score is calculated as described in:
    #     https://en.wikipedia.org/wiki/Okapi_BM25
    #
    #     Args:
    #         word_distr: Description at the top of the file in
    #             type-definitions.
    #         df: The document frequencies of terms in the base corpus.
    #
    #     Return:
    #         A mapping of term's ids to their bm25 score in the form:
    #         {term-id: {clus-label: bm25-score}}
    #     """
    #     print('  In get_bm25_scores...')
    #     bm25_scores = defaultdict(dict)
    #     print('  Calculate idf scores...')
    #     idf = self.get_idf_scores(word_distr)  # {term-id: idf}
    #     k1 = 1.2
    #     b = 0.5
    #     multiplier = 3
    #     print('  Calculate get_len_pseudo_docs...')
    #     len_pseudo_docs = self.get_len_pseudo_docs(word_distr)  # {label: len}
    #     print('  Calculate avgdl...')
    #     avgdl = mean(list(len_pseudo_docs.values()))
    #
    #     print('Rest of bm25 calculations...')
    #     for label, clus in self.clusters.items():
    #         print('  bm25 for label {}'.format(label))
    #         pseud_doc = self.subcorpora[label]
    #         tf_pseudo_doc = self.get_tf_pseudo_doc(pseud_doc, clus, word_distr)
    #         # {term-id: tf}
    #         df_local = self.get_df_clus_subcorp(pseud_doc, clus, df)
    #         max_df = max([len(v) for v in df_local.values()])
    #         len_pseudo_doc = len_pseudo_docs[label]
    #
    #         for term_id in self.clusters_inv:  # {term_id: clus_label}
    #             df_term = len(df_local[term_id])
    #             df_factor = log(1 + df_term, 2) / log(1 + max_df, 2)
    #             tf_td = tf_pseudo_doc[term_id]  # tf_term_doc
    #             numerator = idf[term_id]*(tf_td*(k1+1))
    #             denominator = (tf_td+k1*(1-b+b*(len_pseudo_doc/avgdl)))
    #             bm25_score = numerator/denominator
    #             bm25_score *= df_factor
    #             bm25_score *= multiplier
    #             bm25_scores[term_id][label] = bm25_score
    #     return bm25_scores

    def get_bm25_scores(self,
                        term_distr: term_distr_type,
                        df_scores: Dict[int, List[int]]
                        ) -> DefaultDict[int, Dict[int, float]]:
        """Get the bm25 scores for all terms in clusters.

        The score is calculated as described in:
        https://en.wikipedia.org/wiki/Okapi_BM25

        Args:
            word_distr: Description at the top of the file in
                type-definitions.
            df: The document frequencies of terms in the base corpus.

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
        avgdl = mean(list(len_pds.values()))
        max_df = max(df_scores_pd.values())

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
            df: The document frequencies of terms in the base corpus.
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

    # @staticmethod
    # def get_df_clus_subcorp(pseud_doc: Set[int],
    #                         clus: Set[int],
    #                         df: Dict[int, List[int]]
    #                         ) -> Dict[int, List[int]]:
    #     """Get the local document frequencies.
    #
    #     The local document frequencies only include terms in the current
    #     cluster and documents from the current subcorpus.
    #
    #     Args:
    #         pseud_doc: A set of document ids.
    #         clus: A set of term ids.
    #         df: The global document frequencies.
    #     Return:
    #         The local document frequencies of the form:
    #         {term-id: List of doc-ids the term appears in}
    #     """
    #     local_df = {}
    #     for term_id in clus:
    #         local_df[term_id] = []
    #         for doc_id in df[term_id]:
    #             if doc_id in pseud_doc:
    #                 local_df[term_id].append(doc_id)
    #     return local_df

    def get_idf_scores_pd(self,
                          df_scores_pd: Dict[int, int]
                          ) -> Dict[str, float]:
        """Get idf scores for terms in clusters and pseudo docs.

        idf(t, D) = log(N/|d e D: t e d|)
        There are 5 pseudo-docs, so N = |D| = 5.

        Args:
            word_distr: Description at the top of the file in
                type-definitions.
        Return:
            {term_id: idf_score}
        """
        idf = defaultdict(float)
        for term_id in df_scores_pd:
            df = df_scores_pd[term_id]
            idf[term_id] = log(5/(df+1))
            # TODO: N = 5 is hardcoded, make env-dependent
        return idf

    def form_pseudo_docs_bag_of_words(self,
                                      term_distr: term_distr_type
                                      ) -> Dict[int, Set[int]]:
        """Form bag of words pseudo docs.

        Args:
            word_distr: Description at the top of the file in
                type-definitions.
        Return:
            {label: Bag of document words}
        """
        pd_bof = {}  # pseudo-docs-bag-of-words
        for label, sc in self.subcorpora.items():
            print('      label: {}'.format(label))
            vocab = []
            for doc_id in sc:
                s = [term_id for term_id in term_distr[doc_id] if term_id !=-1]
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
            word_distr: Description at the top of the file in
                type-definitions.

        Return:
            {label: length}
        """
        len_pseudo_docs = {}
        for label, sc in self.subcorpora.items():
            len_pseudo_doc = sum(term_distr[doc_id][-1] for doc_id in sc)
            len_pseudo_docs[label] = len_pseudo_doc
        return len_pseudo_docs

    # @staticmethod
    # def get_tf_pseudo_doc(pseudo_doc: Set[int],
    #                       cluster: Set[int],
    #                       word_distr: term_distr_type
    #                       )-> Dict[str, int]:
    #     """Get term frequencies for the given pseudo-document.
    #
    #     Get: how often terms appears in given pseudo-doc-
    #     Needed: doc-ids of pseudo-doc, set of terms, term-frequency per doc
    #     Which terms? All term in one of the current clusters.
    #
    #     Args:
    #         pseudo_doc: A set of document ids.
    #         cluster: A set of term-ids.
    #         word_distr: Description at the top of the file in
    #             type-definitions.
    #     Return:
    #         {term_id: frequency}
    #     """
    #     tf_pseudo_doc = defaultdict(int)
    #     for doc_id in pseudo_doc:
    #         for term_id in word_distr[doc_id]:
    #             if term_id in cluster:
    #                 tf_pseudo_doc[term_id] += word_distr[doc_id][term_id][0]
    #     return tf_pseudo_doc

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
        dict_template = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        tf_pseudo_docs = {}
        for label, sc in self.subcorpora.items():
            for doc_id in sc:
                for term_id in term_distr[doc_id]:
                    if term_id in self.clusters_inv:
                        if term_id not in tf_pseudo_docs:
                            tf_pseudo_docs[term_id] = dict_template
                        tf_pseudo_docs[term_id][label] += term_distr[doc_id][term_id][0]
        return tf_pseudo_docs

    # def get_term_scores_efficient(self,
    #                               word_distr: term_distr_type
    #                               ) -> Dict[int, Tuple[float, float]]:
    #     """More efficient version of get_term_scores.
    #
    #     Return:
    #         A dict mapping the term-id to concentration and popularity.
    #     """
    #     term_scores = defaultdict(list)
    #     for label, clus in self.clusters.items():
    #         subcorp = self.subcorpora[label]
    #         tf_Dk = sum([word_distr[doc_id][-1] for doc_id in subcorp])
    #         for term_id in clus:
    #             tf_t_Dk = 0
    #             for doc_id in subcorp:
    #                 if term_id in word_distr[doc_id]:
    #                     tf_t_Dk += word_distr[doc_id][term_id][0]
    #
    #             pop_score = log(tf_t_Dk + 1) / log(tf_Dk)
    #             term_scores[term_id].append(pop_score)
    #
    #     return term_scores

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
