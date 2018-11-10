from typing import *
from math import log, sqrt, exp

"""Compute term-candidate scores."""
"""
- prepare pseudo docs using tfidf scores
    - for each doc
        - for each term in the doc
            - calc the idf score idf = log(1.0 + N / keyword_idf[w]) 
                -> see dataset.py
            - calc the tf 
            - get the sum of all the idf's in a doc for each cluster
            - find the cluster for which this sum is the highest
- code formulas:
    - score(t, S_k) = sqrt(pop(t, S_k)*con(t, S_k)*hyp(t, S_k))
    - pop(t, S_k) = log(tf(t, D_k)+1)/log(tf(D_k))
        -> D_k = total number of tokens in D_k
    - con(t, S_k) = exp(rel(t, D_k))/(1+sum(exp(rel(t, D_1...D_n))))
        -> use gensim's bm25 scorer to calculate rel(t, D_k)
    - hyp(t, S_k) = sum(sim(t, projection(z1...zn)*score(z1...zn)))
        -> z1...zn: labels of the parent cluster
"""

class scorer:

    def __init__(self) -> None:
        pass

    def calc_score(self,
                   term_id: str,
                   cluster_id: int,
                   top_level=False
                   ) -> float:
        if top_level:
            score = self.calc_top_level_score(term_id, cluster_id)
        else:
            score = self.calc_gen_score(term_id, cluster_id)

        return score

    def calc_top_level_score(self, term_id: str, cluster_id: str) -> float:
        pop = self.get_pop(term_id, cluster_id)
        con = self.get_con(term_id, cluster_id)
        return sqrt(pop*con)

    def calc_gen_score(self, term_id: str, cluster_id: int) -> float:
        pop = self.get_pop(term_id, cluster_id)
        con = self.get_con(term_id, cluster_id)
        hyp = self.get_hyp(term_id, cluster_id)
        return sqrt(pop*con*hyp)

    def get_pop(self, term_id: str, cluster_id: int) -> float:
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
        topics_docs = self.get_topic_docs(cluster_id)
        numerator = log(self.get_tf(term_id, topics_docs)+1)
        denominator = log(self.get_tf(term_id, self.all_docs))
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
