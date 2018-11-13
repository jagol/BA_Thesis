from typing import *
from math import log, sqrt, exp
from collecctions import defaultdict

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
DO NEXT: implement get_topic_docs and check if implementation of get_tf 
is even necessary -> think about file storage system
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
                 cluster: cluster_type,
                 parent_corpus: Set[int],
                 path_base_corpus: str
                 ) -> None:
        """Initialize a Scorer object.

        Args:
            cluster: The clusters with the term-ids to be scored.
            parent_corpus: The doc-ids of the document, that make up the
                parent corpus (the corpus, to which the cluster belongs).
            path_base_corpus: The path to the original corpus file.
        """
        self.cluster = cluster
        self.parent_corpus = Corpus(parent_corpus, self.path_base_corpus)
        self.parent_corpus.get_corpus_docs()
        self.path_base_corpus = path_base_corpus
        self.topic_corpus = Corpus(self.cluster, self.path_base_corpus)
        self.topic_corpus.get_corpus_docs()

        # precompute term frequencies for all given terms
        self.tf_topic_corpus = self.get_tf(cluster, self.topic_corpus.docs)
        self.tf_parent_corpus = self.get_tf(cluster, self.parent_corpus.docs)

    def get_tf(self,
               cluster: cluster_type,
               corpus: corpus_type
               ) -> Dict[str, int]:
        """Get term frequency of terms in the cluster for the corpus.

        Args:
            cluster: A set of term-ids.
            corpus: A list of document-ids and their documents.
        Return:
            A dictionary mapping each term-id in the cluster to it's
            frequency in the corpus.
        """
        tf_dict = defaultdict(int)
        for id_, doc in corpus:
            for sent in doc:
                for word in sent:
                    for term_id in cluster:
                        if term_id == word:
                            tf_dict[term_id] += 1
        return tf_dict

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
                calcualted.
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
                calcualted.
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

    def get_topic_docs(self):
        pass
