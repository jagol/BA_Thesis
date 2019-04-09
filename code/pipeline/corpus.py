import pickle
import os
from typing import *
from collections import defaultdict
from utility_functions import get_sim, get_config
from numpy import mean
import numpy as np


doc_distr_type = DefaultDict[int, Union[Tuple[int, int], int]]
term_distr_type = DefaultDict[int, doc_distr_type]
# {doc-id: {term-id: (term-freq, tfidf)}} doc-length is at word-id -1


class Corpus:

    @staticmethod
    def get_doc_embeddings(path_out: str) -> Dict[int, List[float]]:
        """Compute document embeddings using term-embeddings and tfidf.

        Compute document embeddings though average of tfidf weighted term
        embeddings.

        The embedding for each document d_e is computed as:
        d_e = avg(tfidf(t1..tn)*emb(t1..tn))
        where t is a term in d.

        Args:
            path_out: Path to the output directory.
        Return:
            doc_embeddings: {doc-id: embedding}
        """
        config = get_config()
        lemmatized = config['lemmatized']
        emb_type = config['embeddings']
        if not lemmatized:
            if emb_type == 'Word2Vec':
                path_doc_embs = os.path.join(
                    path_out, 'embeddings/doc_embs_token_Word2Vec.pickle')
            elif emb_type == 'GloVe':
                path_doc_embs = os.path.join(
                    path_out, 'embeddings/doc_embs_token_GloVe.pickle')
            else:
                raise Exception('Error! Embedding type not recognized.')
        else:
            if emb_type == 'Word2Vec':
                path_doc_embs = os.path.join(
                    path_out, 'embeddings/doc_embs_lemma_Word2Vec.pickle')
            elif emb_type == 'GloVe':
                path_doc_embs = os.path.join(
                    path_out, 'embeddings/doc_embs_lemma_GloVe.pickle')
            else:
                raise Exception('Error! Embedding type not recognized.')
        doc_embeddings = pickle.load(open(path_doc_embs, 'rb'))
        return doc_embeddings

    @staticmethod
    def get_topic_embeddings(clusters: Dict[int, Set[int]],
                             term_ids_to_embs: Dict[int, List[float]]
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

    @classmethod
    def get_subcorpora_tfidf(cls,
                             n: int,
                             clusters: Dict[int, Set[int]],
                             term_distr: term_distr_type,
                             ) -> Dict[int, Set[int]]:
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

        Strategy:
            1. Calculate the strength per topic per cluster
                - create topic_docs of form: {clus-label: {(doc-id, strength)}}
            2. Trim such that each cluster only contains the strongest n
                docs.
                - filter and return the form: {clus-label: {doc-id}}

        Args:
            n: The top n scored documents are chosen for the pseudo corpus.
                n should be chosen as num_docs / n_clus
                where num_docs denotes the number of documents in the base
                corpus and n_clus denotes the number of clusters (or just
                the number of parts) the base corpus is divided into.
            clusters: {clus-label: set of term-ids}
            term_distr: description in type definitions at the top of the
                document
        Return:
            {clus-label: doc-ids}
        """
        clusters_inv = cls.invert_clusters(clusters)
        topic_doc_strengths = cls.get_topic_doc_strengths(
            clusters_inv, term_distr, len(clusters))
        topic_doc_strengths = cls.trim_top_n_per_clus(topic_doc_strengths, n)
        return cls.remove_strengths(topic_doc_strengths)

    @classmethod
    def get_topic_doc_strengths(cls,
                                clusters_inv: Dict[int, int],
                                word_distr: term_distr_type,
                                num_clusters: int
                                ) -> DefaultDict[int, List[Tuple[int, float]]]:
        """For each document get the strength for each cluster.

        Args:
            clusters_inv: {term-id: clus-label}
            word_distr: description in type definitions at the top of the
                document
            num_clusters: The number of clusters.
        Return:
            {clus-label: [(doc-id, strength), ...]}
        """
        clusters_term_ids = set(clusters_inv.keys())
        topic_doc_strengths = defaultdict(list)
        # {label: {(doc-id, strength)}}
        for doc_id in word_distr:
            doc_clus_strengths = num_clusters * [0.0]  # clus-label as index
            for term_id in word_distr[doc_id]:
                if term_id not in clusters_term_ids:
                    continue
                tfidf = word_distr[doc_id][term_id][1]
                clus_label = clusters_inv[term_id]
                doc_clus_strengths[clus_label] += tfidf
            clus_label, strength = cls.get_strongest_topic(doc_clus_strengths)
            topic_doc_strengths[clus_label].append((doc_id, strength))
        return topic_doc_strengths

    @staticmethod
    def trim_top_n_per_clus(topic_docs: DefaultDict[int, List[Tuple[int,
                                                                    float]]],
                            n: int
                            ) -> DefaultDict[int, List[Tuple[int, float]]]:
        """Only return the top n documents by strength per cluster.

        Args:
            topic_docs: {label: {(doc-id, strength)}}
            n: int
        Return:
            {clus-label: Set of doc-ids}
        """
        for clus_label in topic_docs:
            topic_docs[clus_label].sort(key=lambda tpl: tpl[1], reverse=True)
            topic_docs[clus_label] = topic_docs[clus_label][:n]
        return topic_docs

    @staticmethod
    def invert_clusters(clusters: Dict[int, Set[int]]) -> Dict[int, int]:
        """Invert clusters to term-ids as keys and clus-labels as values.

        Args:
            clusters: {clus-label: Set of term-ids}
        Return:
            {term-id: clus-label}
        """
        clusters_inv = {}
        for label in clusters:
            for term_id in clusters[label]:
                clusters_inv[term_id] = label
        return clusters_inv

    @staticmethod
    def remove_strengths(topic_docs: DefaultDict[int, List[Tuple[int, float]]]
                         ) -> Dict[int, Set[int]]:
        """Remove the topic-strength score and return instead set of doc-ids.

        Args:
            topic_docs: topic_docs: {label: {(doc-id, strength)}}
        Return:
            {clus_label: set of doc-ids}
        """
        subcorpora = {}
        for clus_label in topic_docs:
            doc_ids = set([did for did, strength in topic_docs[clus_label]])
            subcorpora[clus_label] = doc_ids
        return subcorpora

    @staticmethod
    def get_strongest_topic(doc_membership: List[float]
                            ) -> Tuple[int, float]:
        """Get the topic with the highest aggregated tfidf score.

        Code from here:
        https://stackoverflow.com/questions/268272/
        getting-key-with-maximum-value-in-dictionary

        Args:
            doc_membership: A dict mapping a topic-label to its
                strength in the document.
        Return:
            A tuple of the form: (label, strength).
        """
        strongest_score = max(doc_membership)
        label = doc_membership.index(strongest_score)
        return label, strongest_score

    @staticmethod
    def get_doc_topic_sims(doc_embeddings: Dict[int, List[float]],
                           topic_embeddings: Dict[int, List[float]]
                           ) -> Dict[int, List[float]]:
        """Get the similarities between topic and document vectors.

        Args:
            doc_embeddings: Maps doc-ids to their embeddings.
            topic_embeddings: Maps topic-ids to their embeddings.
        Return:
            A dict of the form: {doc-id: {topic/cluster-label: similarity}}
        """
        doc_topic_sims = {i: np.empty(5) for i in doc_embeddings}
        for topic, temb in topic_embeddings.items():
            for doc, demb in doc_embeddings.items():
                doc_topic_sims[doc][topic] = get_sim(temb, demb)
        return doc_topic_sims

    @classmethod
    def get_doc_topic_sims_matrix_mul(cls,
                                      doc_embeddings: Dict[int, List[float]],
                                      topic_embeddings: Dict[int, List[float]]
                                      ) -> Dict[int, List[float]]:
        """Get the similarities between topic and document vectors.

        Faster version using matrix multplication.

        Args:
            doc_embeddings: Maps doc-ids to their embeddings.
            topic_embeddings: Maps topic-ids to their embeddings.
        Return:
            A dict of the form: {doc-id: {topic/cluster-label: similarity}}
        """
        doc_topic_sims = {i: np.empty(5) for i in doc_embeddings}
        for tlabel, temb in topic_embeddings.items():
            cls.calc_sims_new_way(temb, doc_embeddings, tlabel, doc_topic_sims)
        return doc_topic_sims

    @staticmethod
    def get_matrix(doc_embs):
        num_embs = len(doc_embs)
        doc_ids = np.zeros(num_embs, dtype=int)
        matrix = np.empty(shape=(num_embs, 100))
        i = 0
        for doc_id, emb in doc_embs.items():
            doc_ids[i] = doc_id
            matrix[i] = emb
            i += 1
        return matrix, doc_ids

    @staticmethod
    def matrix_to_dict(sim_matrix, doc_ids, topic_label, doc_topic_sims):
        for i in range(len(doc_ids)):
            doc_topic_sims[doc_ids[i]][topic_label] = sim_matrix[i]

    @staticmethod
    def cosine_similarities(vector, matrix):
        vector_norm = np.linalg.norm(vector)
        matrix_norm = np.linalg.norm(matrix, axis=1)
        return (matrix @ vector) / (matrix_norm * vector_norm)

    @classmethod
    def calc_sims_new_way(cls,
                          topic_emb,
                          doc_embs,
                          topic_label,
                          doc_topic_sims
                          ):
        matrix, doc_ids = cls.get_matrix(doc_embs)
        del doc_embs
        sim_matrix = cls.cosine_similarities(topic_emb, matrix)
        del matrix
        del topic_emb
        cls.matrix_to_dict(sim_matrix, doc_ids, topic_label, doc_topic_sims)

    @classmethod
    def get_topic_docs(cls,
                       doc_topic_sims: Dict[int, List[float]],
                       m: Union[int, None]=None
                       ) -> Dict[int, Set[int]]:
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
            topic = cls.get_topic_with_max_sim(topic_sims)
            td_sc[topic].append((doc, topic_sims[topic]))
        topic_docs = defaultdict(set)

        # If m is defined only use the top m docs.
        if m:
            for tc in td_sc:
                docs_sort = sorted(td_sc[tc],
                                   key=(lambda t: t[1]), reverse=True)
                topic_docs[tc] = set([tpl[0] for tpl in docs_sort[:m]])
        else:
            for tc in td_sc:
                doc_ids = [doc_id for doc_id, score in td_sc[tc]]
                topic_docs[tc] = set(doc_ids)
        return topic_docs

    @staticmethod
    def get_topic_with_max_sim(topic_sims: List[float]) -> int:
        """Get the topic with the highest similarity score.

        Args:
            topic_sims: Array of floats with length 5.
        Return:
            The id of the topic with the highest similarity which is the
            index of the highest similarity in the array.
        """
        topic = -1
        sim_max = -1
        for i, sim in enumerate(topic_sims):
            if sim > sim_max:
                topic = i
                sim_max = sim
        return topic
