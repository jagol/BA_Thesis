import unittest
import sys
import numpy as np
sys.path.append('/home/pogo/Dropbox/UZH/BA_Thesis/code/pipeline/')
from corpus import Corpus as Cp


clusters = {
    0: {0, 2, 4},
    1: {1, 3}
}


term_ids_to_embs_local = {
    0: np.array([1.0, 0.5]),
    1: np.array([3.0, 1.0]),
    2: np.array([1.0, 1.5]),
    3: np.array([1.0, 1.0]),
    4: np.array([1.0, 4.0]),
}


doc_ids = {1, 10, 3}


doc_embeddings = {
    1: np.array([1.0, 1.0]),
    10: np.array([0.5, 2.0]),
    3: np.array([2.0, 0.5])
}


topic_embeddings = {
    0: np.array([1.0, 2.0]),
    1: np.array([2.0, 1.0]),
    # 2: [2.0, 2.0],
    # 3: [2.0, 2.1],
    # 4: [2.0, 0.0]
}


df = {
    0: [2, 4],
    1: [4],
    2: [9, 7],
    3: [8, 9],
    4: [6]
}


class TestCorpus(unittest.TestCase):
    """Class to test the Corpus-class."""

    # def test_get_subcorpora(self):
    #     pass
    #
    # def test_get_subcorpora_tfidf(self):
    #     pass
    #
    # def test_get_subcorpora_embs(self):
    #     pass

    def test_get_doc_topic_sims(self):
        actual_output = Cp.get_doc_topic_sims(doc_embeddings, topic_embeddings)
        desired_output = {   # {doc-id: {topic/cluster-label: similarity}}
            1: np.array([0.9486832980505138, 0.9486832980505138]),
            10: np.array([0.9761870601839528, 0.6507913734559685]),
            3: np.array([0.6507913734559685, 0.9761870601839528])
        }
        actual_output = {k: list(v) for k, v in actual_output.items()}
        desired_output = {k: list(v) for k, v in desired_output.items()}
        self.assertEqual(actual_output, desired_output)

    def test_get_topic_docs(self):
        doc_topic_sims = {   # {doc-id: {topic/cluster-label: similarity}}
            1: np.array([0.9486832980505138, 0.9486832980505138]),
            10: np.array([0.9761870601839528, 0.6507913734559685]),
            3: np.array([0.6507913734559685, 0.9761870601839528])
        }
        m = 2
        actual_output = Cp.get_topic_docs(doc_topic_sims, m=m)
        desired_output = {  # {cluster-label: {A set of doc-ids}}
            0: {1, 10},
            1: {3}
        }
        self.assertEqual(actual_output, desired_output)

    def test_get_k_most_center_terms(self):
        m = 2
        actual_output = Cp.get_k_most_center_terms(
            clus_centers=topic_embeddings, clusters=clusters,
            term_ids_to_embs_local=term_ids_to_embs_local, k=m)
        desired_output = {  # {cluster-label: {A set of term-ids}}
            0: {2, 4},
            1: {1, 3}
        }
        self.assertEqual(actual_output, desired_output)

    def test_get_relevant_docs(self):
        clus_terms = {  # {cluster-label: {A set of term-ids}}
            0: {2, 4},
            1: {1, 3}
        }
        actual_output = Cp.get_relevant_docs(clus_terms, df)
        desired_output = {  # {cluster-label: {A set of doc-ids}}
            0: {6, 7, 9},
            1: {4, 8, 9}
        }
        self.assertEqual(actual_output, desired_output)


if __name__ == '__main__':
    unittest.main()
