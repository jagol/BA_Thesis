import unittest
import sys
import numpy as np
sys.path.append('/home/pogo/Dropbox/UZH/BA_Thesis/code/pipeline/')
from generate_taxonomy import separate_gen_terms, update_title, \
    perform_clustering


clusters = {
    0: {3, 2, 1},
    1: {4, 5, 6}
}
term_scores = {
    1: (1, 1, 0.3),
    2: (1, 1, 0.2),
    3: (1, 1, 0.24),
    4: (1, 1, 0.25),
    5: (1, 1, 0.26),
    6: (1, 1, 1),
}
term_ids_to_embs_local = {
    1: np.array([0.1, 0.1, 0.1]),
    2: np.array([0.1, 0.1, 0.1]),
    3: np.array([0.1, 0.1, 0.1]),
    4: np.array([0.1, 0.1, 0.1]),
    5: np.array([0.1, 0.1, 0.1]),
    6: np.array([0.1, 0.1, 0.1]),
    7: np.array([0.1, 0.1, 0.1]),
    8: np.array([0.1, 0.1, 0.1]),
}
threshold = 0.25


class TestGenerateTaxonomy(unittest.TestCase):

    def test_separate_gen_terms(self):
        actual_output = separate_gen_terms(clusters, term_scores, threshold)
        proc_clusters = {
            0: {1},
            1: {4, 5, 6}
        }
        general_terms = [(2, 0.2), (3, 0.24)]
        desired_output = (proc_clusters, general_terms)
        self.assertEqual(actual_output, desired_output)

    def test_update_title(self):
        actual_output = update_title(term_ids_to_embs_local, clusters)
        desired_output = {
            1: np.array([0.1, 0.1, 0.1]),
            2: np.array([0.1, 0.1, 0.1]),
            3: np.array([0.1, 0.1, 0.1]),
            4: np.array([0.1, 0.1, 0.1]),
            5: np.array([0.1, 0.1, 0.1]),
            6: np.array([0.1, 0.1, 0.1]),
        }
        actual_output = {k: list(v) for k, v in actual_output.items()}
        desired_output = {k: list(v) for k, v in desired_output.items()}
        self.assertEqual(actual_output, desired_output)

    def test_perform_clusterung_less_than_5(self):
        term_ids_to_embs_local = {
            1: np.array([1.0, 1.0, 1.0]),
            2: np.array([0.1, -1.0, -0.5]),
            3: np.array([0.5, -1.0, -0.5]),
            4: np.array([1.1, 2, 1.5])
        }
        actual_output = perform_clustering(term_ids_to_embs_local)
        desired_output = {
            0: {1},
            1: {2},
            2: {3},
            3: {4},
        }
        self.assertEqual(actual_output, desired_output)

    def test_perform_clusterung_more_than_5(self):
        term_ids_to_embs_local = {
            1: np.array([1.0, 1.0, 1.0]),
            2: np.array([-1.0, -1.0, -0.5]),
            3: np.array([-1.0, -1.1, -0.5]),
            4: np.array([1.1, 2, 1.5]),
            5: np.array([1.1, 2, 1.5]),
            6: np.array([3.1, -2, 3.0]),
            7: np.array([3.1, -2.3, 3.0]),
            8: np.array([6.1, 1, 1.1]),
            9: np.array([6.1, 2, 1.1]),
            10: np.array([6.1, 1.5, 1.1])
        }
        # term_ids_to_embs_local = {k: list(v) for k, v in term_ids_to_embs_local.items()}
        actual_output = perform_clustering(term_ids_to_embs_local)
        desired_output = {
            0: {1},
            1: {2, 3},
            2: {4, 5},
            3: {6, 7},
            4: {8, 9, 10}
        }
        self.assertEqual(actual_output.keys(), desired_output.keys())


if __name__ == '__main__':
    unittest.main()
