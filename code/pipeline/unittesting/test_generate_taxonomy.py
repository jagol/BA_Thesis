import unittest
import sys
import numpy as np
sys.path.append('/home/pogo/Dropbox/UZH/BA_Thesis/code/pipeline/')
from generate_taxonomy import separate_gen_terms, update_title


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


if __name__ == '__main__':
    unittest.main()
