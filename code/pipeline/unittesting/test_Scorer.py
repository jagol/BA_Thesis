import unittest
import sys
sys.path.append('/home/pogo/Dropbox/UZH/BA_Thesis/code/pipeline/')
from score import Scorer


term_distr = {  # {doc-id: {term-id: (tf, tfidf)}}
    1: {1: (1, 8.192153039641536), -1: 1},
    2: {9: (1, 12.984255368000671), -1: 1},
    3: {7: (2, 25.968510736001342), 5: (1, 12.984255368000671), -1: 3},
    4: {8: (1, 12.984255368000671), -1: 1},
    5: {3: (1, 12.984255368000671), 16: (1, 8.192153039641536), -1: 2},
    6: {11: (2, 25.968510736001342), -1: 2},
    7: {16: (4, 32.76861215856614), -1: 4},
    8: {1: (1, 8.192153039641536), -1: 1},
    9: {2: (2, 25.968510736001342), 4: (1, 12.984255368000671), -1: 3},
    -1: 1.3333333333333333
}


df = {  # {term_id: list of doc_ids}
    1: [1, 8],
    2: [9],
    3: [5],
    4: [9],
    5: [3],
    7: [3],
    8: [4],
    9: [2],
    11: [6],
    16: [7, 5]
}


df_pseudo_doc = {
    1: [2],
    2: [1, 2],
    3: [0],
    4: [1],
    5: [3],
    7: [3],
    8: [2],
    9: [0],
    11: [4],
    16: [4, 0]
}


idf = {  # N / log(df+1), where N is the number of docs
    1: 8.192153039641536,
    2: 12.984255368000671,
    3: 12.984255368000671,
    4: 12.984255368000671,
    5: 12.984255368000671,
    7: 12.984255368000671,
    8: 12.984255368000671,
    9: 12.984255368000671,
    11: 12.984255368000671,
    16: 8.192153039641536
}


subcorpora = {  # {clus_label: set of doc_ids}
    0: {2, 3, 5},
    1: {9},
    2: {4, 8, 9},
    3: {3},
    4: {6, 7}
}


clusters = {  # {clus_label: set of term_ids}
    0: {7, 3, 9},
    1: {2},
    2: {4, 1, 8},
    3: {5},
    4: {11, 16}
}


level = 0


class TestScorer(unittest.TestCase):
    """Class to test the Scorer class."""

    scorer = Scorer(clusters=clusters, cluster_centers={},
                    subcorpora=subcorpora, level=0)

    # def test_get_term_scores(self):
    #     actual_output = self.scorer.get_term_scores(word_distr=word_distr,
    #                                                 df=df)
    #     desired_output = {}
    #     self.assertEqual(actual_output, desired_output)

    def test_get_pop_scores(self):
        actual_output = self.scorer.get_pop_scores(term_distr=term_distr,
                                                   df=df)
        desired_output = {  # N / log(df+1),
                            # where N is the number of pseudo -docs
            1: 0.43067655807339306,
            2: 1.0,
            3: 0.3868528072345416,
            4: 0.43067655807339306,
            5: 0.6309297535714574,
            7: 0.6131471927654585,
            8: 0.43067655807339306,
            9: 0.3868528072345416,
            11: 0.6131471927654585,
            16: 0.8982444017039272
        }
        self.assertEqual(actual_output, desired_output)

    # def test_get_con_scores(self):
    #     actual_output = self.scorer.get_con_scores(word_distr=word_distr,
    #                                                df=df)
    #     desired_output = {}
    #     self.assertEqual(actual_output, desired_output)
    #
    # def test_get_bm25_scores(self):
    #     actual_output = self.scorer.get_bm25_scores(term_distr=term_distr,
    #                                                 df_scores=df)
    #     desired_output = {
    #         1: {
    #             0: 0,
    #             1: 0,
    #             2: 0,
    #             3: 0,
    #             4: 0
    #         },
    #         2: {
    #             0: 0
    #         }
    #     }
    #     self.assertEqual(actual_output, desired_output)

    def test_get_idf_scores_pd(self):
        df_scores_pd = {  # N / log(df+1),
                          # where N is the number of pseudo -docs
            1: 1,  # desired: log(5/(1+1))
            2: 2,  # desired: log(5/(2+1))
            3: 1,  # desired: log(5/(1+1))
            4: 2,  # desired: log(5/(2+1))
            5: 2,  # desired: log(5/(2+1))
            7: 2,  # desired: log(5/(2+1))
            8: 1,  # desired: log(5/(1+1))
            9: 1,  # desired: log(5/(1+1))
            11: 1,  # desired: log(5/(1+1))
            16: 2  # desired: log(5/(2+1))
        }
        actual_output = self.scorer.get_idf_scores_pd(
            df_scores_pd=df_scores_pd)
        desired_output = {  # N / log(df+1),
                            # where N is the number of pseudo -docs
            1: 0.9162907318741551,  # desired: log(5/(1+1))
            2: 0.5108256237659907,  # desired: log(5/(2+1))
            3: 0.9162907318741551,  # desired: log(5/(1+1))
            4: 0.5108256237659907,  # desired: log(5/(2+1))
            5: 0.5108256237659907,  # desired: log(5/(2+1))
            7: 0.5108256237659907,  # desired: log(5/(2+1))
            8: 0.9162907318741551,  # desired: log(5/(1+1))
            9: 0.9162907318741551,  # desired: log(5/(1+1))
            11: 0.9162907318741551,  # desired: log(5/(1+1))
            16: 0.5108256237659907  # desired: log(5/(2+1))
        }
        self.assertEqual(actual_output, desired_output)

    def test_get_df_scores_pd(self):
        actual_output = self.scorer.get_df_scores_pd(term_distr=term_distr)
        desired_output = {  # N / log(df+1),
                            # where N is the number of pseudo -docs
            1: 1,  # desired: log(5/(1+1))
            2: 2,  # desired: log(5/(2+1))
            3: 1,  # desired: log(5/(1+1))
            4: 2,  # desired: log(5/(2+1))
            5: 2,  # desired: log(5/(2+1))
            7: 2,  # desired: log(5/(2+1))
            8: 1,  # desired: log(5/(1+1))
            9: 1,  # desired: log(5/(1+1))
            11: 1,  # desired: log(5/(1+1))
            16: 2  # desired: log(5/(2+1))
        }
        self.assertEqual(actual_output, desired_output)

    def test_get_tf_scores_pd(self):
        actual_output = self.scorer.get_tf_scores_pd(term_distr=term_distr)
        desired_output = {  # {term-id: {pd_id: tf}}
            1: {0: 0, 1: 0, 2: 1, 3: 0, 4: 0},
            2: {0: 0, 1: 2, 2: 2, 3: 0, 4: 0},
            3: {0: 1, 1: 0, 2: 0, 3: 0, 4: 0},
            4: {0: 0, 1: 1, 2: 1, 3: 0, 4: 0},
            5: {0: 1, 1: 0, 2: 0, 3: 1, 4: 0},
            7: {0: 2, 1: 0, 2: 0, 3: 2, 4: 0},
            8: {0: 0, 1: 0, 2: 1, 3: 0, 4: 0},
            9: {0: 1, 1: 0, 2: 0, 3: 0, 4: 0},
            11: {0: 0, 1: 0, 2: 0, 3: 0, 4: 2},
            16: {0: 1, 1: 0, 2: 0, 3: 0, 4: 4}
        }
        self.assertEqual(actual_output, desired_output)

    def test_form_pseudo_docs_bag_of_words(self):
        actual_output = self.scorer.form_pseudo_docs_bag_of_words(
            term_distr=term_distr)
        desired_output = {
            0: {3, 5, 7, 9, 16},
            1: {2, 4},
            2: {1, 2, 4, 8},
            3: {5, 7},
            4: {11, 16}
        }
        self.assertEqual(actual_output, desired_output)

    def test_get_len_pseudo_docs(self):
        actual_output = self.scorer.get_len_pseudo_docs(term_distr=term_distr)
        desired_output = {
            0: 6,
            1: 3,
            2: 5,
            3: 3,
            4: 6
        }
        self.assertEqual(actual_output, desired_output)


if __name__ == '__main__':
    unittest.main()
