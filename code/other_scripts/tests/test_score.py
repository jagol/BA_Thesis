import sys
sys.path.append('../pipeline/')
import unittest
from score import Scorer


corpus_raw = {
    1: [1, 2, 3, 4, 5, 6, 7],
    2: [2, 3, 2],
    3: [5, 8, 1],
    4: [9, 1, 1, 3, 4, 9]
}

clusters_raw = {
    1: [1, 9],
    2: [2, 5, 8]
}

tf = {
    1: {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1
    },
    2: {
        2: 2,
        3: 1
    },
    3: {
        5: 1,
        8: 1,
        1: 1
    },
    4: {
        1: 2,
        3: 1,
        4: 1,
        9: 2
    }
}

dl = {
    1: 7,
    2: 3,
    3: 3,
    4: 6
}


class TestScore(unittest.TestCase):

    def setUp(self):
        self.clusters = self.to_str_wlist(clusters_raw)
        self.corpus = self.to_str_wlist(corpus_raw)
        self.subc1 = self.sub_dict(['1', '4'], self.corpus)
        self.subc2 = self.sub_dict(['2', '3'], self.corpus)
        self.subcorpora = {'1': self.subc1, '2': self.subc2}
        self.level = 1
        self.tf = self.to_str_nolist(tf)
        self.dl = self.to_str_nolist(dl)
        self.scorer = Scorer(self.clusters, self.subcorpora, self.level)
        self.maxDiff = None

    def to_str_wlist(self, d):
        "Turn all keys and values (list elements) to strings."
        new_d = {}
        for key in d:
            new_v = set([str(i) for i in d[key]])
            new_key = key
            new_d[new_key] = new_v
        return new_d

    def to_str_nolist(self, d):
        "Turn all keys and values to strings."
        new_d = {}
        for key in d:
            new_v = str(d[key])
            new_key = str(key)
            new_d[new_key] = new_v
        return new_d

    def sub_dict(self, key_list, d):
        new_d = {}
        for key in key_list:
            new_d[key] = d[key]
        return new_d

    def test_get_pop_scores(self):
        actual = self.scorer.get_pop_scores()
        desired = {
            '1': 1
        }
        print(actual)

    def test_get_term_scores(self):
        pass


if __name__ == '__main__':
    unittest.main()