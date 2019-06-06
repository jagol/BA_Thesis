import sys
sys.path.append('../')
from random import randint
from collections import defaultdict
import score


def create_clusters(n, m):
    """Create Clusters.

    n = num clusters
    m = list of length n, with number of cluster members.
    Return:
        {label: term-ids}
    """
    clusters = {}
    cur_term_id = 0
    for i in range(n):
        term_ids_clus = [j+cur_term_id for j in range(1, m+1)]
        cur_term_id = term_ids_clus[-1]
        clusters[i] = set(term_ids_clus)
    return clusters


def create_subcorpora(clusters, n):
    """Create Subcorpora.

    n: number of docs per subcorpus
    clusters = {label: term-ids}
    """
    subcorpora = {}
    cur_doc_id = 0
    for label in clusters:
        # Create doc-ids.
        doc_ids_clus = [j+cur_doc_id for j in range(1, n+1)]
        cur_doc_id = doc_ids_clus[-1]
        subcorpora[label] = set(doc_ids_clus)
    return subcorpora


def create_word_distr(clusters, subcorpora):
    """Return: {doc-id: word-id: (tf, tfidf)}"""
    word_distr = defaultdict(dict)
    all_doc_ids = set()
    for label, doc_ids in subcorpora.items():
        all_doc_ids = all_doc_ids.union(doc_ids)
    all_term_ids = set()
    for label, term_ids in clusters.items():
        all_term_ids = all_term_ids.union(term_ids)
    for term_id in all_term_ids:
        doc_x = randint(1, len(all_doc_ids))
        doc_y = randint(1, len(all_doc_ids))
        doc_z = randint(1, len(all_doc_ids))
        num_occur_x = randint(1, 10)
        num_occur_y = randint(1, 10)
        num_occur_z = randint(1, 10)
        word_distr[doc_x][term_id] = (num_occur_x, num_occur_x*1.3)
        word_distr[doc_y][term_id] = (num_occur_y, num_occur_y*0.75)
        word_distr[doc_z][term_id] = (num_occur_z, num_occur_z*2.314)
    for doc_id in word_distr:
        word_distr[doc_id][-1] = 4
    for label, doc_ids in subcorpora.items():
        for doc_id in doc_ids:
            word_distr[doc_id][-1] = 4
    return word_distr


if __name__ == '__main__':
    clusters = create_clusters(5, 500)
    subcorpora = create_subcorpora(clusters, 100000)
    word_distr = create_word_distr(clusters, subcorpora)
    scorer = score.Scorer(clusters, subcorpora, level=1)
    scores = scorer.get_term_scores(word_distr)