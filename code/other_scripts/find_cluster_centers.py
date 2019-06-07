import os
import csv
from typing import Dict, List, Set, Tuple, Iterator, Union
from numpy import mean
from embeddings import Embeddings
from utility_functions import get_cmd_args, get_config


"""Script to find cluster centers and label concept nodes after the 
clustering process.
# Info on centers/centroids: 
https://stats.stackexchange.com/questions/51743/how-is-finding-the-centroid-different-from-finding-the-mean
"""

taxonomy_type = Dict[int, Dict[str, Union[List[int], List[Tuple[
    float, float, float]]]]]


def load_taxonomy(path_tax: str) -> taxonomy_type:
    """Load the taxonomy.

    Return:
        Taxonomy in the form:
        {node: child_ids: [...], terms: [(id, term, score)]}
    """
    taxonomy = {}
    with open(path_tax, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == 'None':
                continue
            node_id = row[0]
            child_ids = [int(i) for i in row[1:6]]
            terms = [tuple(t.split('|')) for t in row[6:]]
            terms = [(int(id_), term, score) for id_, term, score in terms]
            taxonomy[node_id] = {}
            taxonomy[node_id]['child_ids'] = child_ids
            taxonomy[node_id]['terms'] = terms
    return taxonomy


def get_clus_center(node: int,
                    taxonomy: taxonomy_type,
                    path_out: str,
                    ) -> Iterator[float]:
    """Get the cluster center for given node id."""
    emb_path = os.path.join(path_out, 'embeddings/'+str(node)+'.vec')
    term_ids = set([t[0] for t in taxonomy[node]['terms']])
    local_embeddings = Embeddings.load_term_embeddings(
        term_ids, emb_path, term_ids_to_embs_global)
    clus_center = mean(local_embeddings, axis=0)
    return clus_center


def get_most_center_term(node: int,
                         clus_center: Iterator[float],
                         path_out: str
                         ) -> Tuple[int, str, float]:
    """Get the term for the given cluster which is nearest to the center.

    Return:
        A tuple of the form: (term_id, term, distance_to_center)
    """
    pass


def write_to_csv():
    pass



def load_term_ids_to_embs_global(lemmatized: bool,
                                 emb_type: str,
                                 path_out: str):
    """Load global term embeddings."""
    global term_ids_to_embs_global
    path_emb_dir = os.path.join(path_out, 'embeddings/')
    if lemmatized:
        fname = 'embs_lemma_global_{}.vec'.format(emb_type)
    else:
        fname = 'embs_token_global_{}.vec'.format(emb_type)

    emb_path = path_emb_dir+fname
    term_ids = load_term_ids(lemmatized, path_out)
    term_ids_to_embs_global = Embeddings.load_term_embeddings(term_ids,
                                                              emb_path, {})



def load_term_ids(lemmatized: bool, path_out: str) -> Set[int]:
    """Load term ids from file."""
    path_term_dir = os.path.join(path_out, 'processed_corpus/')
    if lemmatized:
        path_terms = path_term_dir + 'lemma_terms_idxs.txt'
    else:
        path_terms = path_term_dir + 'token_terms_idxs.txt'

    terms = set()
    with open(path_terms, 'r', encoding='utf8') as f:
        for line in f:
            terms.add(int(line.strip('\n')))

    return terms


def main():
    config = get_config()
    args = get_cmd_args()
    path_out = config['paths'][args.location][args.corpus]['path_out']
    path_tax = os.path.join(path_out, 'hierarchy/taxonomy.csv')
    taxonomy = load_taxonomy(path_tax)
    load_term_ids_to_embs_global(config['lemmatized'], config['emb_type'],
                                 path_out)
    for node in taxonomy:
        clus_center = get_clus_center(node, path_out)
        most_center_term = get_most_center_term(node, clus_center, path_out)
        write_to_csv(node, clus_center, most_center_term, path_out)


if __name__ == '__main__':
    main()