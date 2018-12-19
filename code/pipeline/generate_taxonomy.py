import os
import json
from typing import *


def generate_taxonomy():
    """Generate a taxonomy for a preprocessed corpus.

    Steps:
    - check if num-terms < num-clusters to build, if yes: stop
    - given terms, find the most relevant docs
    - build corpus using given doc indices
    - train embeddings on corpus
    - load terms and get their embeddings
    - cluster terms
    - find cluster labels and remove them from cluster:
        - prepare frequencies etc
        - calculate score for each term in cluster
        - get top n most representative terms and store as labels
        - remove them from cluster
        - start again at beginning using the resulting cluster
    """
    # word2vec training
    subprocess.call(
        ["./word2vec", "-threads", "10", "-train", input_f, "-output",
         output_f])
    path_term_ids = os.path.join(path, 'indexing/lemma_idxs_to_terms.json')
    term_ids = load_term_ids(path_term_ids)
    rec_tax_gen(term_ids, cur_node_id=0, level=0)


def rec_find_children(term_ids: Set[int], cur_node_id: int, level: int):
    """Recursive function to generate child nodes for parent node.

    Args:
        term_ids: The ids of the input terms.
        cur_node_id: The id of the current node in the Taxonomy. The
            current node is the node which contains all the terms in
            term_ids.
        level: The level or deepness of the taxonomy. The root node has
            level 0.
    """
    node_id = cur_node_id
    if level >= 5:
        return
    if len(term_ids) =< 5:
        return

    doc_ids = get_relevant_docs(term_ids)  # set of doc ids
    corpus_path = build_corpus(doc_ids, cur_node_id)
    emb_path = train_embeddings(corpus_path, cor_node_id)
    term_ids_to_embs = get_embeddings(emb_path)  # {id: embedding}
    clusters = cluster(term_ids_to_embs)  # list[set(term_ids)] of len == 5

    for clus in clusters:
        repr_terms, unrepr_terms = get_terms(clus, term_ids_to_embs)  # List[Tuple[term_id, score]]
        terms_to_remove = repr_terms.union(unrepr_terms)
        clus = remove(terms_to_remove, clus)
        node_id += 1
        return rec_find_children(clus, node_id, level+1)
