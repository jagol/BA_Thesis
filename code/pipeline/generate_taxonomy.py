import os
import json
import subprocess
from typing import *
from corpus import get_relevant_docs
from utility_functions import get_docs


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
    config = get_config()
    args = get_cmd_args()
    path_out = config['paths'][args.location][args.corpus]['path_out']
    path_term_ids = os.path.join(path, 'indexing/lemma_idxs_to_terms.json')
    term_ids = load_term_ids(path_term_ids)
    rec_tax_gen(term_ids, cur_node_id=0, level=0, path_out)


def rec_find_children(term_ids: Set[int], cur_node_id: int, level: int, path_out: str):
    """Recursive function to generate child nodes for parent node.

    Args:
        term_ids: The ids of the input terms.
        cur_node_id: The id of the current node in the Taxonomy. The
            current node is the node which contains all the terms in
            term_ids.
        level: The level or deepness of the taxonomy. The root node has
            level 0.
        path_out: The path to the output directory.
    """
    node_id = cur_node_id
    if level >= 5:
        return
    if len(term_ids) =< 5:
        return

    doc_ids = get_relevant_docs(term_ids)  # set of doc ids
    corpus_path = build_corpus(doc_ids, cur_node_id)
    emb_path = train_embeddings(corpus_path, cur_node_id)
    term_ids_to_embs = get_embeddings(term_ids, emb_path)  # {id: embedding}
    clusters = cluster(term_ids_to_embs)  # list[set(term_ids)] of len == 5
    least_conc = get_least_concentrated_terms(clusters)

    for clus in clusters:
        repr_terms = get_repr_terms(clus, term_ids_to_embs)  # List[Tuple[term_id, score]]
        terms_to_remove = repr_terms.union(unrepr_terms)
        clus = remove(clus, terms_to_remove)
        node_id += 1
        return rec_find_children(clus, node_id, level+1, path_out)


def build_corpus(doc_ids: Set[int], cur_node_id: int) -> str:
    """Generate corpus file from document ids.

    Reads the TOKEN-Corpus!!!

    Args:
        doc_ids: The ids of the document belongig that make up the
            corpus.
        cur_node_id: Id of the current node. Used for the name of the
            corpus file.
    Return:
        The path to the corpus file:
        'processed_corpora/<cur_node_id>_corpus.txt'
    """
    path_in = 'processed_corpus/pp_token_corpus.txt'
    path_out = 'processed_corpus/{}.txt'.format(cur_node_id)

    # Buffer to store n number of docs. (less writing operations)
    docs_str = ''
    # yields sentences as strings
    with open(path_out, 'w', encoding='utf8') as f_out:
        for i, doc in enumerate(get_docs(path_in, word_tokenized=False)):
            doc_str = ''
            for sent in doc:
                line = sent + '\n'
                doc_str += line
            doc_str += '\n'
            docs_str += doc_str

            if i % 1000 == 0:
                f_out.write(docs_str)
                docs_str = ''

        f_out.write(docs_str)

    return path_out


def train_embeddings(path_corpus: str,
                     cur_node_id: int,
                     path_out_dir: str
                     ) -> str:
    """Train word2vec embeddings on the given corpus.

    Args:
        path_corpus: The path to the corpus file.
        cur_node_id: Id of the current node. Used for the name of the
            embedding file.
        path_out_dir: The path to the output directory.
    Return:
        The path to the embedding file:
        'embeddings/<cur_node_id>_w2v.vec'
    """
    raw_path = 'embeddings/{}.vec'.format(cur_node_id)
    path_out = os.path.join(path_out_dir, raw_path)
    subprocess.call(
        ["./word2vec", "-threads", "12", "-train", path_corpus, "-output",
         path_out])
    return path_out


def get_embeddings(term_ids: Set[int],
                   emb_path: str
                   ) -> Dict[int, List[float]]:
    """Get the embeddings for the given terms.

    Args:
        term_ids: The ids of the input terms.
        emb_path: The path to the given embedding file.
    Return:
        A dictionary of the form: {term_id: embedding}
    """
    pass


def cluster(term_ids_to_embs: Dict[int, List[float]]) -> List[Set[int]]:
    """Cluster the given terms into 5 clusters.

    Args:
        term_ids_to_embs: A dictionary mapping term-ids to their
            embeddings.
    Return:
        A list of clusters. Each cluster is a set of term-ids.
    """
    pass


def get_least_concentrated_terms(clusters: List[Set[int]]):
    """Get the terms the most spread over topics.

    Args:
        clusters: A list of clusters as sets of term-ids.
    Return:
        A set of term ids.
    """
    pass


def get_repr_terms(clus: Set[int],
                   term_ids_to_embs: Dict[int, List[float]]
                   )-> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Get the most representative terms.

    Args:
        clus: The cluster as a set of term indices.
        term_ids_to_embs: A dictionary mapping term-ids to their
            embeddings.
    """
    pass


def remove(clus: Set[int],
           terms_to_remove: List[Tuple[int, float]
           ) -> Set[int]:
    """Remove terms_to_remove from cluster.

    Args:
        clus: A set of term-ids.
        terms_to_remove: A set of term-ids.
    """
    ids_to_remove = set([t[0] for t in terms_to_remove])
    return clus-ids_to_remove