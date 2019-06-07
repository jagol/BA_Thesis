import os
import csv
import json
# from generate_taxonomy import write_tax_to_file
from typing import Dict, List, Set, Tuple, Any
from label_scoring import LabelScorer


def main():
    global idx_to_term
    from utility_functions import get_config, get_cmd_args
    config = get_config()
    args = get_cmd_args()
    path_out = config['paths'][args.location][args.corpus]['path_out']

    path_idx_to_term = os.path.join(path_out, 'indexing/idx_to_token.json')
    print('Load idx-term mappings...')
    with open(path_idx_to_term, 'r', encoding='utf8') as f:
        idx_to_term_str = json.load(f)
        idx_to_term = {int(k): v for k, v in idx_to_term_str.items()}

    taxonomy = load_taxonomy(path_out)

    global ls
    ls = LabelScorer(config, args)

    # Run labeling with repr score as metric.
    path_tax_frep = os.path.join(path_out, 'concept_terms/tax_labels_repr.csv')
    tax_label_file = open(path_tax_frep, 'w', encoding='utf8')
    csv_writer = csv.writer(tax_label_file, delimiter=',')
    rec_find_labels(path_out, taxonomy, 10, [], 0, csv_writer, cos=False,
                    label_score=False)

    # Run labeling with cosine similarity as metric.
    path_tax_fsim = os.path.join(path_out, 'concept_terms/tax_labels_sim.csv')
    tax_label_file = open(path_tax_fsim, 'w', encoding='utf8')
    csv_writer = csv.writer(tax_label_file, delimiter=',')
    rec_find_labels(path_out, taxonomy, 10, [], 0, csv_writer, cos=True,
                    label_score=False)


def rec_find_labels(path_out: str,
                    taxonomy: Dict[int, List[int]],
                    top_k: int,
                    top_parent_terms: List[Tuple[int, float]],
                    node_id: int,
                    csv_writer: Any,
                    cos: bool,
                    label_score: bool
                    ) -> None:
    """Find the most representative labels for each cluster.

    Args:
        path_out: The path to the output directory.
        taxonomy: Dictionary that maps each node-id to its child-ids.
        top_k: The k most representative terms are selected as labels.
        top_parent_terms: The top k terms of the parent.
        node_id: The id of the root node.
        csv_writer: file-object to which the labels are written.
        cos: If true, use the cosine similarity, else use the repr
            score.
        label_score: If true, use the label score to find the top labels
            of a topic.
    """
    child_ids = taxonomy.get(node_id)
    if not child_ids:
        return

    if node_id != 0:
        top_k_terms = get_top_k_terms(path_out, top_k, top_parent_terms,
                                      node_id, cos, label_score)
        child_ids_as_dict = {i: chid for i, chid in enumerate(child_ids)}
        write_tax_to_file(node_id, child_ids_as_dict, top_k_terms, csv_writer)
    else:
        top_k_terms = top_parent_terms

    for child_id in child_ids:
        print(node_id, child_id)
        rec_find_labels(path_out, taxonomy, top_k, top_k_terms, child_id,
                        csv_writer, cos, label_score=False)


def get_top_k_terms(path_out: str,
                    top_k: int,
                    parent_terms: List[Tuple[int, float]],
                    node_id: int,
                    cos: bool,
                    label_score: bool
                    ) -> List[Tuple[int, float]]:
    """Get the top k terms.

    If cos is false, use the term-representativeness score, if it is true
    use the cosine similarity between term and cluster-center as a
    metric.

    Args:
        path_out: The path to the output directory.
        top_k: The k most representative terms are selected as labels.
        parent_terms: The top k terms of the parent node.
        node_id: The id of the root node.
        cos: If true, use the cosine similarity, else use the repr
            score.
        label_score: If true, use the label score to find the top labels
            of a topic.
    Return:
        A list of tuples of the form. (term_id, score).
    """
    path_concept_terms = os.path.join(path_out, 'concept_terms/')
    term_scores = load_scores(path_concept_terms, node_id, cos=cos)
    clus_terms = load_clus_terms(path_concept_terms, node_id)
    clus_terms_scores = {tid: term_scores[tid] for tid in clus_terms}
    # {term_id: (term, score)}
    if label_score:
        label_scores = ls.score(clus_terms_scores, parent_terms)
    else:
        label_scores = term_scores
    concept_term_scores = []  # List of tuples.
    for term_id in clus_terms:
        score = label_scores[term_id][1]
        concept_term_scores.append((term_id, score))
    concept_term_scores.sort(key=lambda t: t[1], reverse=True)
    return concept_term_scores[:top_k]


def load_taxonomy(path_out: str) -> Dict[int, List[int]]:
    """Load the structure of the taxonomy.

    Args:
        path_out: The path to the output directory.
    Return:
        A dict mapping each node to it's list of child-nodes.
    """
    taxonomy = {}
    path_tax = os.path.join(path_out, 'hierarchy/taxonomy.csv')
    with open(path_tax, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            node_id = int(row[0])
            child_ids = [int(nid) for nid in row[1:6]]
            taxonomy[node_id] = child_ids
    return taxonomy


def load_scores(path_concept_terms: str,
                node_id: int,
                cos: bool = False
                ) -> Dict[int, Tuple[str, float]]:
    """Load the term scores for the terms in the given node.

    Args:
        path_concept_terms: The path to the directory containing term-files.
        Used to get file of the form:
            term_id SPACE term SPACE score NEWLINE
        node_id: The id of the node.
        cos: If true, load the cosine similarity, else load the repr
            score.
    Return:
        A dictionary mapping term-ids to a tuple of the form:
        (term, score).
    """
    if not cos:
        # Load representativeness scores.
        fpath = path_concept_terms + '{}_scores.txt'.format(node_id)
    else:
        # Load cosine similarities.
        fpath = path_concept_terms + '{}_cnt_dists.txt'.format(node_id)

    term_scores = {}
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f:
            term_id, term, score = line.strip('\n').split(' ')
            term_scores[int(term_id)] = (term, float(score))

    return term_scores


def load_clus_terms(path_concept_terms: str, node_id: int) -> Set[int]:
    """Load the terms belonging to the given node.

    Args:
        path_concept_terms: The path to the directory containing term-files.
        Used to get file of the form:
            term_id SPACE term SPACE score NEWLINE
        node_id: The id of the node.
    Return:
        A set of term-ids.
    """
    fpath = path_concept_terms + '{}.txt'.format(node_id)
    concept_terms = set()
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f:
            term_id, term, score = line.strip('\n').split(' ')
            concept_terms.add(int(term_id))
    return concept_terms


def write_tax_to_file(cur_node_id: int,
                      child_ids: Dict[int, int],
                      repr_terms: List[Tuple[int, float]],
                      csv_writer: Any,
                      only_id: bool = False
                      ) -> None:
    """Write the current node with terms and child-nodes to file.

    concept_terms is a list containing tuples of the form:
    (idx, term_score).

    The term score is the one which got the term pushed up. (highest one)
    """
    if only_id:
        row = [cur_node_id, str(None)]
    else:
        concept_terms = []
        for idx, score in repr_terms:
            term = idx_to_term[idx]
            term_w_score = '{}|{}|{:.3f}'.format(idx, term, score)
            concept_terms.append(term_w_score)
        child_nodes = [str(c) for c in child_ids.values()]
        row = [str(cur_node_id)] + child_nodes + concept_terms
        # print('Write: {}'.format(row))
    csv_writer.writerow(row)


if __name__ == '__main__':
    main()
