import os
import json
import subprocess
from collections import defaultdict
from typing import *
from corpus import get_relevant_docs
from clustering import Clustering
# from score import Scorer
from utility_functions import *


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
    # Load cmd args and configs.
    config = get_config()
    args = get_cmd_args()

    # Get paths.
    path_base_corpus = config['paths'][args.location][args.corpus]['path_in']
    path_out = config['paths'][args.location][args.corpus]['path_out']
    lemmatized = config['lemmatized']
    if lemmatized:
        path_term_ids = os.path.join(
            path_out, 'indexing/lemma_idxs_to_terms.json')
        path_df = os.path.join(path_out, 'frequency_analysis/df_lemmas.json')
        path_tf = os.path.join(path_out, 'frequency_analysis/tf_lemmas.json')
        path_tfidf = os.path.join(
            path_out, 'frequency_analysis/tfidf_lemmas.json')
    else:
        path_term_ids = os.path.join(
            path_out, 'indexing/token_idxs_to_terms.json')
        path_df = os.path.join(path_out, 'frequency_analysis/df_tokens.json')
        path_tf = os.path.join(path_out, 'frequency_analysis/tf_tokens.json')
        path_tfidf = os.path.join(path_out,
                                  'frequency_analysis/tfidf_tokens.json')

    # Define starting variables.
    term_ids = load_term_ids(path_term_ids)
    # base_corpus = set([i for i in range(get_num_docs(path_corpus))])
    base_corpus = get_base_corpus(path_base_corpus)
    tf_base = {}  # {doc_id: {word_id: freq}}
    with open(path_tf, 'r', encoding='utf8') as f:
        for i, doc_freqs in enumerate(f):
            tf_base[str(i)] = json.loads(doc_freqs.strip('\n'))
    with open(path_df, 'r', encoding='utf8') as f:
        df_base = json.load(f)  # {word_id: freq}
    with open(path_tfidf, 'r', encoding='utf8') as f:
        tfidf_base = json.load(f)

    # Start recursive taxonomy generation.
    rec_find_children(term_ids, base_corpus=base_corpus,
                      path_base_corpus=path_base_corpus,
                      cur_node_id=0, level=0, df_base=df_base,
                      tf_base=tf_base, path_out=path_out,
                      tfidf_base=tfidf_base)


def rec_find_children(term_ids: Set[str],
                      df_base: Dict[str, int],
                      tf_base: Dict[str, Dict[str, int]],
                      cur_node_id: int,
                      level: int,
                      base_corpus: Set[str],
                      path_base_corpus: str,
                      tfidf_base: Dict[str, Dict[str, float]],
                      path_out: str
                      ) -> None:
    """Recursive function to generate child nodes for parent node.

    Args:
        term_ids: The ids of the input terms.
        cur_node_id: The id of the current node in the Taxonomy. The
            current node is the node which contains all the terms in
            term_ids.
        level: The level or deepness of the taxonomy. The root node has
            level 0.
        path_base_corpus: Path to the corpus file with all documents.
        base_corpus: All doc_ids of the documents in the base corpus.
        df_base: df values for all terms in the base corpus, Form:
            {term_id: [doc1, ...]}
        tf_base: df values for all terms in the base corpus, Form:
            {doc_id: term_id: val}
        tfidf_base: tfidf values for all terms in the base corpus, Form:
            {doc_id: term_id: tfidf}
        path_out: The path to the output directory.
    """
    node_id = cur_node_id
    if len(term_ids) <= 5:
        return None

    n = int(len(base_corpus)/(5*level))
    corpus = get_relevant_docs(term_ids, base_corpus, n, tfidf_base)
    corpus_path = build_corpus_file(corpus, path_base_corpus, cur_node_id)
    emb_path = train_embeddings(corpus_path, cur_node_id, path_out)

    df_corpus = get_df_corpus(term_ids, corpus, df_base)
    tf_corpus = get_tf_corpus(corpus, tf_base)

    term_ids_to_embs = get_embeddings(term_ids, emb_path)  # {id: embedding}
    clusters = cluster(term_ids_to_embs)  # list[set(term_ids)] of len == 5
    term_scores = get_term_scores(clusters, df_corpus, tf_corpus)
    proc_clusters, concept_terms = process_clusters(clusters, term_scores)

    for label, clus in proc_clusters.items():
        node_id += 1
        rec_find_children(term_ids=clus, base_corpus=base_corpus,
                          path_base_corpus=path_base_corpus,
                          cur_node_id=node_id, level=level+1,
                          df_base=df_base, tf_base=tf_base, path_out=path_out,
                          tfidf_base=tfidf_base)


def process_clusters(clusters: Dict[str, Set[str]],
                     term_scores: Dict[str, Tuple[float, float]]
                     ) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Remove general terms and unpopular terms from clusters.

    For each cluster remove the unpopular terms and push up and
    remove concept terms.

    Args:
        clusters: A list of clusters. Each cluster is a set of doc-ids.
        term_scores: Maps each term-idx to its popularity and
            concentrations.
    Return:
        proc_cluster: Same as the input variable 'clusters', but with
            terms removed.
        concept_terms: A set of term-ids belonging to the concept.
    """
    proc_clusters = {}  # {label: clus}
    concept_terms = set()
    for label, clus in clusters.items():

        terms_to_remove = get_terms_to_remove(clus, term_scores)
        clus = remove(clus, terms_to_remove)

        concept_terms_clus = get_concept_terms(clus, term_scores)
        concept_terms.union(concept_terms_clus)
        clus = remove(clus, concept_terms)

        proc_clusters[label] = clus

    return proc_clusters, concept_terms


def build_corpus_file(doc_ids: Set[str],
                      path_base_corpus: str,
                      cur_node_id: int
                      ) -> str:
    """Generate corpus file from document ids.

    Args:
        doc_ids: The ids of the document belongig that make up the
            corpus.
        path_base_corpus: Path to the corpus file with all documents.
        cur_node_id: Id of the current node. Used for the name of the
            corpus file.
    Return:
        The path to the generated corpus file:
        'processed_corpora/<cur_node_id>_corpus.txt'
    """
    path_out = 'processed_corpus/{}.txt'.format(cur_node_id)

    # Buffer to store n number of docs. (less writing operations)
    docs_str = ''
    # yields sentences as strings
    with open(path_out, 'w', encoding='utf8') as f_out:
        for i, doc in enumerate(get_docs(path_base_corpus,
                                         word_tokenized=False)):
            if i in doc_ids:
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


def get_embeddings(term_ids: Set[str],
                   emb_path: str
                   ) -> Dict[str, List[float]]:
    """Get the embeddings for the given terms.

    Args:
        term_ids: The ids of the input terms.
        emb_path: The path to the given embedding file.
    Return:
        A dictionary of the form: {term_id: embedding}
    """
    term_id_to_emb = {}
    with open(emb_path, 'r', encoding='utf8') as f:
        embeddings = json.load(f)

    for term_id in term_ids:
        term_id_to_emb[term_id] = embeddings[term_id]

    return term_id_to_emb


def cluster(term_ids_to_embs: Dict[str, List[float]]) -> Dict[str, Set[str]]:
    """Cluster the given terms into 5 clusters.

    Args:
        term_ids_to_embs: A dictionary mapping term-ids to their
            embeddings.
    Return:
        A dictionary of mapping each cluster label to it0s cluster.
        Each cluster is a set of term-ids.
    """
    c = Clustering()
    term_ids_embs_items = [(k, v) for k, v in term_ids_to_embs.items()]
    results = c.fit([it[1] for it in term_ids_embs_items])
    labels = results['labels']
    print('density:', results['density'])
    clusters = defaultdict(set)
    for i in range(len(term_ids_embs_items)):
        term_id = term_ids_embs_items[i][0]
        label = labels[i]
        clusters[label].add(term_id)
    return clusters


def remove(clus: Set[str],
           terms_to_remove: Set[str]
           ) -> Set[str]:
    """Remove terms_to_remove from cluster.

    Args:
        clus: A set of term-ids.
        terms_to_remove: A set of term-ids.
    """
    return clus-terms_to_remove


def load_term_ids(path_term_ids: str) -> Set[str]:
    """Load the ids of all candidate terms.

    Args:
        path_term_ids: The path to the file containing term_ids. The
            file has one id per line.
    """
    term_ids = set()
    with open(path_term_ids, 'r', encoding='utf8') as f:
        for line in f:
            term_ids.add(line.strip('\n'))
    return term_ids


def get_term_scores(clusters: Dict[str, Set[str]],
                    df: Dict[str, int],
                    tf: Dict[str, Dict[str, int]]
                    ) -> Dict[str, Tuple[float, float]]:
    """Get the popularity and concentration for each term in clusters.

    The popularity of a term is always the popularity for the cluster
    the term belongs to. The concentration is cluster-independent.

    Args:
        clusters: A list of clusters. Each cluster is a set of term-ids.
        tf: The term frequencies of the given terms in the given
            subcorpus. Form: {doc_id: {term_id: frequeny}}
        df: The document frequencies of the given terms for the given
            subcorpus. Form: {term_id: frequency}
    """
    pass


def get_df_corpus(term_ids: Set[str],
                  corpus: Set[str],
                  df_base: Dict[str: int]
                  ) -> Dict[str, int]:
    """Get the document frequencies for given corpus and term-ids.

    Args:
        term_ids: The ids of the given terms.
        corpus: The ids of the documents making up the corpus.
        df_base: The document frequencies of the base corpus.
        {term_id: [doc_id1, ...]}
    Return:
        {term_id: num_docs}
    """
    out_dict = defaultdict(int)
    for term_id in term_ids:
        for doc_id in df_base[term_id]:
            if doc_id in corpus:
                out_dict[term_id] += 1
    return out_dict


def get_tf_corpus(corpus: Set[str],
                  tf_base: Dict[str: Dict[str, int]]
                  ) -> Dict[str, Dict[str, int]]:
    """Get the term frequencies for the given corpus.

    Args:
        corpus: The ids of the documents making up the corpus.
        tf_base: The term frequencies of the base corpus.
        {doc_id: {term_id: freq}}
    """
    out_dict = {}
    for doc_id in corpus:
        out_dict[doc_id] = tf_base[doc_id]
    return out_dict


def get_base_corpus(path_base_corpus: str):
    """Get the set of indices making up the base corpus.

    Args:
        path_base_corpus: Path to the corpus file.
    """
    return set([str(i) for i in range(get_num_docs(path_base_corpus))])


def get_terms_to_remove(clus: Set[str],
                        term_scores: Dict[str, Tuple[float, float]]
                        ) -> Set[str]:
    """Determine which terms to remove from the cluster.

    Args:
        clus: A set of doc-ids.
        term_scores: Maps each term-idx to its popularity and
            concentrations.
    Return:
        A set of term-ids of the terms to remove.
    """
    terms_to_remove = set()
    threshhold = 0.5
    for term_id in clus:
        pop = term_scores[term_id][0]
        if pop < threshhold:
            terms_to_remove.add(term_id)
    return terms_to_remove


def get_concept_terms(clus: Set[str],
                      term_scores: Dict[str, Tuple[float, float]]
                      ) -> Set[str]:
    """Determine the concept candidates in the cluster.

    Args:
        clus: A set of doc-ids.
        term_scores: Maps each term-idx to its popularity and
            concentrations.
    Return:
        A set of term-ids of the terms to remove.
    """
    concept_terms = set()
    threshhold = 0.5
    for term_id in clus:
        con = term_scores[term_id][1]
        if con < threshhold:
            concept_terms.add(term_id)
    return concept_terms


if __name__ == '__main__':
    generate_taxonomy()
