# import os
# import json
# import re
import csv
import subprocess
from collections import defaultdict
from math import sqrt
from typing import *
from corpus import get_relevant_docs, get_doc_embeddings, get_topic_embeddings
from clustering import Clustering
from score import Scorer
from utility_functions import *


# Define global variables.
node_counter = 0


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
    print('Load and parse cmd args...')
    config = get_config()
    args = get_cmd_args()

    # Set paths.
    # path_base_corpus = config['paths'][args.location][args.corpus]['path_in']
    print('Set paths...')
    path_out = config['paths'][args.location][args.corpus]['path_out']
    lemmatized = config['lemmatized']
    if lemmatized:
        path_term_ids = os.path.join(
            path_out, 'processed_corpus/lemma_terms_idxs.txt')
        path_df = os.path.join(path_out, 'frequency_analysis/df_lemmas.json')
        path_tf = os.path.join(path_out, 'frequency_analysis/tf_lemmas.json')
        path_tfidf = os.path.join(
            path_out, 'frequencies/tfidf_lemmas.json')
        path_base_corpus = os.path.join(
            path_out, 'processed_corpus/pp_lemma_corpus.txt')
        path_base_corpus_ids = os.path.join(
            path_out, 'processed_corpus/lemma_idx_corpus.txt')
        path_embeddings_global = os.path.join(
            path_out, 'embeddings/token_embeddings_global.vec')
    else:
        path_term_ids = os.path.join(
            path_out, 'processed_corpus/token_terms_idxs.txt')
        path_df = os.path.join(path_out, 'frequencies/df_tokens.json')
        path_tf = os.path.join(path_out, 'frequencies/tf_tokens.json')
        path_tfidf = os.path.join(path_out, 'frequencies/tfidf_tokens.json')
        path_base_corpus = os.path.join(
            path_out, 'processed_corpus/pp_token_corpus.txt')
        path_base_corpus_ids = os.path.join(
            path_out, 'processed_corpus/token_idx_corpus.txt')
        path_embeddings_global = os.path.join(
            path_out, 'embeddings/lemma_embeddings_global.vec')

    path_dl = os.path.join(path_out, 'frequencies/dl.json')
    path_taxonomy = os.path.join(path_out, 'hierarchy/taxonomy.csv')

    tax_file = open(path_taxonomy, 'w', encoding='utf8', newline='')
    csv_writer = csv.writer(tax_file, delimiter=',')

    # Define starting variables.
    print('load term-ids...')
    term_ids = load_term_ids(path_term_ids)
    print('load global embeddings...')
    term_ids_to_embs_global = get_embeddings(term_ids, path_embeddings_global)
    print('load base corpus...')
    base_corpus = get_base_corpus(path_base_corpus)
    print('load tf-base...')
    tf_base = {}  # {doc_id: {word_id: freq}}
    with open(path_tf, 'r', encoding='utf8') as f:
        for i, doc_freqs in enumerate(f):
            tf_base[str(i)] = json.loads(doc_freqs.strip('\n'))
    print('load df-base...')
    with open(path_df, 'r', encoding='utf8') as f:
        df_base = json.load(f)  # {word_id: [doc_id1, ...]}
    print('load tfidf-base...')
    with open(path_tfidf, 'r', encoding='utf8') as f:
        tfidf_base = json.load(f)
    print('load dl-base...')
    with open(path_dl, 'r', encoding='utf8') as f:
        dl = json.load(f)

    # Start recursive taxonomy generation.
    rec_find_children(term_ids_local=term_ids, term_ids_global=term_ids,
                      base_corpus=base_corpus,
                      path_base_corpus_ids=path_base_corpus_ids,
                      cur_node_id=0, level=1, df_base=df_base,
                      tf_base=tf_base, path_out=path_out, dl=dl,
                      tfidf_base=tfidf_base, cur_corpus=base_corpus,
                      csv_writer=csv_writer,
                      term_ids_to_embs_global=term_ids_to_embs_global)

    print('Done.')


def rec_find_children(term_ids_local: Set[str],
                      term_ids_global: Set[str],
                      term_ids_to_embs_global: Dict[str, List[float]],
                      df_base: Dict[str, List[str]],
                      tf_base: Dict[str, Dict[str, int]],
                      dl: Dict[str, Union[int, float]],
                      cur_node_id: int,
                      level: int,
                      base_corpus: Set[str],
                      path_base_corpus_ids: str,
                      cur_corpus: Set[str],
                      tfidf_base: Dict[str, Dict[str, float]],
                      path_out: str,
                      csv_writer: Any
                      ) -> None:
    """Recursive function to generate child nodes for parent node.

    Args:
        term_ids_local: The ids of the current cluster terms.
        term_ids_global: The ids of all terms.
        term_ids_to_embs_global: Maps all term_ids to their global
            embeddings.
        cur_node_id: The id of the current node in the Taxonomy. The
            current node is the node which contains all the terms in
            term_ids.
        level: The level or deepness of the taxonomy. The root node has
            level 0.
        path_base_corpus_ids: Path to the corpus file with all documents
            in index-representation.
        base_corpus: All doc_ids of the documents in the base corpus.
        cur_corpus: All doc_ids of the documents in the current corpus.
        df_base: df values for all terms in the base corpus, Form:
            {term_id: [doc1, ...]}
        tf_base: df values for all terms in the base corpus, Form:
            {doc_id: term_id: val}
        dl: Maps the document ids to their document's length. Form:
            {doc-id: length}
            The average length is stored at key '-1'.
        tfidf_base: tfidf values for all terms in the base corpus, Form:
            {doc_id: {term_id: tfidf}}
        path_out: The path to the output directory.
        csv_writer: csv-writer-object used to write taxonomy to file.
    """
    print(10*'-'+' level {} node {} '.format(level, cur_node_id) + 10*'-')
    msg = 'start recursion on level {} with node id {}...'.format(level,
                                                                  cur_node_id)
    print(msg)
    # node_id = cur_node_id
    print('Number of candidate terms: {}'.format(len(term_ids_local)))
    if len(term_ids_local) <= 5:
        print('Less than 5 terms. Stop recursion...')
        return None

    n = int(len(base_corpus)/(5*level))
    print('build corpus file...')
    corpus_path = build_corpus_file(cur_corpus, path_base_corpus_ids,
                                    cur_node_id, path_out)
    print('train embeddings...')
    if level != 1:
        emb_path_local = train_embeddings(corpus_path, cur_node_id, path_out)
        print('get term embeddings...')
        term_ids_to_embs_local = get_embeddings(term_ids_local, emb_path_local)
        # {id: embedding}
    else:
        term_ids_to_embs_local = term_ids_to_embs_global

    print('cluster terms...')
    clusters = cluster(term_ids_to_embs_local)  # Dict[int, Set[int]]
    print('get subcorpora for clusters...')
    subcorpora = get_subcorpora(clusters, base_corpus, n, tfidf_base,
                                term_ids_to_embs_global)
    # {label: doc-ids}

    print('get term-frequencies...')
    tf_corpus = get_tf_corpus(term_ids_local, cur_corpus, tf_base)
    # TODO: modifiy get_tf_corpus. Get the tf for all docs in cur_docs
    print('compute term scores...')
    term_scores = get_term_scores(clusters, subcorpora, dl, tf_corpus, tf_base,
                                  level)
    print('remove terms from clusters...')
    proc_clusters, concept_terms = process_clusters(clusters, term_scores)
    print('concept terms:', concept_terms)
    child_ids = get_child_ids(proc_clusters)
    print('The child ids of {} are {}'.format(cur_node_id, str(child_ids)))
    print('write concept terms to file...')
    write_tax_to_file(cur_node_id, child_ids, concept_terms, csv_writer)

    print('start new recursion...')
    for label, clus in proc_clusters.items():
        node_id = child_ids[label]
        subcorpus = subcorpora[label]
        rec_find_children(term_ids_local=clus, base_corpus=base_corpus,
                          path_base_corpus_ids=path_base_corpus_ids,
                          cur_node_id=node_id, level=level+1, dl=dl,
                          df_base=df_base, tf_base=tf_base, path_out=path_out,
                          tfidf_base=tfidf_base, cur_corpus=subcorpus,
                          csv_writer=csv_writer,
                          term_ids_to_embs_global=term_ids_to_embs_global,
                          term_ids_global=term_ids_global)


def get_child_ids(proc_clusters: Dict[int, Set[str]]) -> Dict[int, int]:
    """Get the child-node-ids for the current node.

    Args:
        proc_clusters: A dict of the form {label: Set of term-ids}
            where the set of term-ids is a cluster.
    Return:
        A dictionary mapping labels to child-node-ids.
        {label: child-node-id}
    """
    global node_counter
    child_ids = {}
    for label in proc_clusters:
        node_counter += 1
        child_ids[label] = node_counter
    return child_ids


def write_tax_to_file(cur_node_id: int,
                      child_ids: Dict[int, int],
                      concept_terms: Set[str],
                      csv_writer: Any
                      ) -> None:
    """Write the current node with terms and child-nodes to file."""
    row = [cur_node_id] + list(child_ids.values()) + list(concept_terms)
    csv_writer.writerow(row)


def get_subcorpora(clusters: Dict[int, Set[str]],
                   base_corpus: Set[str],
                   n: int,
                   tfidf_base: Dict[str, Dict[str, float]],
                   term_ids_to_embs: Dict[str, List[float]]
                   ) -> Dict[int, Set[str]]:
    """Get the subcorpus for each cluster."""
    subcorpora = {}
    doc_embeddings = get_doc_embeddings(tfidf_base, term_ids_to_embs)
    # {doc_id: embedding}
    topic_embeddings = get_topic_embeddings(clusters, term_ids_to_embs)
    # {cluster/topic_label: embedding}
    for label, clus in clusters.items():
        subcorpora[label] = get_relevant_docs(clus, base_corpus, n, tfidf_base,
                                              doc_embeddings,
                                              topic_embeddings[label])
    return subcorpora


def process_clusters(clusters: Dict[int, Set[str]],
                     term_scores: Dict[str, Tuple[float, float]]
                     ) -> Tuple[Dict[int, Set[str]], Set[str]]:
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

        # terms_to_remove = get_terms_to_remove(clus, term_scores)

        # clus = remove(clus, terms_to_remove)

        concept_terms_clus = get_concept_terms(clus, term_scores)
        concept_terms = concept_terms.union(concept_terms_clus)
        clus = remove(clus, concept_terms)

        proc_clusters[label] = clus

    return proc_clusters, concept_terms


def build_corpus_file(doc_ids: Set[str],
                      path_base_corpus: str,
                      cur_node_id: int,
                      path_out: str
                      ) -> str:
    """Generate corpus file from document ids.

    Args:
        doc_ids: The ids of the document belongig that make up the
            corpus.
        path_base_corpus: Path to the corpus file with all documents.
        cur_node_id: Id of the current node. Used for the name of the
            corpus file.
        path_out: Path to the output directory.
    Return:
        The path to the generated corpus file:
        'processed_corpora/<cur_node_id>_corpus.txt'
    """
    p_out = os.path.join(path_out, 'processed_corpus/{}.txt'.format(
        cur_node_id))

    # Buffer to store n number of docs. (less writing operations)
    docs_str = ''
    # yields sentences as strings
    with open(p_out, 'w', encoding='utf8') as f_out:
        for i, doc in enumerate(get_docs(path_base_corpus,
                                         word_tokenized=False)):
            if str(i) in doc_ids:
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

    return p_out


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
        next(f)
        for line in f:
            vals = line.strip(' \n').split(' ')
            term_id = vals[0]
            if term_id in term_ids:
                emb = [float(f) for f in vals[1:]]
                term_id_to_emb[term_id] = emb

    return term_id_to_emb


def cluster(term_ids_to_embs: Dict[str, List[float]]) -> Dict[int, Set[str]]:
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
    # print('Cluster sizes: {}'.format([len(c) for c in clusters]))
    print('Cluster sizes: {}'.format(clusters))
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


def get_term_scores(clusters: Dict[int, Set[str]],
                    subcorpora: Dict[int, Set[str]],
                    # df_base: Dict[str, List[str]],
                    # df: Dict[str, int],
                    dl: Dict[str, Union[int, float]],
                    tf: Dict[str, Dict[str, int]],
                    tf_base: Dict[str, Dict[str, int]],
                    level: int
                    ) -> Dict[str, Tuple[float, float]]:
    """Get the popularity and concentration for each term in clusters.

    The popularity of a term is always the popularity for the cluster
    the term belongs to. The concentration is cluster-independent.

    Args:
        clusters: A list of clusters. Each cluster is a set of term-ids.
        subcorpora: Maps each cluster label to the relevant doc-ids.
        tf: The term frequencies of the given terms in the given
            subcorpus. Form: {doc_id: {term_id: frequeny}}
        tf_base:
        dl: Maps the document ids to their document's length. Form:
            {doc-id: length}
            The average length is stored at key '-1'.
        level: The recursion level.
    Return:
        A dictionary mapping each term-id to a tuple of the form:
        (popularity, concentration)
    (Old Args:
        df_base: Maps each term-id to a list of all the document-ids
            of the document in which it appears.
        df: The document frequencies of the given terms for the given
            subcorpus. Form: {term_id: frequency})
    """
    sc = Scorer(clusters, subcorpora, level)
    return sc.get_term_scores(tf, tf_base, dl)


def get_df_corpus(term_ids: Set[str],
                  corpus: Set[str],
                  df_base: Dict[str, int]
                  ) -> Dict[str, int]:
    """Get the document frequencies for a given corpus and term-ids.

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


def get_tf_corpus(term_ids: Set[str],
                  corpus: Set[str],
                  tf_base: Dict[str, Dict[str, int]]
                  ) -> Dict[str, Dict[str, int]]:
    """Get the term frequencies for the given corpus.

    Args:
        term_ids: The term-ids in the current subcategory.
        corpus: The ids of the documents making up the corpus.
        tf_base: The term frequencies of the base corpus.
        {doc_id: {term_id: freq}}
    """
    out_dict = {}
    for doc_id in corpus:
        # out_dict[doc_id] = tf_base[doc_id]
        out_dict[doc_id] = {}
        tf_doc = tf_base[doc_id]
        for term_id in tf_doc:
            if term_id in term_ids:
                out_dict[doc_id][term_id] = tf_base[doc_id][term_id]
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
    threshhold = 0.0
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
    # threshhold = 0.25 # According to TaxoGen.
    threshhold = 0.75
    for term_id in clus:
        con = term_scores[term_id][1]
        pop = term_scores[term_id][0]
        score = sqrt(con*pop)
        # if con < threshhold:
        if score < threshhold:
            concept_terms.add(term_id)
    return concept_terms


if __name__ == '__main__':
    generate_taxonomy()
