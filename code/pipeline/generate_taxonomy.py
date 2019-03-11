# import os
import json
# import re
import csv
import time
# import subprocess
# from collections import defaultdict
from math import sqrt
from numpy import median
from typing import *
from corpus import *
from embeddings import Embeddings, Word2VecE, GloVeE, get_emb
from clustering import Clustering
from score import Scorer
from utility_functions import *


# Define global variables.
node_counter = 0
idx_to_term = {}
max_depth = 3


def generate_taxonomy() -> None:
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
    global idx_to_term
    # Load cmd args and configs.
    print('Load and parse cmd args...')
    config = get_config()
    args = get_cmd_args()
    lemmatized = config['lemmatized']
    emb_type = config['embeddings']

    # Set paths.
    # path_base_corpus = config['paths'][args.location][args.corpus]['path_in']
    print('Set paths...')
    path_out = config['paths'][args.location][args.corpus]['path_out']

    if lemmatized:
        path_term_ids = os.path.join(
            path_out, 'processed_corpus/lemma_terms_idxs.txt')
        path_idx_to_term = os.path.join(
            path_out, 'indexing/idx_to_lemma.json')
        path_df = os.path.join(path_out, 'frequencies/df_lemmas.json')
        path_tf = os.path.join(path_out, 'frequencies/tf_lemmas.json')
        path_tfidf = os.path.join(
            path_out, 'frequencies/tfidf_lemmas.json')
        path_base_corpus = os.path.join(
            path_out, 'processed_corpus/pp_lemma_corpus.txt')
        path_base_corpus_ids = os.path.join(
            path_out, 'processed_corpus/lemma_idx_corpus.txt')
        path_embeddings_global = os.path.join(
            path_out, 'embeddings/embs_lemma_global_{}.vec'.format(emb_type))
    else:
        path_term_ids = os.path.join(
            path_out, 'processed_corpus/token_terms_idxs.txt')
        path_idx_to_term = os.path.join(
            path_out, 'indexing/idx_to_token.json')
        path_df = os.path.join(path_out, 'frequencies/df_tokens.json')
        path_tf = os.path.join(path_out, 'frequencies/tf_tokens.json')
        path_tfidf = os.path.join(path_out, 'frequencies/tfidf_tokens.json')
        path_base_corpus = os.path.join(
            path_out, 'processed_corpus/pp_token_corpus.txt')
        path_base_corpus_ids = os.path.join(
            path_out, 'processed_corpus/token_idx_corpus.txt')
        path_embeddings_global = os.path.join(
            path_out, 'embeddings/embs_token_global_{}.vec'.format(emb_type))

    path_dl = os.path.join(path_out, 'frequencies/dl.json')
    path_taxonomy = os.path.join(path_out, 'hierarchy/taxonomy.csv')

    tax_file = open(path_taxonomy, 'w', encoding='utf8', newline='')
    csv_writer = csv.writer(tax_file, delimiter=',')

    # Define starting variables.
    print('load term-ids...')
    term_ids = load_term_ids(path_term_ids)
    print('load idx-term mappings...')
    with open(path_idx_to_term, 'r', encoding='utf8') as f:
        idx_to_term = json.load(f)
    print('load global embeddings...')
    term_ids_to_embs_global = Embeddings.load_term_embeddings(
        term_ids, path_embeddings_global)
    print('load base corpus...')
    base_corpus = get_base_corpus(path_base_corpus)
    print('load tf-base...')
    tf_base = {}  # {doc_id: {word_id: freq}}
    with open(path_tf, 'r', encoding='utf8') as f:
        for i, doc_freqs in enumerate(f):
            doc_freqs_str = json.loads(doc_freqs.strip('\n'))
            tf_base[i] = {int(k): v for k, v in doc_freqs_str.items()}
    print('load df-base...')
    with open(path_df, 'r', encoding='utf8') as f:
        # {word_id: [doc_id1, ...]}
        df_base_str = json.load(f)
        df_base = {int(k): [int(i) for i in v] for k, v in df_base_str.items()}
    print('load tfidf-base...')
    with open(path_tfidf, 'r', encoding='utf8') as f:
        # {doc-id: {term-id: tfidf}}
        tfidf_base_str = json.load(f)
        tfidf_base = {}
        for doc_id, doc_dict in tfidf_base_str.items():
            tfidf_base[int(doc_id)] = {int(k): v for k, v in tfidf_base_str[doc_id].items()}
            # print(doc_tfidfs)
            # tfidf_base[int(doc_id)] =
    print('load dl-base...')
    with open(path_dl, 'r', encoding='utf8') as f:
        dl_str = json.load(f)
        dl = {int(k): v for k, v in dl_str.items()}
        # {doc-id: length}

    # Start recursive taxonomy generation.
    rec_find_children(term_ids_local=term_ids, term_ids_global=term_ids,
                      base_corpus=base_corpus,
                      path_base_corpus_ids=path_base_corpus_ids,
                      cur_node_id=0, level=1, df_base=df_base,
                      tf_base=tf_base, path_out=path_out, dl=dl,
                      tfidf_base=tfidf_base, cur_corpus=base_corpus,
                      csv_writer=csv_writer,
                      term_ids_to_embs_global=term_ids_to_embs_global,
                      emb_type=emb_type)

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
                      csv_writer: Any,
                      emb_type: str
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
        emb_type: The embedding type: 'Word2Vec', 'GloVe' or 'ELMo'.
    """
    if level >= max_depth:
        return None
    print(15*'-'+' level {} node {} '.format(level, cur_node_id) + 15*'-')
    msg = 'start recursion on level {} with node id {}...'.format(level,
                                                                  cur_node_id)
    print(msg)
    print('Number of candidate terms: {}'.format(len(term_ids_local)))

    m = int(len(base_corpus)/(5*level))
    print('build corpus file...')
    corpus_path = build_corpus_file(cur_corpus, path_base_corpus_ids,
                                    cur_node_id, path_out)
    print('train embeddings...')
    if level != 1:
        emb_path_local = train_embeddings(emb_type, corpus_path,
                                          cur_node_id, path_out)
        print('get term embeddings...')
        term_ids_to_embs_local = Embeddings.load_term_embeddings(
            term_ids_local, emb_path_local)
        # {id: embedding}
    else:
        term_ids_to_embs_local = term_ids_to_embs_global

    concept_terms = set()
    print('Start finding general terms...')
    i = 0
    while True:
        i += 1
        print(5*'-' + ' iteration {} '.format(i) + 5*'-')

        print('cluster terms...')
        clusters = perform_clustering(term_ids_to_embs_local)
        # Dict[int, Set[int]]

        print('get subcorpora for clusters...')
        # subcorpora = get_subcorpora(clusters, base_corpus, n, tfidf_base,
        #                             term_ids_to_embs_global)
        subcorpora = get_subcorpora(clusters, tfidf_base,
                                    term_ids_to_embs_global, m=m)
        # {label: doc-ids}

        print('get term-frequencies...')
        tf_corpus = get_tf_corpus(term_ids_local, cur_corpus, tf_base)
        # TODO: modifiy get_tf_corpus. Get the tf for all docs in cur_docs

        print('compute term scores...')
        term_scores = get_term_scores(clusters, subcorpora, dl, tf_corpus,
                                      tf_base, level)

        print('get average and median score...')
        avg_pop, avg_con, avg_total = get_avg_score(term_scores)
        median_pop, median_con, median_total = get_median_score(term_scores)
        msg_avg = ('avg popularity: {:.3f}, avg concentation: {:.3f}, '
                   'avg score: {:.3f}')
        msg_median = ('median popularity: {:.3f}, median concentation: {:.3f},'
                      ' median score: {:.3f}')
        print(msg_avg.format(avg_pop, avg_con, avg_total))
        print(msg_median.format(median_pop, median_con, median_total))

        print('remove terms from clusters...')
        clusters, general_terms = separate_gen_terms(clusters, term_scores)
        concept_terms = concept_terms.union(general_terms)
        cluster_sizes = [len(clus) for label, clus in clusters.items()]
        print('terms pushed up: {}'.format(len(general_terms)))
        print('cluster_sizes: {}'.format(cluster_sizes))
        if len(general_terms) == 0:
            break
        term_ids_to_embs_local = update_title(term_ids_to_embs_local, clusters)

    # Start preparation of next iteration.
    child_ids = get_child_ids(clusters)
    print('The child ids of {} are {}'.format(cur_node_id, str(child_ids)))
    print('write concept terms to file...')
    write_tax_to_file(cur_node_id, child_ids, concept_terms, csv_writer)

    print('start new recursion...')
    for label, clus in clusters.items():
        node_id = child_ids[label]
        subcorpus = subcorpora[label]
        rec_find_children(term_ids_local=clus, base_corpus=base_corpus,
                          path_base_corpus_ids=path_base_corpus_ids,
                          cur_node_id=node_id, level=level+1, dl=dl,
                          df_base=df_base, tf_base=tf_base, path_out=path_out,
                          tfidf_base=tfidf_base, cur_corpus=subcorpus,
                          csv_writer=csv_writer,
                          term_ids_to_embs_global=term_ids_to_embs_global,
                          term_ids_global=term_ids_global, emb_type=emb_type)


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
                      child_ids: Union[Dict[int, int], List[None]],
                      concept_terms: Set[str],
                      csv_writer: Any
                      ) -> None:
    """Write the current node with terms and child-nodes to file."""
    # Map indices to words.
    concept_terms = [idx_to_term[i] for i in concept_terms]
    child_nodes = list(child_ids.values())
    row = [cur_node_id] + child_nodes + concept_terms
    csv_writer.writerow(row)


def get_subcorpora(clusters: Dict[int, Set[str]],
                   # base_corpus: Set[str],
                   tfidf_base: Dict[str, Dict[str, float]],
                   term_ids_to_embs: Dict[str, List[float]],
                   m: Union[int, None]=None
                   ) -> Dict[int, Set[str]]:
    """Get the subcorpus for each cluster."""
    # subcorpora = {}
    doc_embeddings = get_doc_embeddings(tfidf_base, term_ids_to_embs)
    # {doc_id: embedding}
    topic_embeddings = get_topic_embeddings(clusters, term_ids_to_embs)
    # {cluster/topic_label: embedding}
    doc_topic_sims = get_doc_topic_sims(doc_embeddings, topic_embeddings)
    # {doc-id: {topic-label: sim}}
    subcorpora = get_topic_docs(doc_topic_sims, m)
    # {cluster/topic_label: {set of doc-ids}}

    # for label, clus in clusters.items():
    #     subcorpora[label] = get_relevant_docs(
    #         clus, base_corpus, n, tfidf_base,
    #         doc_embeddings,
    #         topic_embeddings[label])

    return subcorpora


def separate_gen_terms(clusters: Dict[int, Set[str]],
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


def train_embeddings(emb_type: str,
                     path_corpus: str,
                     cur_node_id: int,
                     path_out_dir: str
                     ) -> str:
    """Train word2vec embeddings on the given corpus.

    Args:
        emb_type: The type of the embeddings: 'Word2Vec', 'GloVe' or 'ELMo'.
        path_corpus: The path to the corpus file.
        cur_node_id: Id of the current node. Used for the name of the
            embedding file.
        path_out_dir: The path to the output directory.
    Return:
        The path to the embedding file:
        'embeddings/<cur_node_id>_w2v.vec'
    """
    embedding = get_emb(emb_type)
    return embedding.train(path_corpus, str(cur_node_id), path_out_dir)


# def get_embeddings(term_ids: Set[str],
#                    emb_path: str
#                    ) -> Dict[str, List[float]]:
#     """Get the embeddings for the given terms.
#
#     Args:
#         term_ids: The ids of the input terms.
#         emb_path: The path to the given embedding file.
#     Return:
#         A dictionary of the form: {term_id: embedding}
#     """
    # term_id_to_emb = {}
    # with open(emb_path, 'r', encoding='utf8') as f:
    #     next(f)
    #     for line in f:
    #         vals = line.strip(' \n').split(' ')
    #         term_id = vals[0]
    #         if term_id in term_ids:
    #             emb = [float(f) for f in vals[1:]]
    #             term_id_to_emb[term_id] = emb
    # if len(term_id_to_emb) != len(term_ids):
    #     msg_raw = ('Error! Not all term-ids have embeddings.\nNum term-ids: {}'
    #                '\nNum embeddings: {}')# \nNo embeddings for: {}')
    #     # no_emb = sorted([e for e in term_ids if e not in term_id_to_emb])
    #     msg = msg_raw.format(len(term_ids), len(term_id_to_emb)) #, no_emb)
    #     raise Exception(msg)
    # return Word2VecE.load_term_embeddings(term_ids, emb_path)

def perform_clustering(term_ids_to_embs: Dict[str, List[float]]
                       ) -> Dict[int, Set[str]]:
    """Cluster the given terms into 5 clusters.

    Args:
        term_ids_to_embs: A dictionary mapping term-ids to their
            embeddings.
    Return:
        A dictionary of mapping each cluster label to it0s cluster.
        Each cluster is a set of term-ids.
    """
    # Case less than 5 terms to cluster.
    num_terms = len(term_ids_to_embs)
    if num_terms < 5:
        clusters = {}
        for i, tid in enumerate(term_ids_to_embs):
            clusters[i] = {tid}
        return clusters

    # Case more than 5 terms to cluster.
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
    # print('Cluster sizes: {}'.format(clusters))
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
            term_ids.add(int(line.strip('\n')))
    return term_ids


def update_title(term_ids_to_embs_local: Dict[str, List[float]],
                 clusters: Dict[int, Set[str]]
                 ) -> Dict[str, List[float]]:
    """Update the term_ids_to_embs-variable (title).

    Create a new variable that only contains the terms given in
    clusters.

    Args:
        term_ids_to_embs_local: A dict mapping term_ids to embeddings.
        clusters: A dict mapping each cluster label to a cluster.
    """
    updated_title = {}
    for label, clus in clusters.items():
        for tid in clus:
            updated_title[tid] = term_ids_to_embs_local[tid]
    return updated_title


def get_avg_score(term_scores: Dict[str, Tuple[float, float]]
                  ) -> Tuple[float, float, float]:
    """Get the average popularity and concentration score."""
    pop_scores = [sc[0] for id_, sc in term_scores.items()]
    con_scores = [sc[1] for id_, sc in term_scores.items()]
    total_scores = [sqrt(p * c) for p, c in zip(pop_scores, con_scores)]
    avg_pop = float(mean(pop_scores))
    avg_con = float(mean(con_scores))
    avg_total = float(mean(total_scores))
    return avg_pop, avg_con, avg_total


def get_median_score(term_scores: Dict[str, Tuple[float, float]]
                     ) -> Tuple[float, float, float]:
    """Get the median popularity and concentration score."""
    pop_scores = [sc[0] for id_, sc in term_scores.items()]
    con_scores = [sc[1] for id_, sc in term_scores.items()]
    total_scores = [sqrt(p * c) for p, c in zip(pop_scores, con_scores)]
    median_pop = float(median(pop_scores))
    median_con = float(median(con_scores))
    median_total = float(median(total_scores))
    return median_pop, median_con, median_total


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


# def get_df_corpus(term_ids: Set[str],
#                   corpus: Set[str],
#                   df_base: Dict[str, int]
#                   ) -> Dict[str, int]:
#     """Get the document frequencies for a given corpus and term-ids.
#
#     Args:
#         term_ids: The ids of the given terms.
#         corpus: The ids of the documents making up the corpus.
#         df_base: The document frequencies of the base corpus.
#         {term_id: [doc_id1, ...]}
#     Return:
#         {term_id: num_docs}
#     """
#     out_dict = defaultdict(int)
#     print(100*'-')
#     print(type(df_base))
#     for term_id in term_ids:
#         print(type(df_base[term_id]))
#         for doc_id in df_base[term_id]:
#             if doc_id in corpus:
#                 out_dict[term_id] += 1
#     return out_dict


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
    return set([i for i in range(get_num_docs(path_base_corpus))])


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
    threshhold = 0.1
    for term_id in clus:
        pop = term_scores[term_id][0]
        con = term_scores[term_id][1]
        score = sqrt(pop*con)
        # if con < threshhold:
        if score < threshhold:
            concept_terms.add(term_id)
    return concept_terms


if __name__ == '__main__':
    start_time = time.time()
    generate_taxonomy()
    end_time = time.time()
    time_used = end_time - start_time
    print('Time used: {}'.format(time_used))
    print('Done.')
