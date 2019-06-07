import os
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Any, DefaultDict
import numpy as np


def add_embs(fpath: str,
             emb_dict: Dict[int, List[np.array]],
             term_ids: Set[int]
             ) -> None:
    """Add the embeddings from <fpath> to the embeddings in emb_dict.

    Args:
        fpath: Path to a pickle file of the form:
            {term_id: {doc_id: list of embeddings}}
        emb_dict: The dictionary to which the embeddings are added.
        term_ids: The set of term ids.
    """
    # print(fpath)
    with open(fpath, 'rb') as f:
        new_embs = pickle.load(f)
    for term_id in new_embs:
        if term_id not in term_ids:
            continue
        if term_id not in emb_dict:
            emb_dict[term_id] = []
        for doc_id in new_embs[term_id]:
            # if doc_id not in avg_emb_dict[term_id]:
            #     avg_emb_dict[term_id][doc_id] = []
            # avg_emb_dict[term_id][doc_id].extend(new_embs[term_id][doc_id])
            # We can assume that a doc_id will only occur once for a
            # term-id, so there is no risk in overwriting existing lists.
            emb_dict[term_id].extend(new_embs[term_id][doc_id])


def get_subsets(set_in: Set[Any], n: int) -> List[Set[Any]]:
    """Split a set into n evenly sized subsets.

    Args:
        set_in: The input set.
    Return:
        A list of evenly sized output sets.
    """
    set_as_array = np.array(list(set_in))
    sub_arrays = np.array_split(set_as_array, n)
    return [set(s) for s in sub_arrays]


def compute_avg_elmo_embs(tisubset: Set[int],
                          num_files: int,
                          fpaths_in: List[str],
                          path_out: str,
                          i: int
                          ) -> str:
    """Compute the avg elmo embeddings for the given term ids.

    Dump results to pickle file.

    Args:
        tisubset: Subset of term-ids.
        num_files: The total number of files containing term sensitive
            embeddings.
        fpaths_in: A list to the input embeddings files.
        path_out: The path to the output directory.
        i: Current partition.
    Return:
        Path to the dumped pickle file.
    """
    context_emb_dict = {}  # {term_id: list of embeddings}
    avg_emb_dict = {}  # {term_id: embedding}
    avg_num = defaultdict(int)  # {term_id: number of embs already averaged}
    for j, fp in enumerate(fpaths_in):
        print('Adding embeddings from file {} of {}...\r'.format(
            j, num_files), end='\r')
        add_embs(fp, context_emb_dict, tisubset)
        if j % 50 == 0 and j != 0:
            print('Computing the average term-embeddings...')
            avgerage_emb_dict(context_emb_dict, avg_emb_dict, avg_num)
            context_emb_dict = {}
    avgerage_emb_dict(context_emb_dict, avg_emb_dict, avg_num)
    # print('Adding embeddings from file {} of {}...'.format(j, num_files))
    # average embeddings
    # print('Computing average term-doc-embeddings...')
    # for term_id in emb_dict:
    #     for doc_id in emb_dict[term_id]:
    #         emb_dict[term_id][doc_id] = np.mean(emb_dict[term_id][doc_id],
    #                                             axis=0)
    # print('Computing average term-embeddings...')
    # for term_id in emb_dict:
    #     # term_embs = list(emb_dict[term_id].values())
    #     avg_term_emb = np.mean(emb_dict[term_id], axis=0)
    #     avg_emb_dict[term_id] = avg_term_emb
    fout = os.path.join(path_out, 'embs_token_ELMo_avg_{}.pickle'.format(i))
    with open(fout, 'wb') as f:
        pickle.dump(avg_emb_dict, f)
    return fout


def avgerage_emb_dict(context_emb_dict: Dict[int, List[np.array]],
                      avg_emb_dict: Dict[int, np.array],
                      avg_num: DefaultDict[int, int]
                      ) -> None:
    """Average the embeddings already in the dictionary.

    Args:
        context_emb_dict: The dict with the embeddings to average.
        avg_emb_dict: Dict with averaged embeddings.
        avg_num: {term_id: number of embeddings already averaged}
    """
    for term_id in context_emb_dict:
        term_embs = context_emb_dict[term_id]
        num_new_embs = len(term_embs)
        num_old_embs = avg_num[term_id]
        new_avg = np.mean(term_embs, axis=0)
        if term_id not in avg_emb_dict:
            avg_emb_dict[term_id] = np.zeros(1024, dtype=np.float32)
        old_avg = avg_emb_dict[term_id]
        total_avg = np.average([old_avg, new_avg],
                               weights=[num_old_embs, num_new_embs],
                               axis=0).astype('float32')
        avg_emb_dict[term_id] = total_avg
        avg_num[term_id] += num_new_embs


def combine_files(file_paths: List[str], path_out: str) -> None:
    """Combine all pickle embedding files into one file.

    Args:
        file_paths: List of file paths.
        path_out: Path to the output directory.
    """
    all_avg_embs = {}
    for fp in file_paths:
        with open(fp, 'rb') as f:
            partial_avg_embs = pickle.load(f)
            all_avg_embs.update(partial_avg_embs)
    fout = os.path.join(path_out, 'embs_token_ELMo_avg.pickle')
    with open(fout, 'wb') as f:
        pickle.dump(all_avg_embs, f)


def main(path_in: str, path_out: str) -> None:
    """Main function to calculate average elmo embeddings.

    Split terms into subsets and compute the average for each subset.

    Args:
        path_in: Path to a directory containing pickle files. These
            pickle files contain dictionaries of the form:
            {term_id: {doc_id: list of embeddings}}
        path_out: Path to a pickle file to which the averaged embeddings
            are written to in the form:
            {term_id: {doc_id: list of embeddings}}
    """
    fpaths_in = [os.path.join(path_in, fname) for fname in os.listdir(path_in)]
    fpaths_in = [fp for fp in fpaths_in if '.txt.split.emb.json' in fp]
    path_term_ids = ('/mnt/storage/harlie/users/jgoldz/output/dblp/'
                     'processed_corpus/token_terms_idxs.txt')
    term_ids = []
    with open(path_term_ids, 'r', encoding='utf8') as f:
        for line in f:
            term_ids.append(int(line.strip('\n')))
    term_ids = set(term_ids)
    term_ids_subsets = get_subsets(term_ids, 15)
    num_files = len(fpaths_in)
    out_files = []
    for i, tisubset in enumerate(term_ids_subsets):
        print('Processing subset {} of {}...'.format(i, len(term_ids_subsets)))
        outf = compute_avg_elmo_embs(tisubset, num_files, fpaths_in,
                                     path_out, i)
        out_files.append(outf)
    combine_files(out_files, path_out)


if __name__ == '__main__':
    path_emb_dir = '/mnt/storage/harlie/users/jgoldz/output/dblp/embeddings/'
    path_out_dir = '/mnt/storage/harlie/users/jgoldz/output/dblp/embeddings/'
    main(path_emb_dir, path_out_dir)