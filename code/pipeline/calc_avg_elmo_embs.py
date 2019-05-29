import os
import pickle
from typing import Dict, List
import numpy as np


def calc_avg_elmo_embs(path_in: str, path_out: str) -> None:
    """Calculate average elmo embeddings over documents and sentences.

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
    avg_emb_dict = {}
    for fpath in fpaths_in:
        add_embs(fpath, avg_emb_dict)
    with open(path_out, 'wb') as f:
        pickle.dump(avg_emb_dict, f)


def add_embs(fpath: str,
             avg_emb_dict: Dict[int, Dict[int, List[np.array]]]
             ) -> None:
    """Add the embeddings from <fpath> to the embeddings in avg_emb_dict.

    Args:
        fpath: Path to a pickle file of the form:
            {term_id: {doc_id: list of embeddings}}
        avg_emb_dict: The dictionary to which the embeddings are added.
    """
    print(fpath)
    with open(fpath, 'rb') as f:
        new_embs = pickle.load(f)
    for term_id in new_embs:
        if term_id not in avg_emb_dict:
            avg_emb_dict[term_id] = {}
        for doc_id in avg_emb_dict[term_id]:
            # if doc_id not in avg_emb_dict[term_id]:
            #     avg_emb_dict[term_id][doc_id] = []
            # avg_emb_dict[term_id][doc_id].extend(new_embs[term_id][doc_id])
            # We can assume that a doc_id will only occur once for a
            # term-id, so there is no risk in overwriting existing lists.
            avg_emb_dict[term_id][doc_id] = new_embs[term_id][doc_id]


if __name__ == '__main__':
    path_emb_dir = '/mnt/storage/harlie/users/jgoldz/output/dblp/embeddings/'
    path_outf = ('/mnt/storage/harlie/users/jgoldz/output/dblp/embeddings/'
                 'embs_token_ELMo_avg.pickle')
    calc_avg_elmo_embs(path_emb_dir, path_outf)
