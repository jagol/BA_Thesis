from typing import *
import time
import re
import os
import pickle
import json
import multiprocessing as mp
from collections import defaultdict
from numpy import mean
from embeddings import ElmoE
from utility_functions import get_docs, split_corpus


"""
Use this script to extract elmo embedding for a list of given terms in 
a corpus.
"""


output = mp.Queue()


def embed_corpus_terms(path: str, level: str, num_processes: int):
    """Create elmo embeddings for a given corpus with parallel processing.

    Args:
        path: Path to output directory.
        level: 't' if tokens, 'l' if lemmas.
        num_processes: The number of processes.
    """
    path_out = os.path.join(path, 'embeddings/')
    if level == 't':
        path_in = os.path.join(path,
                               'processed_corpus/pp_token_corpus.txt')
        # path_corpus_embs = os.path.join(
        #     path_out, 'corpus_elmo_{}.emb'.format('tokens'))
    elif level == 'l':
        path_in = os.path.join(path,
                               'processed_corpus/pp_lemma_corpus.txt')
        # path_corpus_embs = os.path.join(
        #     path_out, 'corpus_elmo_{}.emb'.format('lemmas'))
    else:
        raise Exception("Error! Choose 't' or 'l'")
    print('Split corpus...')
    fnames, start_nums = split_corpus(path_in, path_out, num_processes)
    print('fnames:', fnames)
    print('starting-numbers:', start_nums)
    msg = 'Get embeddings for corpus, {} parallel processes...'
    print(msg.format(num_processes))
    start_time = time.time()
    print('start-time:', start_time)
    fnames_emb = parallel_embed_terms(path, fnames, start_nums, level)
    end_time = time.time()
    print('end-time:', end_time)
    time_passed = end_time-start_time
    print('time-passed:', time_passed)
    # print('Concatenating corpus files...')
    # merge_dicts(fnames_emb, './elmo_embeddings_l2_{}.pickle'.format(level))
    # print('Cleaning temporary files...')
    # os.system('rm {}*.emb; rm {}*.txt.split'.format(path_out, path_out))
    print('Done.')


def parallel_embed_terms(path: str, fnames: List[str], start_nums, level: str):
    """Process the given files in parallel.

    Args:
        path: Path to output directory.
        fnames: List of names of input files.
        start_nums: Starting numbers of doc id for each file.
        level: 't' if tokens, 'l' if lemmas.
    Return:
        The paths to the outputted files.
    """
    fpaths_in = [os.path.join(path, 'embeddings/'+fname) for fname in fnames]
    fpaths_out = [fpath + '.emb.json' for fpath in fpaths_in]
    args = zip(fpaths_in, fpaths_out, start_nums)
    if level == 't':
        path_term_to_idxs = os.path.join(path, 'indexing/token_to_idx.json')
    elif level == 'l':
        path_term_to_idxs = os.path.join(path, 'indexing/lemma_to_idx.json')
    else:
        raise Exception("Error! Choose 't' or 'l'")
    processes = [mp.Process(target=embed_terms,
                            args=(path_term_to_idxs, path_in, path_out,
                                  start_num))
                 for path_in, path_out, start_num in args]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for _ in processes]
    print(results)

    return fpaths_out


def embed_terms(path_term_to_idxs: str,
                path_in: str,
                path_out: str,
                start_num: int) -> None:
    """Create elmo embeddings for given corpus.

    Args:
        path_term_to_idxs: Path to the terms.
        path_in: Path to input file.
        path_out: Path to output file.
        start_num: The starting doc id.

    term_embs_per_doc: {term_idx: {doc_idx: list of embeddings}}
    """
    dump_counter = 0
    print('Loading terms...')
    with open(path_term_to_idxs, 'r', encoding='utf8') as f:
        terms_to_idxs = json.load(f)

    print('Instanciating ELMo...')
    elmo = ElmoE()
    term_embs_per_doc = {}

    for i, doc in enumerate(get_docs(path_in)):
        doc_id = start_num + i
        # print(30*'-')
        print('processing {}...'.format(doc_id))
        # print('doc_id: {}, doc: {}'.format(doc_id, doc))
        for sent in doc:
            sent_terms = []
            for j in range(len(sent)):
                word = sent[j]
                if word in terms_to_idxs:
                    term_idx = terms_to_idxs[word]
                    sent_terms.append((term_idx, word.split('_'), j))
            # print('doc-id: {}, sent-terms: {}'.format(doc_id, sent_terms))
            if sent_terms:
                # prepped_sent, term_idxs = prepare_sentence(sent, sent_terms)
                # print('prepared_sent: {}, term_idxs: {}'.format(prepped_sent,
                #                                                 term_idxs))
                # print('sent:', sent)
                assert isinstance(sent, list)
                assert isinstance(sent[0], str)
                embs = elmo.get_embeddings(sent, mode=1)
                for k in range(len(sent_terms)):
                    # term_emb = get_term_emb(embs, term_idxs[h])
                    # term_emb = [float(f) for f in embs[term[1]]]
                    term_idx_in_sent = sent_terms[k][2]
                    term_emb = embs[term_idx_in_sent]
                    term_idx = sent_terms[k][0]
                    if term_idx not in term_embs_per_doc:
                        term_embs_per_doc[term_idx] = {}
                    if doc_id not in term_embs_per_doc[term_idx]:
                        term_embs_per_doc[term_idx][doc_id] = []
                    term_embs_per_doc[term_idx][doc_id].append(term_emb)

        if i % 5000 == 0:
            fpath = path_out + str(dump_counter)
            print('Write embeddings to file at: {}...'.format(fpath))
            with open(fpath, 'wb') as f:
                pickle.dump(term_embs_per_doc, f)
            print('Finished writing embeddings.')
            term_embs_per_doc = {}
            dump_counter += 1

    fpath = path_out + str(dump_counter)
    print('Write embeddings to file at: {}...'.format(fpath))
    with open(fpath, 'wb') as f:
        pickle.dump(term_embs_per_doc, f)
    print('Finished writing embeddings.')
    term_embs_per_doc = {}

    output.put('Done')


def get_term_emb(embs: List[Iterator[float]],
                 term: List[int]
                 ) -> List[float]:
    """Get the embedding for the given term.

    For multiword terms, get the average embedding.

    Args:
        embs: A list of word embeddings.
        term: The indices of a term's words in the current sentence.
    Return:
        The embedding of the term.
    """
    term_embs = [embs[i] for i in term]
    mean_emb = mean(term_embs, 0)
    return [float(f) for f in mean_emb]


def prepare_sentence(sent: List[str],
                     sent_terms: List[Tuple[str, List[str], int]]
                     ) -> Tuple[List[str], Dict[int, List[int]]]:
    """Make sent where the words of terms are split into single words.

    Args:
        sent: The original sentence.
        sent_terms: A list of terms in the sentence. Each term in the
        sentence is represented by a Tuple in the form
        (term-id, list of words in term, index of term in the sentence)
    Return:
        1. The sentence, but all multiword terms are split into single
        words again.
        2. Dict of term-indices of the form: {i: [indices_in_sent]}
        i is the position of the term in sent_terms.
    """
    sent_widx = [w for w in sent]
    for i in range(len(sent_terms))[::-1]:
        term = sent_terms[i]
        term_idx = term[2]
        term_words = [t + '__' + str(i) for t in term[1]]
        sent_widx[term_idx] = term_words

    # Flatten list.
    flat_sent = []
    for w in sent_widx:
        if isinstance(w, list):
            for v in w:
                flat_sent.append(v)
        else:
            flat_sent.append(w)

    prepped_sent = []
    indices_in_sent = defaultdict(list)
    pattern = re.compile(r'__(\d+)')
    for i in range(len(flat_sent)):
        w = flat_sent[i]
        # if w[-1].isdigit():
        match = re.search(pattern, w)
        if match:
            term_idx = int(match.group(1))
            indices_in_sent[term_idx].append(i)
            w = re.sub(pattern, '', w)
            prepped_sent.append(w)
        else:
            prepped_sent.append(w)

    return prepped_sent, indices_in_sent


def merge_dicts(fpaths: List[str], path_out: str) -> None:
    """Merge n dictionaries to one dictionary and write to file.

    Args:
        fpaths: A list of paths to json files that contain the dictionaries.
        path_out: Path to the output json file.
    Return:
        Write a dictionary with the structure
        Dict[int, Dict[int, List[float]]] to file.
    """
    out_dict = {}
    for fp in fpaths:
        with open(fp, 'rb') as f:
            cur_dict = pickle.load(f)
            # print(cur_dict.keys())
            for term_id in cur_dict:
                if term_id in out_dict:
                    cur_term_dict = cur_dict[term_id]
                    out_dict_term = out_dict[term_id]
                    for doc_id in cur_term_dict:
                        if doc_id in out_dict_term:
                            print(('Two files have the same doc_ids...'
                                   'there is something wrong!'))
                        else:
                            out_dict_term[doc_id] = cur_term_dict[doc_id]
                else:
                    out_dict[term_id] = cur_dict[term_id]
    with open(path_out, 'wb') as f:
        pickle.dump(out_dict, f)


if __name__ == '__main__':
    embed_corpus_terms('/mnt/storage/harlie/users/jgoldz/output/dblp', 't', 22)
