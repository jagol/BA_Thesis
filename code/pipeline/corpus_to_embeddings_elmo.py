from typing import *
import time
import re
import os
import json
import multiprocessing as mp
from collections import defaultdict
from numpy import mean
from embeddings import ElmoE
from utility_functions import get_docs, get_num_docs, split_corpus, concat_corpus


output = mp.Queue()


# def main():
#     elmo = ElmoE()
#     path_in = 'output/dblp/processed_corpus/pp_lemma_corpus_1000.txt'
#     path_out = 'output/dblp/embeddings/elmo_lemma_embeddings.txt'
#     with open(path_out, 'w', encoding='utf8') as f:
#         docs_embs = []
#         start_time = time.time()
#         for i, doc in enumerate(get_docs(path_in)):
#             print(i)
#             doc_embs = []
#             for sent in doc:
#                 embs = elmo.get_embeddings(sent)
#                 doc_embs.append(embs)
#             docs_embs.append(doc_embs)
#
#             if i % 500 == 0:
#                 docs_str = docs_to_string(docs_embs)
#                 f.write(docs_str)
#
#             if i >= 10:
#                 break
#
#         docs_str = docs_to_string(docs_embs)
#         f.write(docs_str)
#
#     time_stamp = time.time()
#     time_passed = time_stamp - start_time
#     print(time_passed)
#     exp_time = time_passed * 400000
#     minutes = exp_time / 60
#     hours = minutes / 60
#     days = hours / 24
#     print(exp_time)
#     print(minutes)
#     print(hours)
#     print(days)


def embed_corpus_terms(path: str, level: str, num_processes: int):
    """Create elmo embeddings for a given corpus with parallel processing.

    Args:
        path: Path to output directory.
        level: 't' if tokens, 'l' if lemmas.
        num_processes: The number of processes.
    """
    # path_in = 'output/dblp/processed_corpus/pp_lemma_corpus_1000.txt'
    path_out = os.path.join(path, 'embeddings/')
    if level == 't':
        path_in = os.path.join(path,
                               'processed_corpus/pp_token_corpus.txt')
        path_corpus_embs = os.path.join(path_out,
                                        'corpus_elmo_{}.emb'.format('tokens'))
    elif level == 'l':
        path_in = os.path.join(path,
                     'processed_corpus/pp_lemma_corpus.txt')
        path_corpus_embs = os.path.join(path_out,
                                        'corpus_elmo_{}.emb'.format('lemmas'))
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
    print('Concatenating corpus files...')
    merge_dicts(fnames_emb, './all_corpus_term_embs.json')
    print('Cleaning temporary files...')
    # os.system('rm {}*.emb; rm {}*.txt.split'.format(path_out, path_out))
    print('Done.')


def parallel_embed_terms(path: str, fnames: List[str], start_nums, level: str):
    """Process the given files in parallel.

    Args:
        path: Path to output directory.
        fnames: List of names of input files.
        start_nums: Sarting numbers of doc id for each file.
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
    processes = [mp.Process(target=embed_terms, args=(path_term_to_idxs, path_in, path_out, start_num)) for path_in, path_out, start_num in args]

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
    """
    print('Loading terms...')
    with open(path_term_to_idxs, 'r', encoding='utf8') as f:
        terms_to_idxs = json.load(f)

    print('Instanciating ELMo...')
    elmo = ElmoE()
    term_embs_per_doc = defaultdict(lambda: defaultdict(list))

    for i, doc in enumerate(get_docs(path_in)):
        doc_id = start_num + i
        print('processing {}...'.format(doc_id))
        print('doc_id and doc', doc_id, doc)
        for sent in doc:
            sent_terms = []
            for j in range(len(sent)):
                word = sent[j]
                if word in terms_to_idxs:
                    term_idx = terms_to_idxs[word]
                    sent_terms.append((term_idx, word.split('_'), j))
            print('doc-id and sent-terms:', doc_id, sent_terms)
            if sent_terms:
                prepped_sent, term_idxs = prepare_sentence(sent, sent_terms)
                print(prepped_sent, term_idxs)
                embs = elmo.get_embeddings(prepped_sent)
                for h in range(len(sent_terms)):
                    term = sent_terms[h]
                    term_emb = get_term_emb(embs, term_idxs[h])
                    # term_emb = [float(f) for f in embs[term[1]]]
                    term_idx = term[0]
                    term_embs_per_doc[term_idx][doc_id].append(term_emb)
        if i > 100:
            break

    with open(path_out, 'w', encoding='utf8') as f:
        json.dump(term_embs_per_doc, f)

    output.put('Done')


def get_term_emb(embs: List[List[float]],
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
    for i in range(len(sent_terms))[::-1]:
        term = sent_terms[i]
        term_idx = term[2]
        term_words = [t + '__' + str(i) for t in term[1]]
        sent[term_idx] = term_words

    # Flatten list.
    flat_sent = []
    for w in sent:
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
        with open(fp, 'r', encoding='utf8') as f:
            cur_dict = json.load(f)
            print(cur_dict.keys())
            for term_id in cur_dict:
                if term_id in out_dict:
                    cur_term_dict = cur_dict[term_id]
                    out_dict_term = out_dict[term_id]
                    for doc_id in cur_term_dict:
                        if doc_id in out_dict_term:
                            print(
                                'Two files have the same doc_ids...there is something wrong!')
                        else:
                            out_dict_term[doc_id] = cur_term_dict[doc_id]
                else:
                    out_dict[term_id] = cur_dict[term_id]
    with open(path_out, 'w', encoding='utf8') as f:
        json.dump(out_dict, f)


# def embed_docs(path_in: str, path_out: str) -> None:
#     """Create elmo embeddings for given corpus.
#
#     Args:
#         path_in: Path to input file.
#         path_out: Path to output file.
#     """
#     elmo = ElmoE()
#     docs_embs = []
#     with open(path_out, 'w', encoding='utf8') as f:
#         for i, doc in enumerate(get_docs(path_in)):
#             doc_embs = []
#             print('processing {}'.format(i))
#             for sent in doc:
#                 embs = elmo.get_embeddings(sent)
#                 embs = [list(emb) for emb in embs]
#                 print(embs)
#                 doc_embs.append(embs)
#             docs_embs.append(doc_embs)
#
#             if i % 500 == 0:
#                 docs_str = docs_to_string(docs_embs)
#                 f.write(docs_str)
#                 docs_embs = []
#
#             if i >= 10:
#                 break
#         # s = json.dumps(dict(enumerate(docs_embs)))
#         # docs_str = docs_to_string(docs_embs)
#         # f.write(s)
#
#     output.put('Done')
#
#
# def docs_to_string(docs: List[List[List[List[float]]]]) -> str:
#     """Return all documents in documents as one string."""
#     corpus_as_str = ''
#     for doc in docs:
#         for sent in doc:
#             sent_str = ''
#             for emb in sent:
#                 # emb_str = array2string(emb, max_line_width = 1000) + '\t'
#                 emb_str = str(emb)
#                 sent_str += emb_str
#             line = sent_str + '\n'
#             corpus_as_str += line
#         corpus_as_str += '\n'
#     return corpus_as_str


if __name__ == '__main__':
    # embed_corpus('mnt/storage/harlie/users/jgoldz/output/dblp/', 'l', 5)
    embed_corpus_terms('/mnt/storage/harlie/users/jgoldz/output/dblp', 'l', 5)
    # elmo = ElmoE()
    # sent = ["This", "is", "an", "example", "."]
    # embs = elmo.get_embeddings(sent)
    # print(embs)
    # print(len(embs))
    # print(type(embs))
    # print(len(embs[0]))
    # print(type(embs[0]))
    # print(30*'-')
    # x = [list(emb) for emb in embs]
    # print(len(x))
    # print(type(x))
    # print(len(x[0]))
    # print(type(x[0]))
