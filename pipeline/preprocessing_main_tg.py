import os
import json
import pickle
from collections import defaultdict
from typing import Dict, Any, DefaultDict, Union, Tuple
from utility_functions import get_config, get_cmd_args, prep_output_dir
from indexing import Indexer
from frequency_analysis import FreqAnalyzer
from embeddings import get_emb
from document_embeddings import DocEmbedder


doc_distr_type = DefaultDict[int, Union[Tuple[int, int], int]]
term_distr_type = DefaultDict[int, doc_distr_type]
term_distr_base: term_distr_type = defaultdict(dict)

"""
Preprocess dblp input based on TaxoGen's preprocessing.

NOTE: Hierarchy files are not included atm in the pipeline.

Input-Files:
index.txt: maps each word to the doc-ids it appears in.
keyword_cnt.txt: counts how many times a word appears a document.
keywords.txt: one keyword on each line.
papers.txt: One Paper on each line. Already tokenized.
    MWE are concatenated by '_'.

Steps:
1. Copy files:
    a) papers.txt -> pp_token_corpus.txt (insert newlines)
    b) keywords.txt -> processed_corpus/token_terms.txt
2. Index words with idxer.
    a) Go through papers.txt to create idx-term-mappings/ token_idx_corpus.txt.
    b) Create token_terms_idxs.txt.
3. Frequency Analysis based on index.txt and keywords.txt
    a) Create tf_tokens.json based on keyword_cnt.txt.
    b) Create df_tokens.json based on index.txt
    c) Create tfidf_tokens.json based on tf_tokens.json and df_tokens.json.
    d) Create dl.json based on token_idx_corpus.txt.
    e) Create term_distr_tokens.pickle based on all before.
4. Train embeddings.
    a) Train token embeddings.
5. Calculate Doc-Embeddings.
    a) Calculate token_doc_embeddings.

python3 preprocessing_main_tg.py -c dblp -l server -spd -sl -spe-sid-sfa-se-swe
"""


config_type = Dict[str, Any]


def papers_to_pp_token_corpus(config: config_type,
                              location: str,
                              corpus: str
                              ) -> None:
    """Copy papers.txt and insert new-lines."""
    path_papers = config['paths'][location][corpus]['path_papers']
    path_out = config['paths'][location][corpus]['path_out']
    path_pp_token_corpus = os.path.join(
        path_out, 'processed_corpus/pp_token_corpus.txt')
    with open(path_papers, 'r', encoding='utf8') as fin:
        with open(path_pp_token_corpus, 'w', encoding='utf8') as fout:
            for line in fin:
                fout.write(line+'\n')


def copy_keywords_to_terms(config: config_type,
                           location: str,
                           corpus: str
                           ) -> None:
    """Copy keywords.txt to terms.txt."""
    path_keywords = config['paths'][location][corpus]['path_keywords']
    path_out = config['paths'][location][corpus]['path_out']
    path_terms = os.path.join(path_out, 'processed_corpus/token_terms.txt')
    cmd = 'cp {} {}'.format(path_keywords, path_terms)
    os.system(cmd)


def main():
    args = get_cmd_args()
    location = args.location
    corpus = args.corpus
    config = get_config()
    path_out = config['paths'][location][corpus]['path_out']
    emb_type = config['embeddings']

    if not args.skip_prep:
        prep_output_dir(path_out)

    # Copy TG files into dir-system.
    papers_to_pp_token_corpus(config, location, corpus)
    copy_keywords_to_terms(config, location, corpus)

    # Index corpus.
    if not args.skip_idxer:
        # print('Start indexing...')
        idxer = Indexer(path_out)
        # idxer.index_tokens()
        # print('Finished indexing.')
        print('Start building subtoken index...')
        idxer.build_token_contains()
        print('Finished building subtoken index.')

    # Frequency analysis.
    if not args.skip_freq_an:
        print('Start frequency analysis for tf, df and dl...')
        fa = FreqAnalyzer(path_out)
        print('Calculate token term frequencies...')
        fa.calc_tf('t')
        print('Calculate token document frequencies...')
        fa.calc_df('t')
        print('Calculate tfidf for tokens...')
        fa.calc_tfidf('t')
        print('Calculate document lengths...')
        fa.calc_dl()
        print('Finished frequency analysis.')

    if not args.skip_embeddings:
        emb_types = ['Word2Vec', 'GloVe', 'ELMo']
        for etype in emb_types:
            Embedding = get_emb(etype)
            print('Train {} token embeddings...'.format(etype))
            path_input = os.path.join(path_out,
                                      'processed_corpus/token_idx_corpus.txt')
            embs_fname = Embedding.train(
                path_input, 'embs_token_global_'+etype, path_out)
            print('{} embeddings written to: {}'.format(etype, embs_fname))

    if not args.skip_doc_embs:
        print('Calculating document embeddings...')
        doc_embedder = DocEmbedder(path_out, emb_type)
        doc_embedder.embed_token_docs()
        print('Finished document embeddings.')

    if not args.skip_word_distr:
        print('Create term distributions pickle file...')

        path_tf = os.path.join(path_out, 'frequencies/tf_tokens.json')
        path_tfidf = os.path.join(path_out, 'frequencies/tfidf_tokens.json')
        path_dl = os.path.join(path_out, 'frequencies/dl.json')
        path_term_distr = os.path.join(
            path_out, 'frequencies/term_distr_tokens.json')

        # Load frequencies.
        with open(path_tf, 'r', encoding='utf8') as f_tf:
            tf_base = json.load(f_tf)
            with open(path_tfidf, 'r', encoding='utf8') as f_tfidf:
                tfidf_base = json.load(f_tfidf)
                with open(path_dl, 'r', encoding='utf8') as f_dl:
                    dl_base = json.load(f_dl)

        # Create term_distr.
        for doc_id in tfidf_base:
            for word_id in tf_base[doc_id]:
                tf = tf_base[doc_id][word_id]
                tfidf = tfidf_base[doc_id][word_id]
                term_distr_base[int(doc_id)][int(word_id)] = (tf, tfidf)
            term_distr_base[int(doc_id)][-1] = dl_base[doc_id]

        # Dump term_distr.
        with open(path_term_distr, 'wb') as f:
            pickle.dump(term_distr_base, f)


if __name__ == '__main__':
    main()
