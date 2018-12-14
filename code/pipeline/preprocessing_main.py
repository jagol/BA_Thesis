import os
from utility_functions import *
from ling_preprocessing import DBLPLingPreprocessor, SPLingPreprocessor
from pattern_extraction import PatternExtractor
from indexing import Indexer
from frequency_analysis import FreqAnalyzer
from embeddings import train_fasttext

"""
Main script to execute all preprocessing steps.

The preprocessing steps are:
- setting up the output directories
- linguistic preprocessing (tokenizing, tagging, lemmatizing, stop words)
- term extraction
- extraction of hearst patterns
- indexing of the corpus
- calculating term frequencies (tf)
- calculating document frequencies (df)
- calculating document lengths
- embedding training on the entire corpus (vs local embeddings)
- precomputing term vectors
- training hyponym projection model
"""


def main():
    # setting up paths and directories
    args = get_cmd_args()
    config = get_config()
    path_out = get_path_out(args, config)
    path_in = get_path_in(args, config)
    path_lang_model = config['paths'][args.location]['path_lang_model']
    prep_output_dir(path_out)

    print('Start preprocessing...')

    # corpus preprocessing
    print(
        'Start tokenization, tagging, lemmatization and marking stop-words...')
    if args.corpus == 'dblp':
        lpp = DBLPLingPreprocessor(
            path_in, path_out, path_lang_model, max_docs=1000)
    elif args.corpus == 'sp':
        lpp = SPLingPreprocessor(
            path_in, path_out, path_lang_model, max_docs=1000)
    lpp.preprocess_corpus()
    print('Done.')

    # term extraction and hearst pattern extraction
    print('Start term extraction and hearst pattern extraction...')
    te = PatternExtractor(path_out)
    te.extract()
    print('Done.')

    # indexing of corpus
    print('Start indexing...')
    idxer = Indexer(path_out)
    print('index tokens...')
    idxer.index_tokens()
    print('index lemmas...')
    idxer.index_lemmas()
    print('convert relations to index...')
    idxer.hierarch_rels_to_lemma_idx()
    print('Done.')

    # analyze lemma frequencies
    print('Start frequency analysis for tf, df and dl...')
    fa = FreqAnalyzer(path_out)
    print('Calculate token term frequencies...')
    fa.calc_tf('t')
    print('Calculate lemma term frequencies...')
    fa.calc_tf('l')
    print('Calculate token document frequencies...')
    fa.calc_df('t')
    print('Calculate lemma document frequencies...')
    fa.calc_df('l')
    print('Calculate document lengths...')
    fa.calc_dl()
    print('Done.')

    # train embeddings
    # os.system('cd ./output/dblp/processed_corpus')
    os.system('head -n 1000 ./output/dblp/processed_corpus/pp_token_corpus.txt > ./output/dblp/processed_corpus/pp_token_corpus_1000.txt')
    os.system('head -n 1000 ./output/dblp/processed_corpus/pp_lemma_corpus.txt > ./output/dblp/processed_corpus/pp_lemma_corpus_1000.txt')
    print('Train fasttext token embeddings...')
    train_fasttext(path_out, 't')
    print('Train fasttext lemma embeddings...')
    train_fasttext(path_out, 'l')
    # embs.calc_combined_term_vecs()

    # # train hyponym projector
    # hp = HyponymProjector()
    # hp.train()


if __name__ == '__main__':
    main()
