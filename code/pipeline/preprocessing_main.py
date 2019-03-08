# import os
from utility_functions import *
from ling_preprocessing import DBLPLingPreprocessor, SPLingPreprocessor
from pattern_extraction import PatternExtractor
from indexing import Indexer
from frequency_analysis import FreqAnalyzer
from consistency_tests import *
from embeddings import get_emb

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
    emb_type = config['embeddings']
    print(args)
    if not args.skip_prep:
        prep_output_dir(path_out)

    print('Start preprocessing...')

    # corpus preprocessing
    if not args.skip_lingpp:
        print(('Start tokenization, tagging, lemmatization and marking '
               'stop-words...'))
        if args.corpus == 'dblp':
            lpp = DBLPLingPreprocessor(
                path_in, path_out, path_lang_model) # , max_docs=10000)
        elif args.corpus == 'sp':
            lpp = SPLingPreprocessor(
                path_in, path_out, path_lang_model) # , max_docs=10000)
        lpp.preprocess_corpus()
        print('Done.')

    # term extraction and hearst pattern extraction
    if not args.skip_pattern_extr:
        print('Start term extraction and hearst pattern extraction...')
        te = PatternExtractor(path_out)
        te.extract()
        print('Done.')

    # indexing of corpus
    if not args.skip_idxer:
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
    if not args.skip_freq_an:
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
        print('Calculate tfidf for tokens...')
        fa.calc_tfidf('t')
        print('Calculate tfidf for lemmas...')
        fa.calc_tfidf('l')
        print('Calculate document lengths...')
        fa.calc_dl()
        print('Done.')

    if not args.skip_embeddings:
        Embedding = get_emb(emb_type)
        print('Train word2vec token embeddings...')
        path_input = os.path.join(path_out,
                                  'processed_corpus/token_idx_corpus.txt')
        embs_fname = Embedding.train(
            path_input, 'token_embeddings_global', path_out)
        print('Embeddings written to: {}'.format(embs_fname))
        print('Train word2vec lemma embeddings...')
        path_input = os.path.join(path_out,
                                  'processed_corpus/lemma_idx_corpus.txt')
        embs_fname = Embedding.train(
            path_input, 'lemma_embeddings_global', path_out)
        print('Embeddings written to: {}'.format(embs_fname))
        # embs.calc_combined_term_vecs()

        print('Start consistency testing...')
        print('test if all token terms have embeddings...')
        test_all_token_terms_have_embeddings(path_out)
        print('test if all lemma terms have embeddings...')
        test_all_lemma_terms_have_embeddings(path_out)

    # # train hyponym projector
    # hp = HyponymProjector()
    # hp.train()


if __name__ == '__main__':
    main()
