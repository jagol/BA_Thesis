from utility_functions import *
from ling_preprocessing import LingPreprocessor
from term_extraction import TermExtractor
from indexing import indexer
from frequency_analysis import FreqAnalyzer
from embeddings import Embeddings

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
    config = get_config
    path_out = get_path_out(args, config)
    path_in = get_path_in(args, config)
    path_lang_model = config['path_lang_model']
    prep_output_dir(path_out)

    # corpus preprocessing
    lpp = LingPreprocessor(path_in, path_out, path_lang_model)
    lpp.preprocess_corpus()

    # term extraction
    te = TermExtractor()
    te.extract_nps()

    # hypernym extraction with hearst
    he = HearstHypernymExtractor
    he.extract_hypernyms()

    # indexing of corpus
    idxer = Indexer()
    idxer.index_tokens()
    idxer.index_lemmas()

    # analyze lemma frequencies
    fa = FreqAnalyzer(level='lemma')
    fa.calc_tf()
    fa.calc_df()
    fa.calc_dl()

    # train embeddings
    embs = Embeddings('fasttext')
    embs.train()
    embs.calc_term_vecs()
    embs.calc_combined_term_vecs()

    # train hyponym projector
    hp = HyponymProjector()
    hp.train()