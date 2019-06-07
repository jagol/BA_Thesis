import os
import json
from typing import *
from embeddings import Embeddings
from utility_functions import get_docs, get_num_docs


# ---------- Test consistency in corpus files ----------

def test_corpus_files(path_out: str) -> None:
    """Tests if corpus files are not corrupt."""
    path_pp_ling_corpus = os.path.join(path_out,
                                       'processed_corpus/ling_pp_corpus.txt')
    path_pp_token_corpus = os.path.join(path_out,
                                        'processed_corpus/pp_token_corpus.txt')
    path_pp_lemma_corpus = os.path.join(path_out,
                                        'processed_corpus/pp_lemma_corpus.txt')
    paths = [path_pp_ling_corpus, path_pp_token_corpus, path_pp_lemma_corpus]
    test_number_empty_lines(*paths)
    test_num_non_empty_lines(*paths)
    test_all_documents_contain_words(*paths)


def test_number_empty_lines(path_pp_ling_corpus: str,
                            path_pp_token_corpus: str,
                            path_pp_lemma_corpus: str
                            ) -> None:
    """Test if corpus files have the same number of empty lines."""
    pp_ling = open(path_pp_ling_corpus, 'r', encoding='utf8')
    pp_token = open(path_pp_token_corpus, 'r', encoding='utf8')
    pp_lemma = open(path_pp_lemma_corpus, 'r', encoding='utf8')
    num_empty_lines_pp_ling = count_num_empty_lines(pp_ling)
    num_empty_lines_pp_token = count_num_empty_lines(pp_token)
    num_empty_lines_pp_lemma = count_num_empty_lines(pp_lemma)
    if num_empty_lines_pp_ling != num_empty_lines_pp_token:
        raise Exception('num_empty_lines_pp_ling != num_empty_lines_pp_token')
    if num_empty_lines_pp_ling != num_empty_lines_pp_lemma:
        raise Exception('num_empty_lines_pp_ling != num_empty_lines_pp_lemma')


def test_num_non_empty_lines(path_pp_ling_corpus: str,
                             path_pp_token_corpus: str,
                             path_pp_lemma_corpus: str
                             ) -> None:
    """"Test if corpus files have the same number of non-empty lines."""
    pp_ling = open(path_pp_ling_corpus, 'r', encoding='utf8')
    pp_token = open(path_pp_token_corpus, 'r', encoding='utf8')
    pp_lemma = open(path_pp_lemma_corpus, 'r', encoding='utf8')
    num_non_empty_lines_pp_ling = count_num_non_empty_lines(pp_ling)
    num_non_empty_lines_pp_token = count_num_non_empty_lines(pp_token)
    num_non_empty_lines_pp_lemma = count_num_non_empty_lines(pp_lemma)
    if num_non_empty_lines_pp_ling != num_non_empty_lines_pp_token:
        raise Exception(
            'num_non_empty_lines_pp_ling != num_non_empty_lines_pp_token')
    if num_non_empty_lines_pp_ling != num_non_empty_lines_pp_lemma:
        raise Exception(
            'num_non_empty_lines_pp_ling != num_non_empty_lines_pp_lemma')


def test_all_documents_contain_words(path_pp_ling_corpus: str,
                                     path_pp_token_corpus: str,
                                     path_pp_lemma_corpus: str
                                     ) -> None:
    """Test if all documents in the pp-files contain words."""
    for i, doc in enumerate(get_docs(path_pp_ling_corpus,
                                     sent_tokenized=False,
                                     word_tokenized=False)):
        if len(doc) == 0:
            raise Exception('Document {} is empty.'.format(i))
    for i, doc in enumerate(get_docs(path_pp_token_corpus,
                                     sent_tokenized=False,
                                     word_tokenized=False)):
        if len(doc) == 0:
            raise Exception('Document {} is empty.'.format(i))
    for i, doc in enumerate(get_docs(path_pp_lemma_corpus,
                                     sent_tokenized=False,
                                     word_tokenized=False)):
        if len(doc) == 0:
            raise Exception('Document {} is empty.'.format(i))


def count_num_empty_lines(f):
    i = 0
    for line in f:
        if line == '\n':
            i += 1
    return i


def count_num_non_empty_lines(f):
    i = 0
    for line in f:
        if line != '\n':
            i += 1
    return i


# ---------- Test consistency between corpus and pattern files ---------

def test_term_pattern_files(path_out):
    """Test if extracted terms and patterns are consistent."""
    path_token_terms = os.path.join(path_out,
                                    'processed_corpus/token_terms.txt')
    path_lemma_terms = os.path.join(path_out,
                                    'processed_corpus/lemma_terms.txt')
    path_token_hrels = os.path.join(
        path_out, 'hierarchy/hierarchical_relations_tokens.json')
    path_lemma_hrels = os.path.join(
        path_out, 'hierarchy/hierarchical_relations_lemmas.json')
    paths_terms = [path_token_terms, path_lemma_terms]
    paths_hrels = [path_token_hrels, path_lemma_hrels]
    paths = paths_terms + paths_hrels
    test_all_words_in_hierarch_rels_in_terms(*paths)


def test_all_words_in_hierarch_rels_in_terms(path_token_terms: str,
                                             path_lemma_terms: str,
                                             path_token_hrels: str,
                                             path_lemma_hrels: str
                                             ) -> None:
    """Test if words appearing in relations are in the terms."""
    _test_all_words_in_hierarch_rels_in_terms(path_token_terms,
                                              path_token_hrels)
    _test_all_words_in_hierarch_rels_in_terms(path_lemma_terms,
                                              path_lemma_hrels)


def _test_all_words_in_hierarch_rels_in_terms(path_terms: str,
                                              path_hrels: str
                                              ) -> None:
    """Test if words appearing in relations are in the terms."""
    terms = set()
    with open(path_terms, 'r', encoding='utf8') as f:
        for line in f:
            terms.add(line.strip('\n'))
    with open(path_hrels, 'r', encoding='utf8') as f:
        hrels = json.load(f)
    for hyper in hrels:
        if hyper not in terms:
            raise Exception(
                'The hypernym {} is not in the terms-file.'.format(hyper))
        for hypo in hrels[hyper]:
            if hypo not in terms:
                raise Exception(
                    'The hyponym {} is not in the terms-file.'.format(hypo))


# ---------- Test consistency between corpus and indexing files --------

def test_indexing_files(path_out: str) -> None:
    """Test consistency of indexing files."""
    path_idx_to_token = os.path.join(path_out, 'indexing/idx_to_token.json')
    path_idx_to_lemma = os.path.join(path_out, 'indexing/idx_to_lemma.json')
    path_token_to_idx = os.path.join(path_out, 'indexing/token_to_idx.json')
    path_lemma_to_idx = os.path.join(path_out, 'indexing/lemma_to_idx.json')
    # path_token_idx_corpus = os.path.join(
    #     path_out, 'processed_corpus/token_idx_corpus.txt')
    # path_lemma_idx_corpus = os.path.join(
    #     path_out, 'processed_corpus/lemma_idx_corpus.txt')
    path_token_idxs = os.path.join(
        path_out, 'processed_corpus/token_terms_idxs.txt')
    path_lemma_idxs = os.path.join(
        path_out, 'processed_corpus/lemma_terms_idxs.txt')
    path_hrels_token_idxs = os.path.join(
        path_out, 'hierarchy/hierarch_rels_tokens_idx.json')
    path_hrels_lemma_idxs = os.path.join(
        path_out, 'hierarchy/hierarch_rels_lemmas_idx.json')

    test_indexing_continously_from_zero(path_idx_to_token, path_idx_to_lemma)
    test_idx_dicts_have_same_length(path_idx_to_token, path_token_to_idx)
    test_idx_dicts_have_same_length(path_idx_to_lemma, path_lemma_to_idx)
    test_all_term_idxs_in_idx_mappings(path_idx_to_token, path_token_idxs)
    test_all_term_idxs_in_idx_mappings(path_idx_to_lemma, path_lemma_idxs)
    test_all_rel_idxs_in_term_idxs(path_token_idxs, path_hrels_token_idxs)
    test_all_rel_idxs_in_term_idxs(path_lemma_idxs, path_hrels_lemma_idxs)


def test_indexing_continously_from_zero(path_idx_to_token: str,
                                        path_idx_to_lemma: str
                                        ) -> None:
    """Test if indexing has no gaps."""
    with open(path_idx_to_token, 'r', encoding='utf8') as f:
        idx_to_token = json.load(f)
        _test_indexing_continously_from_zero(idx_to_token)
    with open(path_idx_to_lemma, 'r', encoding='utf8') as f:
        idx_to_lemma = json.load(f)
        _test_indexing_continously_from_zero(idx_to_lemma)


def _test_indexing_continously_from_zero(idx_to_token: Dict[str, str]
                                         ) -> None:
    """Test if indexing has no gaps."""
    num_keys = len(idx_to_token)
    cont_idxs = [str(i) for i in range(num_keys)]
    for idx in cont_idxs:
        if idx not in idx_to_token:
            raise Exception('Index {} not in indexing file.'.format(idx))


def test_idx_dicts_have_same_length(path_idx_to_word: str,
                                    path_word_to_idx: str
                                    ) -> None:
    """Check mappings from wordtoidx and idxtoword have same length."""
    with open(path_idx_to_word, 'r', encoding='utf8') as f:
        idx_to_word = json.load(f)
    with open(path_word_to_idx, 'r', encoding='utf8') as f:
        word_to_idx = json.load(f)
    if not len(idx_to_word) == len(word_to_idx):
        raise Exception('Idx to word mappings do not have the same length')


def test_idx_dicts_have_same_content(path_idx_to_word: str,
                                     path_word_to_idx: str
                                     ) -> None:
    """Check that the idx dicts have the same keys and values in rev."""
    with open(path_idx_to_word, 'r', encoding='utf8') as f:
        idx_to_word = json.load(f)
    with open(path_word_to_idx, 'r', encoding='utf8') as f:
        word_to_idx = json.load(f)
    idx_to_word_keys = set(idx_to_word.keys())
    word_to_idx_keys = set(word_to_idx.keys())
    idx_to_word_values = set(idx_to_word.values())
    word_to_idx_values = set(word_to_idx.values())
    if not idx_to_word_keys == idx_to_word_values:
        raise Exception("Ids of idx_to_word don't match word_to_idx_values.")
    if not word_to_idx_keys == word_to_idx_values:
        raise Exception("word_to_idx_keys don't match word_to_idx_values.")


def test_all_term_idxs_in_idx_mappings(path_idx_to_word: str,
                                       path_term_idxs: str
                                       ) -> None:
    """Test if all term-idxs appear in the idx-to-term-mappings."""
    with open(path_idx_to_word, 'r', encoding='utf8') as f:
        idx_to_word = json.load(f)
    term_idxs = list()
    with open(path_term_idxs, 'r', encoding='utf8') as f:
        for line in f:
            term_idxs.append(line.strip('\n'))
    for term_idx in term_idxs:
        if term_idx not in idx_to_word:
            raise Exception('{} is in term-idxs but in the mappings.')


def test_all_rel_idxs_in_term_idxs(path_term_idxs: str,
                                   path_hrels_term_idxs: str
                                   ) -> None:
    """Test if all terms in relations are in the term-idx files."""
    msg_hyper = '{} hyper in relation but not in term set.'
    msg_hypo = '{} hypo in relation but not in term set.'
    with open(path_hrels_term_idxs, 'r', encoding='utf8') as f:
        hrels_term_idxs = json.load(f)
        term_idxs = set()
    with open(path_term_idxs, 'r', encoding='utf8') as f:
        for line in f:
            term_idxs.add(line.strip('\n'))

    for hyper_idx in hrels_term_idxs:
        if hyper_idx not in term_idxs:
            raise Exception(msg_hyper.format(hyper_idx))
        hypos = hrels_term_idxs[hyper_idx]
        for hypo_idx in hypos:
            if str(hypo_idx) not in term_idxs:
                raise Exception(msg_hypo.format(hypo_idx))


# Test consistency between corpus and frequency files.

def test_frequency_files(path_out: str) -> None:
    """Test consistency of the frequency files."""
    # get_num_docs
    path_tf_tokens = os.path.join(path_out, 'frequencies/tf_tokens.json')
    path_tf_lemmas = os.path.join(path_out, 'frequencies/tf_lemmas.json')
    path_tfidf_tokens = os.path.join(path_out, 'frequencies/tfidf_tokens.json')
    path_tfidf_lemmas = os.path.join(path_out, 'frequencies/tfidf_lemmas.json')
    path_dl = os.path.join(path_out, 'frequencies/dl.json')
    # path_df_tokens = os.path.join(path_out, 'frequencies/df_tokens.json')
    # path_df_lemmas = os.path.join(path_out, 'frequencies/df_lemmas.json')
    path_ling_pp_corpus = os.path.join(path_out,
                                       'processed_corpus/ling_pp_corpus.txt')
    path_token_idxs = os.path.join(path_out, 'token_terms_idxs.txt')
    path_lemma_idxs = os.path.join(path_out, 'lemma_terms_idxs.txt')
    num_docs = get_num_docs(path_ling_pp_corpus)
    test_tf_has_all_doc_ids(path_tf_tokens, num_docs)
    test_tf_has_all_doc_ids(path_tf_lemmas, num_docs)
    test_tf_has_all_word_ids(path_tf_tokens, path_token_idxs)
    test_tf_has_all_word_ids(path_tf_lemmas, path_lemma_idxs)
    test_tf_tfidf_dl_have_same_length(path_tf_tokens, path_tfidf_tokens,
                                      path_dl)
    test_tf_tfidf_dl_have_same_length(path_tf_lemmas, path_tfidf_lemmas,
                                      path_dl)
    test_tf_values_tfidf_values(path_tf_tokens, path_tfidf_tokens)
    test_tf_values_tfidf_values(path_tf_lemmas, path_tfidf_lemmas)


def test_tf_has_all_doc_ids(path_tf: str, num_docs: int) -> None:
    """Test if the tf-file contains all doc-ids."""
    with open(path_tf, 'r', encoding='utf8') as f:
        tf = json.load(f)
    doc_ids = [str(i) for i in range(num_docs)]
    for doc_id in doc_ids:
        if doc_id not in tf:
            raise Exception('Doc_id {} not in tf.'.format(doc_id))


def test_tf_has_all_word_ids(path_tf: str,
                             path_term_idxs: str
                             ) -> None:
    """Check if all term-idxs appear in the term frequencies."""
    with open(path_tf, 'r', encoding='utf8') as f:
        tf = json.load(f)
    term_idxs = set()
    with open(path_term_idxs, 'r', encoding='utf8') as f:
        for line in f:
            term_idx = line.strip('\n')
            term_idxs.add(term_idx)
    for doc_id in tf:
        doc_freqs = tf[doc_id]
        for tid in doc_freqs:
            term_idxs.remove(tid)
    if len(term_idxs) != 0:
        msg = '{} term-idxs do not appear in the tf-file.'
        raise Exception(msg.format(len(term_idxs)))


def test_tf_tfidf_dl_have_same_length(path_tf: str,
                                      path_tfidf: str,
                                      path_dl: str
                                      ) -> None:
    """Test if tf, tfidf and dl have the same keys."""
    with open(path_tf, 'r', encoding='utf8') as f:
        tf = json.load(f)
    with open(path_tfidf, 'r', encoding='utf8') as f:
        tfidf = json.load(f)
    with open(path_dl, 'r', encoding='utf8') as f:
        dl = json.load(f)
    keys_tf = set(tf.keys())
    keys_tfidf = set(tfidf.keys())
    keys_dl = set(dl.keys())
    if keys_tf != keys_tfidf:
        raise Exception('tf and tfidf do not have the same keys.')
    if keys_tf != keys_dl:
        raise Exception('tf and dl do not have the same keys.')


def test_tf_values_tfidf_values(path_tf: str, path_tfidf: str) -> None:
    """Test if tf and tfidf have the same word-ids in the same docs."""
    with open(path_tf, 'r', encoding='utf8') as f:
        tf = json.load(f)
    with open(path_tfidf, 'r', encoding='utf8') as f:
        tfidf = json.load(f)
    for doc_id in tf:
        doc_tf = tf[doc_id]
        doc_tfidf = tfidf[doc_id]
        if doc_tf.keys() != doc_tfidf.keys():
            msg = 'Document id {} has different term ids in tf and tfidf.'
            raise Exception(msg.format(doc_id))


# ---------- Test consistency between terms and embeddings. ----------

def test_embedding_files(path_out):
    test_all_token_terms_have_embeddings(path_out)
    test_all_lemma_terms_have_embeddings(path_out)


def test_all_token_terms_have_embeddings(path_out: str) -> None:
    """Test if all token terms have embeddings."""
    path_token_terms = os.path.join(
        path_out, 'processed_corpus/token_terms_idxs.txt')
    path_token_embeddings = os.path.join(
        path_out, 'embeddings/embs_token_global_Word2Vec.vec')
    test_all_terms_have_embeddings(path_token_terms, path_token_embeddings)
    path_token_embeddings = os.path.join(
        path_out, 'embeddings/embs_token_global_GloVe.vec')
    test_all_terms_have_embeddings(path_token_terms, path_token_embeddings)


def test_all_lemma_terms_have_embeddings(path_out: str) -> None:
    """Test if all lemma terms have embeddings."""
    path_lemma_terms = os.path.join(
        path_out, 'processed_corpus/lemma_terms_idxs.txt')
    path_lemma_embeddings = os.path.join(
        path_out, 'embeddings/embs_lemma_global_Word2Vec.vec')
    test_all_terms_have_embeddings(path_lemma_terms, path_lemma_embeddings)
    path_lemma_embeddings = os.path.join(
        path_out, 'embeddings/embs_lemma_global_GloVe.vec')
    test_all_terms_have_embeddings(path_lemma_terms, path_lemma_embeddings)


def test_all_terms_have_embeddings(path_terms: str,
                                   path_embeddings: str
                                   ) -> None:
    """Test if all terms have embeddings for the given two files.

    Args:
        path_terms: Text-file with 1 term per line.
        path_embeddings: vec-file with term and dimension values
            separated by space.
    """
    terms = load_terms(path_terms)
    idx_to_term = json.load('<intput here correct path>')
    embeddings = Embeddings.load_term_embeddings(terms, path_embeddings,
                                                 idx_to_term)
    embedded_terms = set(embeddings)
    not_in_et = []
    for t in terms:
        if t not in embedded_terms:
            not_in_et.append(t)

    if len(not_in_et) != 0:
        msg1 = 'Error! Not all terms have embeddings. '
        msg2 = 'Num terms without embeddings: {}. '.format(len(not_in_et))
        if len(not_in_et) < 20:
            msg3 = 'Terms without embeddings: {}'.format(not_in_et)
        else:
            msg3 = ''
        raise Exception(msg1 + msg2 + msg3)


def load_terms(path_terms: str) -> Set[int]:
    terms = set()
    with open(path_terms, 'r', encoding='utf8') as f:
        for line in f:
            terms.add(int(line.strip('\n')))
    return terms


if __name__ == '__main__':
    from utility_functions import get_config, get_cmd_args
    config = get_config()
    args = get_cmd_args()
    path = config['paths'][args.location][args.corpus]['path_out']
    print('Test if all token terms have embeddings...')
    test_all_token_terms_have_embeddings(path)
    print('Test if all lemma terms have embeddings...')
    test_all_lemma_terms_have_embeddings(path)
