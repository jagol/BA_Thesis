import os
from utility_functions import *


def test_all_token_terms_have_embeddings(path_out: str) -> None:
    """Test if all token terms have embeddings."""
    path_token_terms = os.path.join(
        path_out, 'processed_corpus/token_terms.txt')
    path_token_embeddings = os.path.join(
        path_out, 'embeddings/token_embeddings_global_w2v.vec')
    test_all_terms_have_embeddings(path_token_terms, path_token_embeddings)


def test_all_lemma_terms_have_embeddings(path_out: str) -> None:
    """Test if all lemma terms have embeddings."""
    path_lemma_terms = os.path.join(
        path_out, 'processed_corpus/lemma_terms.txt')
    path_lemma_embeddings = os.path.join(
        path_out, 'embeddings/lemma_embeddings_global_w2v.vec')
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
    embeddings = load_embeddings(path_embeddings)
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
        raise Exception(msg1+msg2+msg3)
    else:
        print('All terms have embeddings.')


def load_terms(path_terms: str) -> Set[str]:
    terms = set()
    with open(path_terms, 'r', encoding='utf8') as f:
        for line in f:
            terms.add(line.strip('\n'))
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