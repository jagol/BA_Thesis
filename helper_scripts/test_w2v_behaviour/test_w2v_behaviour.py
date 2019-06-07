import subprocess


def get_w2v_vocab(path_embs):
    vocab = set()
    with open(path_embs, 'r', encoding='utf8') as f:
        next(f)
        for line in f:
            word = line.split(' ')[0]
            vocab.add(word)
    return vocab - {'</s>'}


def train(path_corpus, path_embs):
    subprocess.call(["./word2vec", "-threads", "6", "-train", path_corpus,
                     "-output", path_embs, "-min-count", "1"])


def get_unique_words_in_corpus(path_corpus):
    vocab = []
    with open(path_corpus, 'r', encoding='utf8') as f:
        for line in f:
            vocab.extend(line.strip('\n').split(' '))
    return set(vocab)


def check_equality(expected, actual):
    if not expected == actual:
        diff = len(expected - actual)
        raise Exception('Not equal! Vocab expected: {}, Vocab actual: {}, Diff: {}'.format(len(expected), len(actual), diff))
    print('Expected vocab and actual vocab are equal.')



def main():
    path_corpus = 'test_corpus2.txt'
    path_embs = 'embeddings.vec'
    vocab_expected = get_unique_words_in_corpus(path_corpus)
    train(path_corpus, path_embs)
    vocab_actual = get_w2v_vocab(path_embs)
    check_equality(vocab_expected, vocab_actual)


if __name__ == '__main__':
    main()