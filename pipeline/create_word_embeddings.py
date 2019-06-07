import json
import tensorflow as tf
import tensorflow_hub as hub


def get_text_input(path):
    """Load text input for word_embeddings from json.

    :param path: str, path to json-file
    :return: list[list[str]], tokenized sentences
    """
    with open(path, 'r', encoding='utf8') as f:
        sent_dict = json.load(f)
    sents = [sent_dict[i] for i in sent_dict]
    tokenized_sents = [[word[0] for word in sent] for sent in sents]
    return tokenized_sents


def get_sents_length(tokenized_sents):
    """Get the length for each sentence.

    :param tokenized_sents: list[list[str]], tokenized sentences
    :return: list[int], list with lenght for each sentence
    """
    return [len(sent) for sent in tokenized_sents]


def format_sents_tf(tokenized_sents, length):
    """Format senteces to same length for Tensorflow.

    Fill up each sentence with empty strings up to maxlen.

    :param tokenized_sents: list[list[str]], tokenized sentences
    :param length: int, number to which lists get extended with
        empty strings
    """
    formatted_sents = []
    for sent in tokenized_sents:
        formatted_sents.append(sent + [''] * (length - len(sent)))
    return formatted_sents


def get_embeddings(tokenized_sents, num_sents, save=True,):
    """Get word_embeddings for tokenized sentences using ELMO.

    :param tokenized_sents: list[list[str]], tokenized sentences
    :param save: bool, if true, save embeddings as pickle
    :param num_sents: int, number of sents
    :return: elmo object containing the embeddings
    """
    print('Downloading model...')
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    print('Calculating embeddings...')
    embeddings = elmo(
        inputs={
            "tokens": format_sents_tf(tokenized_sents, 150),
            "sequence_len": get_sents_length(tokenized_sents)
        },
        signature="tokens",
        as_dict=True)["elmo"]

    print('Save embeddings')
    if save:
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run(embeddings)
        saver = tf.train.Saver()
        saver.save(embeddings, 'embeddings_{}'.format(num_sents))

    return embeddings


def main():
    path = './preprocessed_corpus/500.json'
    tokenized_sents = get_text_input(path)
    print(len(tokenized_sents))
    get_embeddings(tokenized_sents, 500)
    print('Done')


if __name__ == '__main__':
    main()
