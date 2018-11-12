import json
import argparse
from typing import Tuple, Dict, List, Generator, Union

# ----------------------------------------------------------------------
# type definitions
corpus_config_type = Dict[str, Dict[str, str]]
doc_type = Union[List[str], List[List[str]], str]

# ----------------------------------------------------------------------


def get_docs(fpath: str,
             word_tokenized: bool = True,
             sent_tokenized: bool = True
             ) -> Generator[doc_type, None, None]:
    """Yield each document of a corpus.

    Load the corpus from a file. There is one line per sentence. Between
    each document there is an additional newline. If tokenized is true,
    yield each document as a list of strings. If not, yield each
    document as a string.

    Args:
        fpath: path to corpus file
        word_tokenized: indicate, if output should be word_tokenized
        sent_tokenized: indicate, if output should be split into
            sentences
    Return:
        a generator object of documents
    """
    with open(fpath, 'r', encoding='utf8') as f:
        doc = []
        for line in f:
            if line == '\n':
                yield format_doc(doc, word_tokenized, sent_tokenized)
            else:
                doc.append(line.strip('\n'))


def format_doc(doc: List[str],
               word_tokenized: bool,
               sent_tokenized: bool
               ) -> doc_type:
    """Return a document in the required format.

     Returns document as a list of tokens, a list of sentences, a list
     of tokenized sentences or a string, depending on the input
     configuration.

     Args:
         doc: A List of sentences. Each token in the sentence is
            separated by a space.
        word_tokenized: indicate, if output should be word-tokenized
        sent_tokenized: indicate, if output should be split into
            sentences
     """
    if word_tokenized and sent_tokenized:
        return [sent.split(' ') for sent in doc]
    elif word_tokenized:
        doc_str = ' '.join(doc)
        return [w for w in doc_str.split(' ')]
    elif sent_tokenized:
        return doc
    else:
        return ' '.join(doc)


def get_corpus_config(unit: str) -> Tuple[str, corpus_config_type]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--server',
        help='indicate if local paths or server paths should be used',
        action='store_true')
    parser.add_argument(
        '-c',
        '--corpus',
        help='indicate name of corpus to be processed: europarl; dblp;'
    )
    args = parser.parse_args()

    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        if args.server:
            location = 'server'
        else:
            location = 'local'

        dblp = 'dblp'
        euprl = 'europarl'
        sp = 'sp'

        if args.corpus == dblp:
            return dblp, configs[location][dblp][unit]
        elif args.corpus == euprl:
            return euprl, configs[location][euprl][unit]
        elif args.corpus == sp:
            return sp, configs[location][sp][unit]
        else:
            raise Exception('Error! Corpus not known.')


def get_clus_config():
    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        return configs['clustering']