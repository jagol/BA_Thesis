import sys
import os
import json
import argparse
import shutil
from typing import Dict, List, Generator, Union, Any

# ----------------------------------------------------------------------
# type definitions
corpus_config_type = Dict[str, Dict[str, str]]
doc_type = Union[List[str], List[List[str]], str]


# ------------------ functions to prepare environment ------------------

def prep_output_dir(path: str) -> None:
    """Prepare the output directory by creating needed subdirectories.

    The following subdirectories are created:
    - embeddings/
    - hierarchy/
    - processed_corpus/
    - indexing/
    - concept_terms/
    - frequencies/
    """
    if not os.path.isdir(path):
        raise Exception('ERROR! Path does not lead to a directory.')
    dir_names = ['embeddings', 'hierarchy', 'processed_corpus',
                 'indexing', 'concept_terms', 'frequencies']
    for dir_name in dir_names:
        path_dir = os.path.join(path, dir_name)
        if os.path.exists(path_dir):
            msg = ('One or more of the directories exist already. Should the '
                   'directories be overwritten? y or n: ')
            y_or_n = input(msg)

            for dir_name in dir_names:
                if y_or_n == 'y':
                    path_dir = os.path.join(path, dir_name)
                    shutil.rmtree(path_dir)
                else:
                    sys.exit()

            break

    for dir_name in dir_names:
        dir_path = os.path.join(path, dir_name)
        os.mkdir(dir_path)


# ------------------ configuration and cmd args functions --------------

def get_cmd_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--location',
        help='indicate if local paths or server paths should be used',
    )
    parser.add_argument(
        '-c',
        '--corpus',
        help='name of corpus to be processed: europarl; dblp; sp;'
    )
    args = parser.parse_args()
    return args


def get_clus_config():
    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        return configs['clustering']


def get_config():
    with open('configs.json', 'r', encoding='utf8') as f:
        configs = json.load(f)
        return configs


def get_path_out(args: Any, config: Dict[str, Any]) -> str:
    path_out = config['paths'][args.location][args.corpus]['path_out']
    return path_out


def get_path_in(args: Any, config: Dict[str, Any]) -> str:
    path_in = config['paths'][args.location][args.corpus]['path_in']
    return path_in


# ------------------ corpus processing functions -----------------------

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
