import sys
import os
import json
import argparse
import shutil
from typing import *
from scipy.spatial.distance import cosine


# ----------------------------------------------------------------------
# type definitions
corpus_config_type = Dict[str, Dict[str, str]]
doc_type = Union[List[str], List[List[str]], str]


# ------------------ functions to prepare environment ------------------

def prep_output_dir(path: str) -> None:
    """Prepare the output directory by creating subdirectories.

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
    parser.add_argument('-spd', '--skip_prep',
                        help='Skip preparation of output directories.',
                        action='store_true')
    parser.add_argument('-sl', '--skip_lingpp',
                        help='Skip ling-preprocessing.',
                        action='store_true')
    parser.add_argument('-spe', '--skip_pattern_extr',
                        help='Skip pattern-extraction.',
                        action='store_true')
    parser.add_argument('-sid', '--skip_idxer',
                        help='Skip idxer.',
                        action='store_true')
    parser.add_argument('-sfa', '--skip_freq_an',
                        help='Skip frequency analysis',
                        action='store_true')
    parser.add_argument('-se', '--skip_embeddings',
                        help='Skip embedding training',
                        action='store_true')
    parser.add_argument('-sde', '--skip_doc_embs',
                        help='Skip doc embeddings',
                        action='store_true')
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
        word_tokenized: Indicate, if output should be word_tokenized.
        sent_tokenized: Indicate, if output should be split into
            sentences.
    Return:
        A generator object of documents.
    """
    with open(fpath, 'r', encoding='utf8') as f:
        doc = []
        for line in f:
            if line == '\n':
                if doc:
                    yield format_doc(doc, word_tokenized, sent_tokenized)
                    doc = []
            else:
                doc.append(line.strip('\n'))


def get_docs_list(fpath: str,
                  word_tokenized: bool,
                  sent_tokenized: bool
                  ) -> List[doc_type]:
    """Same as get_docs, but returns a list instead of a generator."""
    with open(fpath, 'r', encoding='utf8') as f:
        docs = []
        doc = []
        for line in f:
            if line == '\n':
                if doc:
                    docs.append(
                        format_doc(doc, word_tokenized, sent_tokenized))
                    doc = []
            else:
                doc.append(line.strip('\n'))

        return docs


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


def get_num_docs(path: str) -> int:
    """Get the number of documents in a corpus."""
    with open(path, 'r', encoding='utf8') as f:
        counter = 0
        for line in f:
            if line == '\n':
                counter += 1
    return counter


def get_sim(v1: Iterator[float], v2: Iterator[float]) -> float:
    """Calcualte the consine similarity between vectors v1 and v2.

    Args:
        v1: vector 1
        v2: vector 2
    Return:
        The cosine similarity.
    """
    return 1-cosine(v1, v2)


def split_corpus(path_in: str,
                 path_out: str,
                 n: int
                 ) -> Tuple[List[str], List[int]]:
    """Split a corpus into n evenly sized chunks of documents.

    Args:
        path_in: File containing corpus to be split up.
        path_out: Path to directory of output file.
        n: The number of chunks.
    Return:
        A list of the filenames that were created.
        A list of starting numbers (doc id) for each filename.
    """
    num_docs = get_num_docs(path_in)
    split_points = [int(num_docs/n*i) for i in range(1, n+1)]
    split_points.append(num_docs)
    j = 0  # pointer to the current split-point
    num_docs = 0

    fnames = []
    start_nums = [0]
    with open(path_in, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(line)

            if line == '\n':
                num_docs += 1

            if num_docs == split_points[j]:
                fname = str(start_nums[j]) + '.txt.split'
                fnames.append(fname)
                start_nums.append(num_docs)
                fpath = os.path.join(path_out, fname)
                f = open(fpath, 'w', encoding='utf8')
                docs = ''.join(lines)
                f.write(docs)
                f.close()
                lines = []
                print(j, split_points[j])
                j += 1

    start_nums = start_nums[:-1]

    return fnames, start_nums


def concat_corpus(paths_in: List[str], path_out) -> None:
    """Concatenate the given files.

    Args:
        paths_in: The list of paths to the files.
        path_out: The path to the output file.
    """
    paths_in_str = ' '.join(paths_in)
    os.system('cat {0} > {1}'.format(paths_in_str, path_out))


def load_embeddings(emb_path: str) -> Dict[str, List[float]]:
    """Get the embeddings for the given terms.

    Args:
        term_ids: The ids of the input terms.
        emb_path: The path to the given embedding file.
    Return:
        A dictionary of the form: {term_id: embedding}
    """
    term_id_to_emb = {}
    with open(emb_path, 'r', encoding='utf8') as f:
        next(f)
        for line in f:
            vals = line.strip(' \n').split(' ')
            term_id = vals[0]
            emb = [float(f) for f in vals[1:]]
            term_id_to_emb[term_id] = emb

    return term_id_to_emb


def main():
    path = 'output/dblp/processed_corpus/pp_lemma_corpus.txt'
    split_corpus(path, 8)


if __name__ == '__main__':
    main()
