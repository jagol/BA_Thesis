from typing import Generator, List, Set, TextIO


class Corpus:

    """Class to represent corpora and pseudo corpora as a collection of
    indices to the documents in the original corpus file.
    """

    def __init__(self, docs: Set[int], path: str) -> None:
        """Initialize the corpus.

        Args:
            docs: A list of indices to the docs belonging to the corpus.
            path: The path to the original corpus file
        """
        self.docs = docs
        self.num_docs = len(self.docs)
        self.docs_read = set()
        self.path = path

    def get_corpus_docs(self) -> Generator[List[List[str]], None, None]:
        """Get all the documents belonging to the corpus.

        Yield a generator of documents. Each document is a list of
        sentences and each sentence a list of words.
        """
        for i, doc in enumerate(self.get_docs()):
            if i in self.docs:
                doc = [line.strip('\n').split(' ') for line in doc]
                self.docs_read.add(i)
                yield doc
            # stop iterating if all docs were fetched
            if len(self.docs_read) == self.num_docs:
                break
        # check if all docs were fetched
        not_extracted = []
        for i in self.docs:
            if i not in self.docs_read:
                not_extraced.append(i)
        # throw exception if not all documents were fetched
        if not_extracted:
            doc_ids = ', '.join([str(i) for i in not_extraced])
            msg = 'Not all documents were extracted. DocIDs: {}'
            raise Exception(msg.format(doc_ids))

    def get_docs(self) -> Generator[List[str], None, None]:
        """Yield documents from given file.

        Each document is a list of lines
        """
        doc = []
        with open(self.path, 'r', encoding='utf8') as f:
            for line in f:
                if line == '\n':
                    yield doc
                    doc = []
                else:
                    doc.append(line)

    def get_orig_corpus_len(self) -> int:
        i = 0
        for _ in self.get_docs():
            i += 1
        return i