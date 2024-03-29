import os
from typing import *

class TextProcessingUnit:

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 max_docs: Union[int, None] = None
                 ) -> None:
        """Initialize a text processing unit.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
        """
        self.path_in = path_in
        self.path_out = path_out
        self._max_docs = max_docs
        if os.path.isdir(self.path_in):
            self._fnames = [fname for fname in os.listdir(self.path_in)
                        if os.path.isfile(os.path.join(self.path_in, fname))]
        elif os.path.isfile(self.path_in):
            self._fnames = [self.path_in]
        self._fnames.sort()
        self._num_files = len(self._fnames)
        self._upper_bound = self._get_upper_bound()
        self._num_sents = 0
        self._docs_processed = 0
        self._sents_processed = 0
        self._num_docs = 0

        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)

    def _get_upper_bound(self) -> int:
        if self._max_docs:
            return min(self._max_docs, self._num_docs)
        return self._num_docs