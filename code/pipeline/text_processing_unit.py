import os
from typing import *

class TextProcessingUnit:

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 max_files: Union[int, None] = None
                 ) -> None:
        """Initialize a text processing unit.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
        """
        self.path_in = path_in
        self.path_out = path_out
        self._max_files = max_files
        if os.path.isdir(self.path_in):
            self._fnames = [fname for fname in os.listdir(self.path_in)
                        if os.path.isfile(os.path.join(self.path_in, fname))]
        elif os.path.isfile(self.path_in):
            self._fnames = [self.path_in]
        self._fnames.sort()
        self._num_files = len(self._fnames)
        self._upper_bound = self._get_upper_bound()
        self._num_sents = 0
        self._files_processed = 0
        self._sents_processed = 0

        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)

    def _get_upper_bound(self) -> int:
        if self._max_files:
            return min(self._max_files, self._num_files)
        return self._num_files

    def _update_cmd(self) -> None:
        """Update the information on the command line."""
        final_msg = False
        if self._files_processed == self._upper_bound:
            if self._sents_processed == self._num_sents:
                msg = 'Processing: sentence {}, file {} of {}'
                print(msg.format(self._sents_processed, self._files_processed,
                                 self._num_files))
                final_msg = True
        if not final_msg:
            msg = 'Processing: sentence {}, file {} of {}\r'
            print(msg.format(self._sents_processed, self._files_processed,
                             self._num_files), end='\r')