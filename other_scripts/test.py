from typing import *

def _concat_indices(idx_sent: List[int],
                    term_indices: List[List[int]]
                    ) -> List[Union[int, str]]:
    """Concatenate the indices in idx_sent given by term_indices.

    Use an underscore as concatenator. For example: If idx_sent is
    [7, 34, 5] and term_indices is [[1, 2]] the output should be
    [7, 34_5]. Note: While idx_sent contains an actual index
    represenation term_indices only contains list indices that
    indicate which index representations in idx_sent belong together
    as one term.

    Args:
        idx_sent: a sentence representation using indices
        term_indices: a list of indices in the sentence (starting by
            0 to lengh of the sentence-1) indicating which
    """
    for ti in term_indices[::-1]:
        str_idxs = [str(idx) for idx in idx_sent[ti[0]:ti[-1]+1]]
        joined = '_'.join(str_idxs)
        idx_sent[ti[0]:ti[-1]+1] = [joined]

    return idx_sent

idx_sent = [3, 19, 6]
term_indices = [[1, 2]]
print(_concat_indices(idx_sent, term_indices))
idx_sent = [3, 19, 6, 4, 430, 30]
term_indices = [[0, 2]]
print(_concat_indices(idx_sent, term_indices))
idx_sent = [3, 19, 6, 4, 430, 30]
term_indices = [[0, 1], [3, 5]]
print(_concat_indices(idx_sent, term_indices))