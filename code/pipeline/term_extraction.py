from typing import *

class TermExractor:
    """Class to extract relevant terms from a corpus.

    To determine relevancy, use a combination of TFIDF and
    C-Value-Method.

    A tokenized, tagged and lemmatized corpus as json-file is expected
    as input.

    The Extractor splits the corpus into parts (documents) to calculate
    a TFIDF-value. It operates under the assumption that important
    content words appear concentrated in one place of the corpus
    (because the word belongs to a topic talked about) whereas
    non-topic words appear everywhere in the corpus and are thus get a
    low TFIDF-value.

    Nested Terms or multiword terms are not captured by TFIDF but should
    get a high number in the c-value.

    For a term to be 'important' and thus get extracted it is enough to
    have a high TFIDF or a high c-value.
    """

    def __init__(self, path_corpus, path_out, threshhold_tfidf=0.8,
                 threshhold_cvalue=0.8) -> None:
        self.path_corpus = path_corpus
        self.path_out = path_out
        self.threshhold_tfidf = threshhold_tfidf
        self.threshhold_cvalue = threshhold_cvalue

    def split_to_documents(self) -> None:
        """Split corpus into documents.

        This method is necessary to be able to calculate the importance
        of a Term for a document."""
        pass

    def get_terms(self) -> List[str]:
        """Get a list of the most important terms in the corpus."""
