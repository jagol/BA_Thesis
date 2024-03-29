import os
import re
import json
import time
from typing import *


# Type of hypernym relations with hypernym as key and a list of hyponyms
# as values.
rels_type = Dict[str, List[str]]


class PatternExtractor:
    """Class to extract terms and hearst patterns.

    Terms are just NPs matching a pos-tag pattern defined in the
    TermExtractor.

    The extract method outputs the following files:
    - 'pp_token_corpus.txt': Tokens are space separated. Multiword-
        terms are concatenated by '_'. One sent per line. Two newlines
        between document borders.
    - 'pp_lemma_corpus.txt': Like 'pp_token_corpus', but with lemmas
        instead.
    - 'token_terms.txt': One extracted term per line. Multiword-terms are
        concatenated by '_'.
    - 'lemma_terms.txt': Like 'token_terms.txt' but with lemmas instead.
    - 'hierarchical_relations.json': Dict[str, List[str]]
        Each hypernym (key) is mapped to a list of hyponyms.
    """

    def __init__(self, path: str, max_docs: Union[int, None] = None) -> None:
        self.path = os.path.join(path, 'processed_corpus')
        self.path_in_file = os.path.join(self.path, 'ling_pp_corpus.txt')
        self.path_out_corpus = os.path.join(path, 'processed_corpus')
        self.path_out_hierarchy = os.path.join(path, 'hierarchy')
        self.path_out_pp_tokens = os.path.join(
            self.path_out_corpus, 'pp_token_corpus_non_tg.txt')
        self.path_out_pp_lemmas = os.path.join(
            self.path_out_corpus, 'pp_lemma_corpus_non_tg.txt')
        self.path_hierarchy_rels_tokens = os.path.join(
            self.path_out_hierarchy, 'hierarchical_relations_tokens_non_tg.json')
        self.path_hierarchy_rels_lemmas = os.path.join(
            self.path_out_hierarchy, 'hierarchical_relations_lemmas_non_tg.json')
        self.path_token_terms = os.path.join(
            self.path_out_corpus, 'token_terms_non_tg.txt')
        self.path_lemma_terms = os.path.join(
            self.path_out_corpus, 'lemma_terms_non_tg.txt')

        self.hierarch_rels_tokens = {}    # {hypernym: list of hyponyms}
        self.hierarch_rels_lemmas = {}

        self._file_write_threshhold = 10000
        self._docs_processed = 0
        self._token_corpus = []
        self._lemma_corpus = []
        self._token_terms = set()
        self._lemma_terms = set()

        self._start_time = 0
        self._start_doc = 0
        self._max_docs = max_docs
        self._upper_bound = self._max_docs

    def extract(self) -> None:
        """Extract terms and hierarchical relations."""
        self._start_time = time.time()
        for i, doc in enumerate(get_pp_corpus(self.path_in_file)):

            # Option to skip documents.
            if i < self._start_doc:
                i += 1
                continue

            doc_concat_tokens = []
            doc_concat_lemmas = []
            for sent in doc:
                # Extract hierarchical relations on token level.
                hierarch_rels = HearstExtractor.get_hierarch_rels(sent, 't')
                self.add_hierarch_rels(hierarch_rels, 't')

                # Extract hierarchical relations on lemma level.
                hierarch_rels = HearstExtractor.get_hierarch_rels(sent, 'l')
                self.add_hierarch_rels(hierarch_rels, 'l')

                # Find terms and get their indices in the sentence.
                term_indices, term_heads = TermExtractor.get_term_indices(sent)

                # Lowercase all tokens and lemmas.
                tokens = [w[0].lower() for w in sent]
                lemmas = [w[2].lower() for w in sent]

                # Get terms.
                token_terms, tokens = self._get_token_terms(
                    tokens, term_indices)
                lemma_terms, lemmas = self._get_lemma_terms(
                    tokens, lemmas, term_indices, term_heads)

                # Add terms to set of all terms.
                for tt in token_terms:
                    self._token_terms.add(tt)

                for lt in lemma_terms:
                    self._lemma_terms.add(lt)

                # Add sentence with concatentated terms to document.
                doc_concat_tokens.append(tokens)
                doc_concat_lemmas.append(lemmas)

            # Add document to corpus.
            self._token_corpus.append(doc_concat_tokens)
            self._lemma_corpus.append(doc_concat_lemmas)

            self._docs_processed += 1
            self.update_cmd_counter()

            if self._upper_bound:
                if self._docs_processed >= self._upper_bound:
                    break

            if self._docs_processed % self._file_write_threshhold == 0:
                self.update_cmd_time_info()
                self.write_corpus(self._token_corpus, self.path_out_pp_tokens)
                self.write_corpus(self._lemma_corpus, self.path_out_pp_lemmas)

                self._token_corpus = []
                self._lemma_corpus = []

        self.update_cmd_time_info(end=True)
        if self._token_corpus:
            self.write_corpus(self._token_corpus, self.path_out_pp_tokens)
            self.write_corpus(self._lemma_corpus, self.path_out_pp_lemmas)

        print('Writing token terms to file...')
        self.write_terms(self._token_terms, self.path_token_terms)
        print('Writing lemma terms to file...')
        self.write_terms(self._lemma_terms, self.path_lemma_terms)
        print('Writing hierarchical relations to file...')
        self.write_hierarch_rels()

    @staticmethod
    def write_terms(terms: Set[str], path: str) -> None:
        with open(path, 'w', encoding='utf8') as f:
            for t in terms:
                f.write(t + '\n')

    @staticmethod
    def _get_token_terms(tokens: List[str],
                         term_indices: List[List[int]]
                         ) -> List[List[str]]:
        """Get the sentences terms (token form) and tokens.

        Args:
            tokens: A list of tokens.
            term_indices: A list of lists. Each list is a term and
                contains the indices belonging to that term.
        """
        token_terms = []
        for ti in term_indices:
            token_terms.append(tokens[ti[0]])

        return [token_terms, tokens]

    @classmethod
    def _get_lemma_terms(cls,
                         tokens: List[str],
                         lemmas: List[str],
                         term_indices: List[List[int]],
                         term_heads: List[int]
                         ) -> List[List[str]]:
        """Get the lemmatized terms and sentences with lemmatized terms.

        Args:
            tokens: A list of tokens.
            lemmas: A list of lemmas.
            term_indices: A list of lists. Each list is a term and
                contains the indices belonging to that term.
            term_heads: A list relative indices that point to the head
                of the term/np.
        """
        lemma_terms = []

        # Loop backwards through term-indices.
        for i in range(len(term_indices)-1, -1, -1):
            term_idx = term_indices[i][0]
            term_head_idx = term_heads[i]

            token_term = tokens[term_idx]
            lemma_term = lemmas[term_idx]

            lemma_term = cls._exchange_head(token_term, lemma_term,
                                            term_head_idx)

            lemma_terms.append(lemma_term)
            lemmas[term_idx] = lemma_term

        return [lemma_terms, lemmas]

    @staticmethod
    def _exchange_head(token_term: str,
                       lemma_term: str,
                       term_head_idx: int
                       ) -> str:
        """Exchange the token head with lemma head.

        Args:
            token_term: The tokenized term.
            lemma_term: The lemmatized term.
            term_head_idx: The index of the head word of the np/term.

        """
        # Avoid errors by skipping missparsed cases.
        token_term_words = token_term.split('_')
        lemma_term_words = lemma_term.split('_')

        error = False
        if len(lemma_term_words) != len(token_term_words):
            error = True
        if term_head_idx >= len(token_term_words):
            error = True
        if not token_term_words:
            error = True

        # As head, use lemma, tokens for everything else.
        if not error:
            token_term_words[term_head_idx] = lemma_term_words[term_head_idx]
        lemma_term = '_'.join(token_term_words)
        return lemma_term

    def add_hierarch_rels(self, rels: rels_type, level: str) -> None:
        """Add given hierarchical relations to relation dictionary.

        Args:
            rels: a dict containing hypernym-hyponym relations
            level: 't' if token level, 'l' if lemma level.
        """
        if level == 't':
            hierarch_rels = self.hierarch_rels_tokens
        elif level == 'l':
            hierarch_rels = self.hierarch_rels_lemmas

        for hypernym in rels:
            hyponyms = rels[hypernym]
            if hypernym in hierarch_rels:
                hierarch_rels[hypernym].extend(hyponyms)
            else:
                hierarch_rels[hypernym] = hyponyms

    def write_corpus(self, docs, f_out):
        mode = self._get_write_mode()
        with open(f_out, mode, encoding='utf8') as f:
            f.write(self.docs_to_string(docs))

    @staticmethod
    def docs_to_string(docs: List[List[List[str]]]) -> str:
        docs_str = ''
        for doc in docs:
            for sent in doc:
                sent_str = ' '.join(sent)
                line = sent_str + '\n'
                docs_str += line
            docs_str += '\n'
        return docs_str

    def _get_write_mode(self) -> str:
        """Return the mode in which the file should be written to."""
        if self._docs_processed <= self._file_write_threshhold:
            return 'w'
        return 'a'

    def write_hierarch_rels(self):
        with open(self.path_hierarchy_rels_tokens, 'w', encoding='utf8') as f:
            json.dump(self.hierarch_rels_tokens, f)
        with open(self.path_hierarchy_rels_lemmas, 'w', encoding='utf8') as f:
            json.dump(self.hierarch_rels_lemmas, f)

    def update_cmd_counter(self) -> None:
        """Update the information on the command line."""
        if self._docs_processed == self._upper_bound:
            msg = 'Processing: document {} of {}'
            print(msg.format(self._docs_processed, self._upper_bound))
        else:
            msg = 'Processing: document {} of {}\r'
            print(msg.format(self._docs_processed, self._upper_bound),
                  end='\r')

    def update_cmd_time_info(self, end=False):
        """Update time information on the command line.

        Args:
            end: set to True if last writing procedure
        """
        time_stamp = time.time()
        time_passed = time_stamp - self._start_time
        if end:
            docs_proc_now = self._docs_processed % self._file_write_threshhold
            if docs_proc_now == 0:
                msg = ('Written {} documents to file in total. '
                       'Time passed: {:2f}')
                print(msg.format(self._docs_processed, time_passed))
            else:
                msg = ('Writing {} documents to file. '
                       'Written {} documents to file in total. '
                       'Time passed: {:2f}')
                print(msg.format(
                    docs_proc_now, self._docs_processed, time_passed))
        else:
            msg = ('Writing {} documents to file. '
                   'Written {} documents to file in total. '
                   'Time passed: {:2f}')
            print(msg.format(self._file_write_threshhold,
                             self._docs_processed, time_passed))


class TermExtractor:
    """Class to handle functions to extract terms from sentences."""

    # term_pattern = re.compile((r'(JJ[RS]{0,2}\d+ |NN[PS]{0,2}\d+ '
    #                            r'|IN\d+ |VB[NG]\d+ )*NN[PS]{0,2}\d+'))
    # term_pattern = re.compile(r'(JJ[RS]{0,2}\d+ |NN[PS]{0,2}\d+ |'
    #                           r'VB[NG]\d+ )*NN[PS]{0,2}\d+')
    term_pattern = r'\d+np\d+'
    index_pattern = re.compile(r'^(\d+)([^0-9]+)(\d+)$')

    @classmethod
    def get_term_indices(cls,
                         sent: List[List[str]]
                         ) -> Tuple[List[List[int]], List[int]]:
        """Extract terms of a sentence.

        Args:
            sent: see in type definitions
        Return:
            List of terms. Each term is a list of sentence indices,
            which make up the term.
        """
        lex_words = ['such', 'as', 'and', 'or',
                     'other', 'including', 'especially']
        poses = []
        for i, word in enumerate(sent):
            # Check if word is a special hearst pattern word.
            # if so, exclude it from any potential terms by
            # not using it's pos but the token/lemma.
            if word[2] in lex_words:
                poses.append(word[2])
            else:
                pos = word[1] + str(i)
                poses.append(pos)

        pos_sent = ' '.join(poses)
        matches = list(re.finditer(cls.term_pattern, pos_sent))
        term_indices = [cls.get_indices(match.group()) for match in matches]
        term_heads = [cls.get_head(match.group()) for match in matches]
        return term_indices, term_heads

    @classmethod
    def get_indices(cls, match: str) -> List[int]:
        """Get the indices in match.

        The indices are concatenated with their corresponding words.
        For example the input 'the0 pig1 is2 is3 pink4' leads to the
        output [0, 1, 2, 3, 4].

        Args:
            match: The input string with words and indices.
        """
        indices = []
        for pos in match.split(' '):
            if pos and pos[-1].isdigit():
                index = int(re.search(cls.index_pattern, pos).group(3))
                indices.append(index)
        return indices

    @classmethod
    def get_head(cls, pos: str) -> int:
        """Get the relative index of the head word of the term/np.

        Args:
            pos: The pos-tag with an prefixed index.
        Return:
            The relative index of the term head in the np.
        """
        pattern = re.compile(r'(\d+).*')
        match = re.search(pattern, pos)
        return int(match.group(1))


class HearstExtractor:

    # np = r'(JJ[RS]{0,2}\d+ |NN[PS]{0,2}\d+ |IN\d+ |VB[NG]\d+ )*
    # NN[PS]{0,2}\d+'

    # np = r'(JJ[RS]{0,2}\d+ |NN[PS]{0,2}\d+ |VB[NG]\d+ )*NN[PS]{0,2}\d+'
    # <- extended

    # np = r'(NN[PS]{0,2}\d+ )*NN[PS]{0,2}\d+' # <- only_NN
    np = r'\d+np\d+'
    comma = r',\d+'
    conj = r'(or|and)'

    str1 = (r'(?P<hyper>{NP}) such as (?P<hypos>((({NP}) ({Comma} )?)+)'
            r'{Conj} )?(?P<hypo>{NP})')
    str1 = str1.format(NP=np, Comma=comma, Conj=conj)

    str2 = (r'such (?P<hyper>{NP}) as (?P<hypos>({NP} ({Comma} )?)*)'
            r'({Conj} )?(?P<hypo>{NP})')
    str2 = str2.format(NP=np, Comma=comma, Conj=conj)

    str3_4 = (r'(?P<hypo>{NP}) (({Comma} (?P<hypos>{NP}))* ({Comma} )?)?'
              r'{Conj} other (?P<hyper>{NP})')
    str3_4 = str3_4.format(NP=np, Comma=comma, Conj=conj)

    str5_6 = (r'(?P<hyper>{NP}) ({Comma} )?(including |especially )'
              r'(?P<hypos>({NP} ({Comma} )?)*)({Conj} )?(?P<hypo>{NP})')
    str5_6 = str5_6.format(NP=np, Comma=comma, Conj=conj)

    pattern1 = re.compile(str1)
    pattern2 = re.compile(str2)
    pattern3_4 = re.compile(str3_4)
    pattern5_6 = re.compile(str5_6)

    hearst_patterns = [pattern1, pattern2, pattern3_4, pattern5_6]

    hyp_idx_pattern = re.compile(r'(\w+?)(\d+)')
    head_idx_pattern = re.compile(r'(\d+)(\w+)')

    @classmethod
    def get_hierarch_rels(cls,
                          nlp_sent: List[List[str]],
                          level: str
                          ) -> rels_type:
        """Extract and return hierarchical relations from a sentence.

        Args:
            nlp_sent: A list of words. Each word is a tuple of token,
                pos-tag, lemma, is_stop.
            level: 't' if token level, 'l' if lemma level.
        Return:
            A dictionary with the extracted hierarchical relations.
        """
        rel_dict = {}
        rels = cls._get_hypernyms(nlp_sent, level)

        for rel in rels:
            hyper, hypo = rel[0], rel[1]
            if hyper in rel_dict:
                rel_dict[hyper].append(hypo)
            else:
                rel_dict[hyper] = [hypo]

        return rel_dict

    @classmethod
    def _get_hypernyms(cls,
                       sent: List[List[str]],
                       level: str
                       ) -> List[Tuple[str, str]]:
        """Extract hypernym relations from a given sentence.

        Args:
            sent: input sentence
            level: 't' if token level, 'l' if lemma level.
        Return:
            A list of hypernyms and hyponyms as tuples.
            (hypernym at index 0, hyponym at index 1)
        """
        hyp_rels = []
        pw = cls._get_poses_words(sent)
        for p in cls.hearst_patterns:
            matches = [m.groupdict() for m in re.finditer(p, pw)]
            for match in matches:
                if match:
                    rels = cls._get_matches(sent, match, level)
                    for rel in rels:
                        hyp_rels.append(rel)
        return hyp_rels

    @classmethod
    def _get_matches(cls,
                     sent: List[List[str]],
                     match: Dict[str, str],
                     level: str
                     ) -> List[Tuple[str, str]]:
        """Use the result of re.findall to extract matches.

        Tokens are lowercased.

        Args:
            sent: The input sentence.
            match: A dictionary of matched group and it's contents.
            level: 't' if token level, 'l' if lemma level.
        """
        if match['hypos']:
            hypos_matches = re.split(r',\d+', match['hypos'])
            hypos_matches = cls._clean_matches(hypos_matches)
        else:
            hypos_matches = []

        hyper_index = cls._get_hyp_index(match['hyper'])
        hypos_indices = [cls._get_hyp_index(match['hypo'])]
        hypos_indices.extend([cls._get_hyp_index(m) for m in hypos_matches])

        rel_inds = []
        for hypo_inds in hypos_indices:
            rel_inds.append([hyper_index, hypo_inds])

        results = []
        for rel_i in rel_inds:
            hyper_idx = rel_i[0]
            hypo_idx = rel_i[1]
            if level == 't':
                # Additionally lowercase all tokens
                hyper = sent[hyper_idx][0].lower()
                hypo = sent[hypo_idx][0].lower()
            elif level == 'l':
                hyper = cls._get_lemma_form(hyper_idx, sent)
                hypo = cls._get_lemma_form(hypo_idx, sent)
            results.append((hyper, hypo))

        return results

    @classmethod
    def _clean_matches(cls, matches: List[str]) -> List[str]:
        """In case a comma is at the end of the match remove commna.

        Remove comman by only taking the part before empty space as
        a match.
        """
        matches_out = []
        for match in matches:
            if ' ' in match:
                match = match.split(' ')[0]
                matches.append(match)
        return matches_out

    @classmethod
    def _get_lemma_form(cls, term_idx: int, sent: List[List[str]]) -> str:
        """Get the lowercased, lemmatized form of the given term.

        Args:
            term_idx: The index of the term to lemmatize.
            sent: The preprocessed sentence.
        Return:
            The lemmatized form of the term.
        """
        term_head_idx = cls._get_term_head(term_idx, sent)
        term_tokens = sent[term_idx][0].split('_')
        term_lemmas = sent[term_idx][2].split('_')
        term_tokens[term_head_idx] = term_lemmas[term_head_idx]
        return '_'.join(term_tokens).lower()

    @classmethod
    def _get_term_head(cls, term_idx: int, sent: List[List[str]]) -> int:
        pos = sent[term_idx][1]
        match = re.search(cls.head_idx_pattern, pos)
        if match:
            return int(match.group(1))
        print('Warning! Could not extract term-head!')
        print('sentence:', sent)
        print('term_idx:', term_idx)
        return 0

    @classmethod
    def _get_hyp_index(cls, pos_np: str) -> int:
        """Get the index at the end of pos_np.

        Args:
            pos_np: A np-pos-tag with it's index at the end.
        Return:
            The index of the hypernym.
        """
        index = int(re.search(cls.hyp_idx_pattern, pos_np).group(2))
        return index

    @staticmethod
    def _get_poses_words(sent: List[List[str]]) -> str:
        """Merge the token and pos level for Hearst Patters.

        Args:
            sent: input sentence
        Return:
            A mix of pos-tags and lexical words used in Hearst Patterns
        """
        lex_words = ['such', 'as', 'and', 'or',
                     'other', 'including', 'especially']
        poses_words = []
        for i in range(len(sent)):
            word = sent[i]
            if word[0] not in lex_words:
                poses_words.append(word[1] + str(i))
            else:
                poses_words.append(word[0])
        poses_words = ' '.join(poses_words)
        return poses_words


def get_pp_corpus(fpath: str) -> Generator[List[List[List[str]]], None, None]:
    """Yield the documents of the corpus in the given file.

    Load the corpus from a file. There is one sentence per line. Between
    each document there is an additional newline. Words are separated by
    tabs. Token, tag, lemma and isStop of a word are separated by a
    space.

    Each document is yielded as a List of sentences. Each sentence is a
    list of words. Each word is a list of the form
    [token, tag, lemma, isStop].

    Args:
        fpath: The path to the corpus file.
    """
    with open(fpath, 'r', encoding='utf8') as f:
        doc = []
        for line in f:
            if line == '\n':
                if doc:
                    yield doc
                    doc = []
            else:
                sent_str = line.strip('\n')
                words = sent_str.split('\t')
                analyzed_words = [word.split(' ') for word in words]
                doc.append(analyzed_words)


def main():
    from utility_functions import get_config, get_cmd_args
    config = get_config()
    args = get_cmd_args()
    path = config['paths'][args.location][args.corpus]['path_out']
    pe = PatternExtractor(path)
    pe.extract()


if __name__ == '__main__':
    main()
