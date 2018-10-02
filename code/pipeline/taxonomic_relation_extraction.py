import os
import json
import re
from typing import *
from text_processing_unit import TextProcessingUnit


class HypernymExtractor(TextProcessingUnit):

    def __init__(self,
                 path_in: str,
                 path_out: str,
                 max_files: int = None
                 ) -> None:
        """Initialize Hypernym Extractor.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
            max_files: max number of files to be processed
        """
        self._hypernyms = {}  # {Tuple(str, str): Dict[str, List[int]]}
        super().__init__(path_in, path_out, max_files)

    def extract_hypernyms(self):
        raise NotImplementedError


class HearstHypernymExtractor(HypernymExtractor):

    def __init__(self, path_in: str,
                 path_out: str,
                 max_files: int = None
                 ) -> None:
        np = r'((JJ[RS]{0,2}\d+ )|(NN[PS]{0,2}\d+ ))*NN[PS]{0,2}\d+'
        # To also match all/some: |(P?DT\d+ )
        comma = r',\d+'
        conj = r'(or|and)'

        str1 = (r'(?P<hyper>{NP}) such as (?P<hypos>((({NP}) ({Comma} )?)+)'
                r'{Conj} )?(?P<hypo>{NP})')
        str1 = str1.format(NP=np, Comma=comma, Conj=conj)

        str2 = (r'such (?P<hyper>{NP}) as (?P<hypos>({NP} ({Comma} )?)*)'
                r'({Conj} )?(?P<hypo>{NP})')
        str2 = str2.format(NP=np, Comma=comma, Conj=conj)

        str3_4 = (r'(?P<hypo>{NP}) (?P<hypos>({Comma} {NP})*) ({Comma} )?'
                  r'{Conj} other (?P<hyper>{NP})')
        str3_4 = str3_4.format(NP=np, Comma=comma, Conj=conj)

        str5_6 = (r'(?P<hyper>{NP}) ({Comma} )?(including |especially )'
                  r'(?P<hypos>({NP} ({Comma} )?)*)({Conj} )?(?P<hypo>{NP})')
        str5_6 = str5_6.format(NP=np, Comma=comma, Conj=conj)

        pattern1 = re.compile(str1)
        pattern2 = re.compile(str2)
        pattern3_4 = re.compile(str3_4)
        pattern5_6 = re.compile(str5_6)

        self._patterns = [pattern1, pattern2, pattern3_4, pattern5_6]

        super().__init__(path_in, path_out, max_files)

    def extract_hypernyms(self) -> None:
        """Extract all hypernym-relations using Hearst Patterns.

        For all extracted relations store all their mentions as in
        where they occur.

        Write output to 'hypernyms_hearst.json' in the form:
        {
            (hypernym, hyponym): {}
        }
        """
        for fname in self._fnames:
            fpath = os.path.join(self.path_in, fname)
            with open(fpath, 'r', encoding='utf8') as f:
                sentences = json.load(f)

            self._files_processed += 1
            self._num_sents = len([i for i in sentences])-1

            for i in sentences:
                sent = sentences[i]
                hyper_rels = self._get_hypernyms(sent)
                for rel in hyper_rels:
                    hyper, hypo = rel[0], rel[1]
                    if hyper in self._hypernyms:
                        self._hypernyms[hyper].append(hypo)
                    else:
                        self._hypernyms[hyper] = [hypo]

                self._sents_processed += 1
                self._update_cmd()

            self._sents_processed = 0

        with open('./temp/hypernyms_hearst.json', 'w', encoding='utf8') as f:
            json.dump(self._hypernyms, f)

    @staticmethod
    def _get_poses_words(sent: List[List[Union[str, None]]]) -> str:
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

    def _get_hypernyms(self, sent: List[List[Union[str, bool]]]
                       ) -> List[Tuple[str, str]]:
        """Extract hypernym relations from a given sentence.

        Args:
            sent: input sentence
        Return:
            A list of hypernyms and hyponyms as tuples.
            (hypernym at index 0, hyponym at index 1)
        """
        hyp_rels = []
        pw = self._get_poses_words(sent)
        for p in self._patterns:
            matches = [m.groupdict() for m in re.finditer(p, pw)]
            for match in matches:
                if match:
                    rels = self._get_matches(sent, match)
                    for rel in rels:
                        hyp_rels.append(rel)
        return hyp_rels

    def _get_matches(self,
                     sent: List[List[Union[str, bool]]],
                     match: Dict[str, str]
                     ) -> List[Tuple[str, str]]:
        """Use the result of re.findall to extract matches."""
        if match['hypos']:
            hypos_matches = re.split(r',\d+', match['hypos'])
        else:
            hypos_matches = []
        hyper_indices = self._get_hyp_indices(match['hyper'])
        hypos_indices = [self._get_hyp_indices(match['hypo'])]
        hypos_indices.extend([self._get_hyp_indices(m) for m in hypos_matches])
        rel_inds = []
        for hypo_inds in hypos_indices:
            rel_inds.append([hyper_indices, hypo_inds])

        results = []
        for reli in rel_inds:
            i0 = reli[0]
            i1 = reli[1]
            hyper = ' '.join([sent[i][2] for i in i0])
            hypo = ' '.join([sent[i][2] for i in i1])
            results.append((hyper, hypo))

        return results

    @staticmethod
    def _get_hyp_indices(match: str) -> List[int]:
        """Get the indices of one match."""
        pattern = re.compile(r'(\w+?)(\d+)')
        indices = []
        for pos in match.split(' '):
            if pos and pos[-1].isdigit():
                index = int(re.search(pattern, pos).group(2))
                indices.append(index)
        return indices


def test() -> None:
    """Needs lemmas for test to work."""
    test_he = HearstHypernymExtractor('.', '.')

    sent1 = [['Works', 'NN'], ['such', 'G'], ['as', 'G'], ['Brennan', 'NNP'],
             [',', ','], ['Joyce', 'NNP'], ['or', 'G'], ['Galandi', 'NNP']]
    sent2 = [['Works', 'NN'], ['by', 'G'], ['such', 'G'], ['Authors', 'NNS'],
             ['as', 'G'], ['Brennan', 'NNP'], [',', ','], ['Joyce', 'NNP'],
             ['or', 'G'], ['Galandi', 'NNP']]
    sent3_4 = [['Bruises', 'NN'], [',', ','], ['broken', 'JJS'],
               ['bones', 'NNS'], ['or', 'G'], ['other', 'G'],
               ['Injuries', 'NNP']]
    sent5 = [['All', 'PDT'], ['common', 'JJ'], ['Law', 'NN'],
             ['Countries', 'NNS'], ['especially', 'G'], ['England', 'NN'],
             ['and', 'G'], ['Canada', 'NNP']]

    sents = [sent1, sent2, sent3_4, sent5]

    for i in range(len(sents)):
        print(sents[i])
        print(test_he._get_hypernyms(sents[i]))
        print(30 * '-')


if __name__ == '__main__':
    from utility_functions import get_config
    config = get_config('tax_relation_extraction')
    he = HearstHypernymExtractor(**config)
    he.extract_hypernyms()