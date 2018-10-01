import os
import json
import re
from typing import *


class HypernymExtractor:

    def __init__(self, path_in: str, path_out: str, max_files: int = None
                 ) -> None:
        """Initialize Hypernym Extractor.

        Args:
            path_in: path to corpus, can be file or directory
            path_out: path to output directory
            max_files: max number of files to be processed
        """
        self.path_in = path_in
        self.path_out = path_out
        self._max_files = max_files
        self._files_processed = 0
        self._sents_processed = 0
        self._fnames = [fname for fname in os.listdir(self.path_in)
                        if os.path.isfile(os.path.join(self.path_in, fname))]
        self._fnames.sort()
        self._num_files = len(self._fnames)
        self._max_files = max_files
        self._upper_bound = self._get_upper_bound()
        self._hypernyms = {}    # {Tuple(str, str): Dict[str, List[int]]}

        if not os.path.isdir(self.path_out):
            os.makedirs(self.path_out)

    def _get_upper_bound(self) -> int:
        if self._max_files:
            return min(self._max_files, self._num_files)
        return self._num_files

    def extract_hypernyms(self):
        raise NotImplementedError


class HearstHypernymExtractor(HypernymExtractor):

    def __init__(self, path_in: str, path_out: str, max_files: int = None
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

    def extract_hypernyms(self):
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
            for i in sentences:
                sent = sentences[i]
                hyper_rels = self._get_hypernyms(sent)
                for rel in hyper_rels:
                    hyper, hypo = rel[0], rel[1]
                    if hyper in self._hypernyms:
                        self._hypernyms[hyper].append(hypo)
                    else:
                        self._hypernyms[hyper] = [hypo]

        with open('./temp/hypernyms_hearst.json', 'w', encoding='utf8') as f:
            json.dump(self._hypernyms, f)

        # for pattern in self._patterns:
        #     match = re.search(pattern, poses_words)
        #     if match:
        #         print(pattern)
        #         print(match.groupdict())

    @staticmethod
    def _get_poses_words(sent: List[Tuple[Union[str, None]]]) -> str:
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

    def _get_hypernyms(self, sent: List[Tuple[Union[str, bool]]]
                       ) -> List[Tuple[str]]:
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

    def _get_matches(self, sent: List[Tuple[Union[str, bool]]], match: Tuple[str])-> Tuple[str]:
        """Use the result of re.findall to extract matches."""
        hyper_poses = match['hyper'].split(' ')
        pattern = re.compile(r'(\w+)(\d+)')
        hyper_indices = []
        for pos in hyper_poses:
            index = int(re.search(pattern, pos).group(2))
            hyper_indices.append(index)
        hypo_indices = []
        for pos in match['hypo'].split(' '):
            index = int(re.search(pattern, pos).group(2))
            hypo_indices.append(index)

        hypos_indices = []
        if match['hypos']:
            for hypo in match['hypos'].split(' '):
                l = []
                if len(hypo) > 2:
                    if hypo[-1].isdigit():
                        index = int(re.search(pattern, hypo).group(2))
                        l.append(index)
                    hypos_indices.append(l)

        matches = [[hyper_indices, hypo_indices]]

        for hypo_ind in hypos_indices:
            matches.append([hyper_indices, hypo_ind])

        results = []
        for match in matches:
            i0 = match[0]
            i1 = match[1]
            hyper = ' '.join([sent[i][0] for i in i0])
            hypo = ' '.join([sent[i][0] for i in i1])
            results.append((hyper, hypo))

        return results




if __name__ == '__main__':

    he = HearstHypernymExtractor('./preprocessed_corpus/', './temp/', max_files=2)
    # he.extract_hypernyms()

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
    poses_words1 = he._get_poses_words(sent1)
    poses_words2 = he._get_poses_words(sent2)
    poses_words3_4 = he._get_poses_words(sent3_4)
    poses_words5 = he._get_poses_words(sent5)

    for i in range(len(sents)):
        print(sents[i])
        print(he._get_hypernyms(sents[i]))
        print(30*'-')
