import csv
import os
import json
import argparse
from typing import Dict, Any, Tuple


"""Test how much the generated taxonomy overlaps with the taxonomy
generated by the TaxoGen team.

python3 process_eval.py -t ../evaluation/results/taxonomies/all_true_like_taxogen_depth_3_th_0.35_aggl_ward_eucl.csv -e ../evaluation/results/evaluations/all_true_like_taxogen_depth_3_th_0.35_aggl_ward_eucl
"""


term_type = Tuple[int, str, int]


class Evaluator:
    """Calculate relation precision and update previous labels."""

    def __init__(self, path_eval_files: str, path_taxonomy: str) -> None:
        self.path_eval_files = path_eval_files
        self.path_taxonomy = path_taxonomy
        self.path_prev_labels = ('../evaluation/results/evaluations/'
                                 'previous_labels')
        self.prev_hyper_labels = self.load_prev_hyper_labels()
        self.prev_issubt_labels = self.load_prev_issubt_labels()
        self.taxogen_tax = self.load_taxogen_tax()
        self.taxonomy = self.read_taxonomy()
        self.taxonomy[0] = {'terms': [], 'child_ids': [1, 2, 3, 4, 5]}

    # ---------- Methods to load eval-files ----------

    @staticmethod
    def load_taxogen_tax() -> Dict[str, Dict[str, str]]:
        """Load the taxonomy created by the taxogen paper.

        Load the top 5 topics.
        """
        with open('taxogen_tax.json', 'r', encoding='utf8') as f:
            return json.load(f)

    def read_taxonomy(self) -> Dict[int, Dict[str, Any]]:
        """Read the taxonomy from csv input file.

        Output:
            {node-id: {'child_ids': list of child-ids, 'terms': list of
                terms, each term of the form (id, name, score)}}
        """
        taxonomy = {}
        with open(self.path_taxonomy, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                node_id = int(row[0])
                if node_id > 643:
                    continue
                # get num subtopics
                num_subtopics = 0
                for i in range(1, 6):
                    try:
                        int(row[i])
                        num_subtopics += 1
                    except ValueError:
                        break
                child_ids = [int(idx) for idx in row[1:1+num_subtopics]]
                terms = row[1+num_subtopics:]
                terms = [tuple(term.split('|')) for term in terms]
                taxonomy[node_id] = {'child_ids': child_ids, 'terms': terms}
        return taxonomy

    def load_prev_hyper_labels(self) -> Dict[Tuple[str, str], int]:
        prev_labels = {}  # {(hyper, hypo): label}
        path_prev_labels = os.path.join(
            self.path_prev_labels, 'prev_hypernym_labels.csv')
        with open(path_prev_labels, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                hyper, hypo, label = row[0], row[1], int(row[2])
                prev_labels[(hyper, hypo)] = label
        return prev_labels

    def load_prev_issubt_labels(self) -> Dict[Tuple[str, str], int]:
        prev_labels = {}  # {(parent, subt): label}
        path_prev_labels = os.path.join(
            self.path_prev_labels, 'prev_issubtopic_labels.csv')
        with open(path_prev_labels, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                parent, subt, label = row[0], row[1], int(row[2])
                prev_labels[(parent, subt)] = label
        return prev_labels

    # ---------- Methods to compute evaluation scores ----------

    def compute_eval_scores(self):
        """Compute evaluation scores."""
        path_hypo = os.path.join(
            self.path_eval_files, 'hypernym_eval.csv')
        path_issubtopic = os.path.join(
            self.path_eval_files, 'issubtopic_eval.csv')
        path_topic_level = os.path.join(
            self.path_eval_files, 'topic_level_eval.csv')

        print(10*'-' + ' TaxoGen similarity ' + 10*'-')
        self.compute_taxogen_sim()
        print(10 * '-' + ' Hypernym relation precision ' + 10 * '-')
        self.compute_rel_prec(path_hypo, topic_level=False)
        print(10 * '-' + ' Issubtopic relation precision ' + 10 * '-')
        self.compute_rel_prec(path_issubtopic, topic_level=False)
        print(10 * '-' + ' Topic level relation precision ' + 10 * '-')
        self.compute_rel_prec(path_topic_level, topic_level=True)

    def compute_taxogen_sim(self):
        """Compute the similarity to the taxonomy generated by taxogen.

        Divide the number of labels in both taxonomies at level 1 or 2
        by the total number of labels in the first two levels.
        """
        top_level_terms = list(self.taxogen_tax.keys())
        second_level_terms = []
        for tlt in top_level_terms:
            for slt in self.taxogen_tax[tlt]:
                second_level_terms.append(slt)
        taxogen_terms = set(top_level_terms + second_level_terms)
        tax_terms = []
        for chid1 in self.taxonomy[0]['child_ids']:
            if len(self.taxonomy[chid1]['terms']) > 0:
                tax_terms.append(self.taxonomy[chid1]['terms'][0][1])
            child_ids = self.taxonomy[chid1]['child_ids']
            for chid2 in child_ids:
                if chid2 in self.taxonomy:
                    if len(self.taxonomy[chid2]['terms']) > 0:
                        tax_terms.append(self.taxonomy[chid2]['terms'][0][1])
        taxonomy_terms = set(tax_terms)
        common = taxonomy_terms.intersection(taxogen_terms)
        similarity = len(common) / len(taxogen_terms)
        # print('TaxoGen vs own implementation:')
        msg = 'total: {}, common: {}, similarity: {}'
        print(msg.format(len(taxogen_terms), len(common), similarity))

    @staticmethod
    def compute_rel_prec(fname: str, topic_level: bool = False) -> None:
        """Compute the relation precision for a given annotation.

        Look for a file named 'eval_rel_prec.csv' in this directory.
        Every row has the form:
        Each row has the form:
        hyper_idx,hyper_name,hyper_score,
        hypo_idx,hyper_name,hyper_score,0/1
        0/1 denotes a true or false relation.
        """
        num_true = 0
        total = 0
        with open(fname, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                if topic_level:
                    true_or_not = int(row[2])
                else:
                    true_or_not = int(row[6])
                num_true += true_or_not
                total += 1
        num_false = total - num_true
        precision = num_true / total
        msg = 'total: {}, true: {}, false: {}, precision: {}'
        print(msg.format(total, num_true, num_false, precision))

    def update_previous_evals(self):
        """Update hypernym-eval and issubtopic-eval."""
        self.update_previous_hypernym_evals()
        self.update_previous_issubtopic_evals()

    def update_previous_hypernym_evals(self) -> None:
        """Update previous hypernym-eval."""
        path_hypo = os.path.join(
            self.path_eval_files, 'hypernym_eval.csv')
        hyper_labels = {}
        with open(path_hypo, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                hyper, hypo, label = row[1], row[4], int(row[6])
                hyper_labels[(hyper, hypo)] = label
        self.prev_hyper_labels.update(hyper_labels)
        self.write_prev_hyper_labels()

    def write_prev_hyper_labels(self):
        """Write previous hypernym labels to file."""
        path_prev_labels = os.path.join(
            self.path_prev_labels, 'prev_hypernym_labels.csv')
        with open(path_prev_labels, 'w', encoding='utf8') as f:
            writer = csv.writer(f)
            for tpl, key in self.prev_hyper_labels.items():
                hyper, hypo = tpl[0], tpl[1]
                row = [hyper, hypo, key]
                writer.writerow(row)

    def update_previous_issubtopic_evals(self) -> None:
        """Update previous issubtopic-eval."""
        path_issubtopic = os.path.join(
            self.path_eval_files, 'issubtopic_eval.csv')
        issubt_labels = {}
        with open(path_issubtopic, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                parent, subt, label = row[1], row[4], int(row[6])
                issubt_labels[(parent, subt)] = label
        self.prev_issubt_labels.update(issubt_labels)
        self.write_prev_issubtopic_labels()

    def write_prev_issubtopic_labels(self):
        """Write previous subtopic labels to file."""
        path_prev_labels = os.path.join(
            self.path_prev_labels, 'prev_issubtopic_labels.csv')
        with open(path_prev_labels, 'w', encoding='utf8') as f:
            writer = csv.writer(f)
            for tpl, key in self.prev_issubt_labels.items():
                parent, subt = tpl[0], tpl[1]
                row = [parent, subt, key]
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--path_eval_files',
        help='Path to the directory with the evaluation files.',
    )
    parser.add_argument(
        '-t',
        '--path_taxonomy',
        help='Path to the taxonomy.'
    )
    args = parser.parse_args()
    e = Evaluator(args.path_eval_files, args.path_taxonomy)
    e.compute_eval_scores()
    e.update_previous_evals()


if __name__ == '__main__':
    main()
