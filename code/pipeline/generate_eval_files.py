import csv
import os
import json
import argparse
from random import sample
from typing import Dict, Any, List, Tuple, Union


"""Generate evaluation files."""


term_type = Tuple[int, str, int]


class EFGenerator:
    """Class to handle functions to generate evaluation files."""

    def __init__(self, path_in: str, path_out: str) -> None:
        # self.path_in = path_in
        # self.path_taxonomy = os.path.join(self.path_in, 'tax_labels_sim.csv')
        self.path_taxonomy = path_in
        self.path_out = path_out
        self.path_prev_labels = ('../evaluation/results/evaluations/'
                                 'previous_labels')
        self.taxogen_tax = self.load_taxogen_tax()
        self.prev_hypernym_labels = self.load_prev_hypernym_labels()
        self.prev_subtopic_labels = self.load_prev_subtopic_labels()
        self.taxonomy = self.read_taxonomy()
        self.taxonomy[0] = {'terms': [], 'child_ids': [1, 2, 3, 4, 5]}
        self.hyp_rels = []  # List of relations (hypernym, hyponym)
        self.subtopic_rels = []
        self.topic_rels = []

    # ---------- Methods to load eval-files ----------

    @staticmethod
    def load_taxogen_tax():
        """Load the taxonomy created by the taxogen paper.

        Load the top 5 topics.
        """
        with open('taxogen_tax.json', 'r', encoding='utf8') as f:
            return json.load(f)

    def load_prev_hypernym_labels(self) -> Dict[Tuple[str, str], int]:
        """Load previously evaluated hypernym relations.

        Format of csv:
        Hypernym COMMA Hyponym COMMA 0/1
        """
        inf = os.path.join(self.path_prev_labels, 'prev_hypernym_labels.csv')
        prev_hyp_labels = {}
        with open(inf, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                hyper = row[0]
                hypo = row[1]
                label = row[2]
                prev_hyp_labels[(hyper, hypo)] = label
        return prev_hyp_labels

    def load_prev_subtopic_labels(self):
        """Load previously evaluated subtopic relations."""
        inf = os.path.join(self.path_prev_labels, 'prev_issubtopic_labels.csv')
        prev_subt_labels = {}
        with open(inf, 'r', encoding='utf8') as f:
            reader = csv.reader(f)
            for row in reader:
                topic = row[0]
                subt = row[1]
                label = row[2]
                prev_subt_labels[(topic, subt)] = label
        return prev_subt_labels

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
                if node_id > 643:  # stop at depth 4
                    continue
                child_ids = [int(idx) for idx in row[1:6]]
                terms = row[6:]
                terms = [tuple(term.split('|')) for term in terms]
                taxonomy[node_id] = {'child_ids': child_ids, 'terms': terms}
        return taxonomy

    # ---------- Methods to generate eval-files ----------

    def generate_eval_files(self) -> None:
        """Generate 3 eval files under the directory path_out.

        Files generated:
            hyponym_eval.csv
            issubtopic.csv
            topic_level.csv
        """
        self.generate_hypernym_eval_file()
        self.generate_issubtopic_eval_file()
        self.generate_topic_level_eval_file()

    def generate_hypernym_eval_file(self):
        """Generate a csv file to evaluate the hypernym relation precision."""
        n = 100
        top_n = 1
        self.rec_get_hyp_rels(0, top_n)
        rels_subset = sample(self.hyp_rels, n)
        self.write_hyp_rels_to_file(rels_subset)

    def generate_issubtopic_eval_file(self):
        """Generate a csv file to evaluate issubtopic relation precision."""
        n = 100
        top_n = 3
        self.rec_get_subtopic_rels(0, top_n)
        rels_subset = sample(self.subtopic_rels, n)
        self.write_subtopic_rels_to_file(rels_subset)

    def generate_topic_level_eval_file(self):
        """Generate the evaluation file for topic level evaluation."""
        self.rec_get_topic_rels(0)
        # [(str, str)] each string is a comma-separated concatenation
        # of the 10 top terms of a topic
        self.write_topic_rels_to_file()

    # ---------- Methods to recursively generate relations ----------

    def rec_get_hyp_rels(self, node_id: int, top_n: int) -> None:
        """Recursively get all relation from node_id down the tree.

        Args:
            node_id: The id of the top node to get the relations.
            top_n: The top_n terms of each topic are chosen for relation
                term pairs.
        """
        hyper_terms = self.taxonomy[node_id]['terms']
        hypo_terms = []

        for child_id in self.taxonomy[node_id]['child_ids']:
            if child_id not in self.taxonomy:
                return
            hypo_terms.extend(self.taxonomy[child_id]['terms'][:top_n])

        for hyper in hyper_terms:
            for hypo in hypo_terms:
                self.hyp_rels.append((hyper, hypo))

        for child_id in self.taxonomy[node_id]['child_ids']:
            self.rec_get_hyp_rels(child_id, top_n)

    def rec_get_subtopic_rels(self, node_id: int, top_n: int) -> None:
        """Recursively get all relations from node_id down the tree.

        Args:
            node_id: The id of the top node to get the relations.
            top_n: The top_n terms of each topic are chosen for relation
                term pairs.
        """
        topic_terms = self.taxonomy[node_id]['terms']
        subtopic_terms = []

        for child_id in self.taxonomy[node_id]['child_ids']:
            if child_id not in self.taxonomy:
                return
            subtopic_terms.extend(self.taxonomy[child_id]['terms'][:top_n])

        for hyper in topic_terms:
            for hypo in subtopic_terms:
                self.subtopic_rels.append((hyper, hypo))

        for child_id in self.taxonomy[node_id]['child_ids']:
            self.rec_get_subtopic_rels(child_id, top_n)

    def rec_get_topic_rels(self,
                           node_id: int
                           ) -> Union[List[Tuple[str, str]], None]:
        """Generate hyponym-hypernym pairs on a topic level.

        Args:
            node_id: The node id to start with.

        Return:
            A list of topic pairs. Each topic is a string concatenation
            of the top 10 terms of the topic.
        """
        if node_id != 0:
            hypernym_terms = [t[1] for t in self.taxonomy[node_id]['terms']]
            hypernym_terms_str = ', '.join(hypernym_terms)
            hyponym_list = []
            for child_id in self.taxonomy[node_id]['child_ids']:
                if child_id not in self.taxonomy:
                    return
                hyponym = [t[1] for t in self.taxonomy[child_id]['terms']]
                hyponym_str = ', '.join(hyponym)
                hyponym_list.append(hyponym_str)
            topic_relations = [(hypernym_terms_str, hypo) for hypo in
                               hyponym_list]
            topic_rels_full = [(hyper, hypo) for hyper, hypo in topic_relations
                               if hyper and hypo]  # no empty rels
            self.topic_rels.extend(topic_rels_full)
        for child_id in self.taxonomy[node_id]['child_ids']:
            self.rec_get_topic_rels(child_id)

    # ---------- Methods for file writing ----------

    def write_topic_rels_to_file(self):
        """Write topic relations to file."""
        outf = os.path.join(self.path_out, 'topic_level_eval.csv')
        with open(outf, 'w', encoding='utf8') as f:
            writer = csv.writer(f)
            for rel in self.topic_rels:
                row = [rel[0], rel[1]]
                writer.writerow(row)

    def write_hyp_rels_to_file(self,
                               rels_subset: List[Tuple[term_type, term_type]]
                               ) -> None:
        """Write the given hyponym relations to a csv-file.

        Each row has the form:
        hyper_idx,hyper_name,hyper_score,hypo_idx,hyper_name,hyper_score
        """
        outf = os.path.join(self.path_out, 'hypernym_eval.csv')
        with open(outf, 'w', encoding='utf8') as f:
            writer = csv.writer(f)
            for rel in rels_subset:
                hpe_idx, hpe_name, hpe_sim = rel[0]
                hpo_idx, hpo_name, hpo_sim = rel[1]
                tpl = (hpe_name, hpo_name)
                if tpl in self.prev_hypernym_labels:
                    label = self.prev_hypernym_labels[tpl]
                    row = [hpe_idx, hpe_name, hpe_sim, hpo_idx, hpo_name,
                           hpo_sim, label]
                else:
                    row = [hpe_idx, hpe_name, hpe_sim, hpo_idx, hpo_name,
                           hpo_sim]
                writer.writerow(row)

    def write_subtopic_rels_to_file(
            self,
            rels_subset: List[Tuple[term_type, term_type]]
            ) -> None:
        """Write the given subtopic relations to a csv-file.

        Each row has the form:
        hyper_idx,hyper_name,hyper_score,hypo_idx,hyper_name,hyper_score
        """
        outf = os.path.join(self.path_out, 'issubtopic_eval.csv')
        with open(outf, 'w', encoding='utf8') as f:
            writer = csv.writer(f)
            for rel in rels_subset:
                hpe_idx, hpe_name, hpe_sim = rel[0]
                hpo_idx, hpo_name, hpo_sim = rel[1]
                tpl = (hpe_name, hpo_name)
                if tpl in self.prev_subtopic_labels:
                    label = self.prev_subtopic_labels[tpl]
                    row = [hpe_idx, hpe_name, hpe_sim, hpo_idx, hpo_name,
                           hpo_sim, label, label]
                else:
                    row = [hpe_idx, hpe_name, hpe_sim, hpo_idx, hpo_name,
                           hpo_sim]
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--path_in_dir',
        help='Path to taxonomy directory.',
    )
    parser.add_argument(
        '-o',
        '--path_out_dir',
        help='Path to the output directory.'
    )
    args = parser.parse_args()
    e = EFGenerator(args.path_in_dir, args.path_out_dir)
    e.generate_eval_files()


if __name__ == '__main__':
    main()
