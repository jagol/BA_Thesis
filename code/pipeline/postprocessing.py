import os
import csv
from typing import *
from graphviz import Digraph
from utility_functions import get_config, get_cmd_args


nodes_type = Dict[str, Dict[str, List[str]]]


def get_paths() -> Dict[str, str]:
    """Generate paths for postprocessing."""
    config = get_config()
    args = get_cmd_args()
    pout = config['paths'][args.location][args.corpus]['path_out']
    paths = {
        'out': pout,
        'tax_csv': os.path.join(pout, 'hierarchy/taxonomy.csv'),
        'tax_png': os.path.join(pout, 'hierarchy/taxonomy.png')
    }
    return paths


def read_csv(path_tax_csv: str) -> nodes_type:
    """Read the taxonomy from the csv-file.

    Return:
        {node_id: {'child_ids': [1, ...,5], 'terms': [term1, ...]}}
    """
    nodes = {}
    with open(path_tax_csv, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            child_ids = [id_ for id_ in row[1:6] if id_]
            node = {
                'child_ids': child_ids,
                'terms': row[6:][:3]
            }
            nodes[row[0]] = node
    return nodes


def make_dot_tree(fpath, nodes: nodes_type) -> None:
    graph = Digraph(comment='The taxonomy')
    rec_add_nodes('0', nodes, graph)
    graph.render(fpath, view=True)


def rec_add_nodes(source_id: str,
                  nodes: nodes_type,
                  graph: Any
                  ) -> None:
    """Recursively add nodes to the graph."""
    source = nodes[source_id]
    source_text = ' '.join(source['terms'])
    graph.node(source_id, source_text)
    if source['child_ids']:
        for child_id in source['child_ids']:
            child = nodes[child_id]
            child_text = ' '.join(child['terms'])
            graph.node(child_id, child_text)
            graph.edge(source_id, child_id)
        for child_id in source['child_ids']:
            rec_add_nodes(child_id, nodes, graph)


def main():
    paths = get_paths()
    nodes = read_csv(paths['tax_csv'])
    make_dot_tree(paths['tax_png'], nodes)


if __name__ == '__main__':
    main()
