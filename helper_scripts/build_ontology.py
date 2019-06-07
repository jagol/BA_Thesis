def build_ontology():
    path = '.'
    corpus = read_json_corpus(path)
    processed_corpus = process_corpus(corpus)
    domain_terms = get_domain_terms(processed_corpus)
    taxonomic_relations = get_hierarchical_relations(
        processed_corpus, domain_terms)
    taxonomies = construct_taxonomies(taxonomic_relations)
    non_taxonomic_relations = get_non_taxonomic_relations(
        processed_corpus, domain_terms)
    write_ontology(taxonomies, non_taxonomic_relations)


if __name__ == '__main__':
    build_ontology()