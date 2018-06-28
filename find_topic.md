# Find Topic for BA-thesis

Ideas for possible topics:

### Unsupervised OL-Pipeline using Elmo

ELMo Link: https://allennlp.org/elmo

Idea (as far as understood): create a fully unsupervised ontology-learning pipeline. 

Questions:
* Can ELMo really help to make all OL-steps (Term-Extraction, Taxonomic-Relation-Extraction, Non-Taxonomic-Relation-Extraction...) unsupervised?
  * for Term-Extraction: only use similarity measures from ELMo
  * for Taxonomic-Relation-Extraction: find vectors that represent taxonomic relations???
  * Non-Taxonomic-Relation-Extraction: ???

### Use Inductive-Logic-Programming and NodeToVec to find axioms

Idea: [This paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/89D6CA2D72EB2A8CE36EB9992C4863D2/S1471068407003195a.pdf/building_rules_on_top_of_ontologies_for_the_semantic_web_with_inductive_logic_programming.pdf) introduces ILP to derive axioms from ontologies. Enhance the methods by [NodeToVec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

Other resources:
* https://pdfs.semanticscholar.org/presentation/068d/4625fbb0ef09a4bb10ce5fd7b478a21fd509.pdf
* https://pageperso.lis-lab.fr/bernard.espinasse/PUBLIS-BE-PDF/ACTI/%5BACTI%2013.4%5D-Lima-DEXA%202013.pdf

### Test binary vs ternary relations on concrete ontology

Idea: OWL only has binary relations. A lot of sentences are intransitive, thereby have ternary relations. While ternary relations can be encoded in binary relations (for example as properties of the relation), this seems to be suboptimal. To test if including ternary relations would help in maintaining/creating more expressive ontologies with actual use, take binary ontologies find implicit ternary relations, create an ontology with explicit ternary relations and then test in a downstream task, if the ontologies with ternary relations are of more use.

Relevant resources: 
* https://eprints.nottingham.ac.uk/10708/1/thesis3.pdf
* https://www.w3.org/TR/swbp-n-aryRelations/

Alternative: 
How would a relation extraction method for ternary relations look like? Try to convert methods for extracting binary relations to extract ternary relations. Test the resultung ontologies (binary vs ternary) against each other.

### NodeToVec for Ontology Matching

Idea: Create a domain neutral ontology integration tool. Use [NodeToVec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) as a way to match concepts/nodes in the ontologies. Combine this tecnique with other methods such as Leveshtein, PathSimilarity???, etc. 

Focus on: How much does NodeToVec help in matching concepts/nodes from two ontologies.

Possible procedure: 
* normalize all labels in both ontologies (stemming, ...)
* add to all labels an identifier to identify which ontology it belongs to
* find one to one matches in the labels
* use Node2Vec to project labels from ontologies into a vectorspace (the labels of each ontology get their own vectorspace)
* then start iterative process (breadth-first-algorithm-like):
  * start from one to one matching nodes and go to neighboring nodes
  * for each neighboring node, get the distance to known node
  * now go to the other ontology, add the distance from before to the corresponding node and check for possible matches
  * use additional metrics levenshtein to classify as match or non-match
  * repeat for all found matches until no node is untested for a match
  
But what to do with non-matches?

Compare results with techniques presented in: 
https://jbiomedsem.biomedcentral.com/track/pdf/10.1186/2041-1480-5-21 

Other resources:
* https://www.sciencedirect.com/science/article/pii/S0957417414005144
* https://pdfs.semanticscholar.org/32f3/fc09c880448e1688dc269d03c399cf4f5c55.pdf
* Node2Vec-implementation in python: https://github.com/aditya-grover/node2vec

Addition: Is also axiom matching possible using embeddings?
Idea: Find axioms of both ontologies that have the same logic structure. Then use word embeddings to determine the similarity between the concepts is high enough. Maybe a combined similarity of the whole axiom by adding/multplying the similarity of the concepts can be used.

### Ever expanding ontology

For a very specific domain, that's for example regurarly in the news (example ISIS) create an ontology pipeline that constantly scrapes the web for news and updates and evaluates the ontology automatically. 

Expansion: automatically identify topics in the news and decide if an ontology should be created for them. Then continiously update the ontology when news about the topic appears.

parts for the expanded version:
* web-scraper 
* topic modelling
* classifier to decide if topic should get an ontology
* ontology learning pipeline
* update mechanism
* automatic quality evaluation

Resources: No recources???
