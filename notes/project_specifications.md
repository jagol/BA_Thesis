###  Basic Goals
- build unsupervised OL-pipeline by relying only on embedding-derived features for all processing steps
- ontology domain: economic and political organizations and their relations
- intrinsic evaluation of resulting ontology (using manual annotation and class richness/inheritance)
- basic questions to answer: 
  - Is it possible to build a fully unsupervised ontology learning pipeline for the domain of organizations in politics and economics? 
  - What is the quality of the resulting ontology?
  - (How do the results compare to a supervised pipeline?)
  - Which parts of the pipeline perform better/worse than their supervised counterparts?

### Extended Goals
- compare different embedding-types (ELMO vs GloVe vs ...)
  - compare results for single steps (as features in TE, RE, ...)
  - compare the resulting ontology
- n-ary relation extraction instead of binary relation extraction (how to insert into ontology?)
- extrinsic evaluation of resulting ontology in a question answering task

### Methodology
##### Ontology-Learning pipeline in steps:
- Term Extraction: TF-IDF, c-value-Method. Enhance recall by using euclidian and jaccard/manhattan distance.
- Taxonomic Relation Extraction: find patterns in vectorspace, vector offset, linear projections (http://ir.hit.edu.cn/~car/papers/acl14embedding.pdf) (could also be used for Tax. Const.)
- Taxonomy Construction: use hierarchical clustering and/or minimum cost flow approach.
- Non-Taxonomic Relations Extraction: use dependency parser to extract triples (-> OpenIE?) and filter for relevance, Relation Extraction with Matrix Factorization (https://www.aclweb.org/anthology/N13-1008)

##### Evaluation:
- calculate class richness and class inheritance 
- manual, intrinsic evaluation:
  - extract a certain number of randomly selected hypernym/non-taxonomic relations and check if they are correct
  - calculate recall (use samplesize to get a confidence value)
  - randomly select relations from the text and check if they are present in the ontology
  - calculate precision (use samplesize to get a confidence value)
  - calculate estimated accuracy and f1-score (and calculate confidence given the confidence values of recall and precision)
- extrinsic evaluation:
  - use library to build a parser from natural language query to sparql query
  - find resources with lots of queries about relations between organizations
  - build question answerig system
  - (manually) evaluate the correctnes of the answer
  - calculate accuary and f1-score

### Top-Level-Ontology Candidates
- https://www.w3.org/TR/vocab-org/ <-
- http://epimorphics.com/public/vocabulary/org.html

### Corpus Candidates
- http://www.statmt.org/europarl/ <-
- https://www.gpo.gov/fdsys/bulkdata
- https://corpus.byu.edu/wikipedia.asp create virtual corpus on topic economics/government/eu to extract more abstract concepts
