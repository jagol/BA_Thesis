### Deep contextualized word representations
paper: https://arxiv.org/abs/1802.05365
##### Goal
- word embeddings that model syntax and semantics
- account for polysemy
##### Method
- the representation of each token is a function of the entire input sentence (how is it in other models?)
- purely character based input representation
- bidirectional LSTM, coupled with language model objective
- language model: 
  - forward language model: predict token t given the before sequence (past context)
  - backward language model: predict token t given the following sequence (future context)
  - bidirectional: combinde backward and forward LM. In this case: Maximize the log-likelyhood of forward and backward LM (max(log_p(forward)+log_p(backward))) 
- three layers of representation (normally word embedding methods have 1 repr layer)
- architecture:
  - 2 biLSTM layers
  - residual connection from layer 1 to layer 2 (direct connection)
  - 3rd layer: context insensitive type representation: character n-gram convolutional filters followed by two highway layers
  - highway network: allow unimpeded information flow across several layers using "highways" -> so are highway layers layers, that have an information highway "over" it?
  
##### Results
- when plugged into common nlp-tasks like NER, SRL or CR it provides better results than previous embedding methods like context2vec and gloVe
- error reductions from 6% - 20% across different tasks
- experiments confirm that the different layers encode different information 
- encodes more information than word2vec -> what information? -> words are disambiguated (word sense level and on POS-level)

### Domain Ontology Induction using word embeddings
paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7838131&tag=1
##### Goal
- ontology learning pipeline using embeddings

##### Method
- Concept and Taxonomy Identification
  - Extract NPs
  - learn Vector for NP by concatenating the terms of the NP (-> model sees it as one word) and then train model to get a vector repr for the whole NP
  - alternative: generate NP Vectors by adding the vectors of the terms in the NP
  - then recursive divisive clustering:
    - each cluster represents a concept
    - then the cluster is divided again
    - supercluster to subclusters as hypernyms to hyponyms
- Attribute Identification:
  - First approach:
  - coocurrence analysis: "Nationality"  and "person" have high coocurrence -> attribute
  - subject-verb-object frequency analysis to identify non taxonomical relationships (high frequency -> probably a relation)
- Non-Taxonomical Relation identification
  - Second approach:
  - using embeddings: bi-partite matching of two concept-clusters (but has exponential time complexity)
  - not clear how that helps to identify Attributes/Relations! ???

#### Results
-> look up yahoo finance dataset, has gold standart
- precision and recall below 0.66/0.8

### CRIM at SemEval-2018 Task 9: A Hybrid Approach to Hypernym Discovery
paper: http://aclweb.org/anthology/S18-1116

##### Goal
- create model to predict if a term is a hypernym for a given query (concept os NE)

##### Data
- a raw corpus
- small training set of query and its hypernyms

##### Method
- hybrid method using Hearst patterns and projections in Vector Space
- First method: 
  - Hearst pattern based, 
  - Problem: low recall (lots of relations don't occur in the same sentence)
  - Thus: find for each query cohyponyms using patterns and add these to the query
  - Additionally: find hypernyms through multi-word-expressions, take prevailing head-modifier relation as hypernymy
  - Create scores for cohyponyms by multiplying their frequency with their embedding cosine similarity to the original query
  - Use these scores to create score and rank the final hypernym-candidates
- Second Method:
  - supervised learning approach based on word embeddings
  - learn function that takes as input the embedding of a query and a hypernym candidate and output the probability that the candidate is a hypernym (posed as a regression problem, not as a classification problem)
  - how the probabilty is calculated:
    - previous work proposed learning a projection, such that the projection of the embedding of the query is close to the embedding of the hypernym
    - newer version (and used here) learn projection matrices for multiple senses of hypernym relation (get multiple senses by clustering the given query-hyper-pairs based on their vector offset -> learn projection matrik for each cluster)
    - use soft clustering 
- Data augmentation: use given data to generate more training data

##### Results
- first place at SemEval 2019
- up to 40% Mean Average Precision


### Relation Extraction with Matrix Factorization and Universal Schemas
Paper: https://aclanthology.coli.uni-saarland.de/papers/N13-1008/n13-1008
##### Goal
- supervised RE needs labeled data and has a fixed number of relations
- OpenIE has unlimited number of relations but does not generalize well 
(because surface structure = relation)
- thus find method, that has a unlimited number of relations, but also generalizes (and does not need labeled data)
##### Method
- use universal schema: union of all source schemas (= union(surface pattern variants, preexisting relation schemas))
- focus on learning asymmetric implicature


### Cross Sentence N-ary Relation Extraction with Graph LSTMs
Paper: https://arxiv.org/abs/1708.03743
##### Goal
- extract n-ary relations
- relations can either be intra- or inter-sentential
- incorporate sequential, syntactic and discourse relations
##### Methods
- use graph based LSTM
  - subsumes chain and tree based LSTMs
##### Results
