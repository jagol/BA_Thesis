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
### Results
-> look up yahoo finance dataset, has gold standart
- precision and recall below 0.66/0.8


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
