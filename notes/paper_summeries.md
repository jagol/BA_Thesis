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


### Do Supervised Distributional Methods Really Learn Lexical Inference Relations?
paper: https://www.aclweb.org/anthology/N15-1098 2015

##### Goal
- Find out if actual relations like hypernymity or causality are learned. Thesis: It is only learned if a wordpair is a "prototypical hypernym"

##### Method 
- 3 repr Models are testet: SkipGram, PPMI and SVD
- 3 Context Types are testet: window 5 bag of words, positional 2 token window, window defined over syntactic dependency
- For testing 5 labeled datasets are used, that store for a word pair if x entails y
- but some pairs label hypernymy while others label causality
- for compositional methods for entailment are tested:
  - vector concatenation
  - vector difference (offset)
  - only x
  - only y
- train two classifiers: logistic regression and SVM on dataset
- check the results of the classifier

##### Results
- classifiers learn memorize words, that appear typically as hypernyms instead of learning the relation between x and y
- the performance of only y was compared with concatenation/offset and was found to be almost as good as if x was included -> x's information seems to be almost irrelevant for the classifier
- many features of the classifiers turned out to be dataspecific
-> these methods for lexical inference only appear to be learning if y is a prototypical hypernym

### Distributional Hypernym Generation by Jointly Learning Clusters and Projections
paper: https://www.aclweb.org/anthology/C16-1176 2016

##### Goal
- Build a hypernym generation model (instead of a classification model) that jointly learns clusters of hypernymrelations and projections from hyponyms to hypernyms
- thus do not classify if a given pair of terms has a hypernymrelation, instead find a hypernym for a given hyponym
- find different hypernym-relation types through clustering

##### Method
- divide relations into more and more clusters during training
- for each pair, find a cluster and update it's projection matrix according to the cluster

##### Results
- Not so high scores for hypernym generation (MRR = 0.343)
- High scores for hypernym classification (F1 = 0.766 vs state of the art F1 = 0.802)

### Learning Semantic Hierarchies via Word Embeddings
paper: http://ir.hit.edu.cn/~jguo/papers/acl2014-hypernym.pdf 2014

##### Goal
- Create classifier for hypernym relations with word embeddings

##### Method
- piece wise linear projection, by first clustering the relations, classify a pair using the projection
- naive assumption: words can be projected to their hypernyms using a universal transition matrix
- Currently at 3.3.1

### TaxoGen: Unsupervised Topic Taxonomy Construction by Adaptive Term Embedding and Clustering
paper: http://www.kdd.org/kdd2018/accepted-papers/view/taxogen-constructing-topical-concept-taxonomy-by-adaptive-term-embedding-an

##### Goal
- learn a topic taxonomy in an unsupervised way
- topic taxonomy: a taxonomy that doesn't use the actual words as nodes in the taxonomy but finds similar words/synonyms and clusters those into a 'topic'. Each node of the taxonomy represents a topic.

##### Method
- initialize root node containing all terms
- generate topics by spherical clustering
- for each cluster:
  - recognize the most representitive terms for the cluster (by popularity and concentration)
  - all terms over a threshhold, push it to the general terms, remove from cluster
- for lower levels in the taxonomy:
  - train new local embeddings only using the documents belonging to the given topic
  - then do the above calculations

##### Results
- Relation accuracy (True positive parent-child-relations) of 0.775/0.520 


### A Cluster Separation Measure
Paper: https://www.researchgate.net/publication/224377470_A_Cluster_Separation_Measure


### Structured Learning for Taxonomy Induction with Belief Propagation
Paper: https://pdfs.semanticscholar.org/e219/e935d3e9ed32e084fcbedc7ad8ce57abc5f9.pdf


### Cluster merging and splitting in hierarchical clustering algorithms
Paper: https://pdfs.semanticscholar.org/5199/3ada11bf9ebac936d70e6f65cce33bdacc11.pdf
Year: 2002


### A Divisive Information-Theoretic Feature Clustering Algorithm for Text Classification
Paper: http://www.jmlr.org/papers/volume3/dhillon03a/dhillon03a.pdf
Year: 2003


### Hierarchical Clustering of Words and Application to NLP Tasks 
Paper: https://www.aclweb.org/anthology/W96-0103


### Comparing conceptual, divisive and agglomerative clustering for learning taxonomies from text
Paper: https://books.google.ch/books?hl=de&lr=&id=rU_onmzozu0C&oi=fnd&pg=PA435&dq=divisive+clustering+taxonomy&ots=w7lw42V5X7&sig=DZBreE-0UC2uTHPsOLO6Lm4-vPQ#v=onepage&q=divisive%20clustering%20taxonomy&f=false
Year: 2004


### 

### Resources for Hierarchical Clustering
- https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF4820/h13/undervisningsmateriale/07_clustering_handout.pdf
- https://stp.lingfil.uu.se/~santinim/ml/UnsupervisedLearningMagnusRosell_Slides.pdf
- http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering



What I could do:
- divisive clustering on top level
- match HE examples to clusters
- train classifier using clusterhyponyms and their HE extracted Hypernyms
- predict Hypernyms for all clusters that don't have a Hypernym
- train embeddings for lower clusters etc 

- agglomerative clustering
- agglomerate until a known hyponym is in the same cluster as a known hypernym, take last cluster where this wasn't the case
- 



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