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
  - 3. layer: context insensitive type representation: character n-gram convolutional filters followed by two highway layers
  - highway network: allow unimpeded information flow across several layers using "highways" -> so are highway layers layers, that have an information highway "over" it?
  
##### Results
- when plugged into common nlp-tasks like NER, SRL or CR it provides better results than previous embedding methods like context2vec and gloVe
- error reductions from 6% - 20% across different tasks
- experiments confirm that the different layers encode different information 
- encodes more information than word2vec -> what information? -> words are disambiguated (word sense level and on POS-level)

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
