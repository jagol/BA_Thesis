import os
import json
import pickle
from typing import *
from numpy import mean
from embeddings import Embeddings


class DocEmbedder:

    def __init__(self, path_out: str, emb_type: str) -> None:
        self.path_out = path_out
        self.emb_type = emb_type
        self.path_term_embs_lemma_w2v = os.path.join(
            self.path_out, 'embeddings/embs_lemma_global_Word2Vec.vec')
        self.path_term_embs_token_w2v = os.path.join(
            self.path_out, 'embeddings/embs_token_global_Word2Vec.vec')
        self.path_term_embs_lemma_glove = os.path.join(
            self.path_out, 'embeddings/embs_lemma_global_GloVe.vec')
        self.path_term_embs_token_glove = os.path.join(
            self.path_out, 'embeddings/embs_token_global_GloVe.vec')
        self.path_tfidf_tokens = os.path.join(
            path_out, 'frequencies/tfidf_tokens.json')
        self.path_tfidf_lemmas = os.path.join(
            path_out, 'frequencies/tfidf_lemmas.json')
        self.path_token_term_idxs = os.path.join(
            path_out, 'processed_corpus/token_terms_idxs.txt')
        self.path_lemma_term_idxs = os.path.join(
            path_out, 'processed_corpus/lemma_terms_idxs.txt')
        self.path_doc_embs_token_w2v = os.path.join(
            path_out, 'embeddings/doc_embs_token_Word2Vec.pickle')
        self.path_doc_embs_lemma_w2v = os.path.join(
            path_out, 'embeddings/doc_embs_lemma_Word2Vec.pickle')
        self.path_doc_embs_token_glove = os.path.join(
            path_out, 'embeddings/doc_embs_token_GloVe.pickle')
        self.path_doc_embs_lemma_glove = os.path.join(
            path_out, 'embeddings/doc_embs_lemma_GloVe.pickle')

    def embed_docs(self):
        """Compute document embeddings.

        Output:
            A Keyed Vector file mapping each doc-id to its embedding.
        """
        print('Calculate token w2v embeddings...')
        self._embed_docs(self.path_term_embs_token_w2v,
                         self.path_token_term_idxs,
                         self.path_tfidf_tokens,
                         self.path_doc_embs_token_w2v)
        print('Calculate lemma w2v embeddings...')
        self._embed_docs(self.path_term_embs_lemma_w2v,
                         self.path_lemma_term_idxs,
                         self.path_tfidf_lemmas,
                         self.path_doc_embs_lemma_w2v)
        print('Calculate token glove embeddings...')
        self._embed_docs(self.path_term_embs_token_glove,
                         self.path_token_term_idxs,
                         self.path_tfidf_tokens,
                         self.path_doc_embs_token_glove)
        print('Calculate lemma glove embeddings...')
        self._embed_docs(self.path_term_embs_lemma_glove,
                         self.path_lemma_term_idxs,
                         self.path_tfidf_lemmas,
                         self.path_doc_embs_lemma_glove)

    def _embed_docs(self,
                    path_term_embs: str,
                    path_term_idxs: str,
                    path_tfidf: str,
                    path_doc_embs: str
                    ) -> None:
        """Calculate Document embeddings."""
        term_idxs = self.load_term_idxs(path_term_idxs)
        term_embs = Embeddings.load_term_embeddings(term_idxs, path_term_embs)
        with open(path_tfidf, 'r', encoding='utf8') as f:
            tfidf = json.load(f)
        doc_embeddings = {}

        for doc_id in tfidf:
            doc_emb = []
            tfidf_doc = tfidf[doc_id]
            if len(tfidf_doc) == 0:
                continue
            for term_id in tfidf_doc:
                term_emb = term_embs[int(term_id)]
                tfidf_term = tfidf_doc[term_id]
                term_emb_weighted = tfidf_term * term_emb
                doc_emb.append(term_emb_weighted)
            doc_embeddings[int(doc_id)] = mean(doc_emb, axis=0)

        with open(path_doc_embs, 'wb') as f:
            pickle.dump(doc_embeddings, f)


    def load_term_idxs(self, path: str) -> Set[int]:
        term_idxs = set()
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                term_idxs.add(int(line.strip('\n')))
        return term_idxs