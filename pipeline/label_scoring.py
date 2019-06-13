import os
import json
from math import sqrt
from typing import Any, Dict, List, Tuple, Set
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from hypernym_classification import HypernymClassifier


class LabelScorer:

    def __init__(self, config: Dict[str, Any], args: Any) -> None:
        self.path_out = config['paths'][args.location][args.corpus]['path_out']
        self.path_clf = os.path.join(self.path_out, 'hierarchy/hyp_clf.pickle')
        self.insc = InclusionScorer(self.path_out)
        self.hypsc = HyponymScorer(self.path_out)

    def score(self,
              term_scores: Dict[int, Tuple[str, float]],
              top_parent_scores: List[Tuple[int, float]],
              cos_score: bool=True,
              hypo_score: bool=True,
              incl_score: bool=True
              ) -> Dict[int, Tuple[str, float]]:
        """Compute the label scores for a given cluster in a given taxonomy.

        All nodes in the given taxonomy above the current node are
            assumed to already have been scored.

        Args:
            term_scores: The already calculated cluster-center-similarity.
            top_parent_scores: The top parent scores.
            cos_score: Determines if cosine score should be used.
            hypo_score: Determines if hyponym score should be used.
            incl_score: Determines if distributional inclusion score
                should be used.
        Return:
            {term_id: (term, score)}
        """
        scores = {}
        if hypo_score:
            # term_scores: {term_id: (term, score)}
            # []
            hypo_scores = self.hypsc.score_topic_terms(
                term_scores, top_parent_scores)
        else:
            hypo_scores = {}
        if incl_score:
            incl_scores = self.insc.score_topic_terms(
                term_scores, top_parent_scores)
        else:
            incl_scores = {}

        for term_id in term_scores:
            term = term_scores[term_id][0]
            cos = term_scores[term_id][1]

            if hypo_scores:
                hypo = hypo_scores[term_id]
            else:
                hypo = 1
            if incl_scores:
                incl = incl_scores[term_id]
            else:
                incl = 1
            if cos_score:
                norm_cos = (cos + 1) / 2  # Map cos-sim to [0, 1].
            else:
                norm_cos = 1
            total = sqrt(hypo*incl*norm_cos)
            scores[term_id] = (term, total)

        return scores


class InclusionScorer:

    def __init__(self, path_out: str) -> None:
        self.path_df = os.path.join(path_out, 'frequencies/df_tokens.json')
        d = json.load(open(self.path_df, 'r', encoding='utf8'))
        self.df = {int(k): v for k, v in d.items()}

    def get_parent_docs(self,
                        top_parent_scores: List[Tuple[int, float]]
                        ) -> Set[int]:
        """Get the documents in which the given parent terms occur."""
        parent_docs = []
        for tpl in top_parent_scores:
            term_id = tpl[0]
            parent_docs.extend(self.df[term_id])
        return set(parent_docs)

    def score_topic_terms(self,
                          terms_scores: Dict[int, Tuple[str, float]],
                          top_parent_scores: List[Tuple[int, float]]
                          ) -> Dict[int, float]:
        """Score all terms in a topic.

        Args:
            terms_scores: Maps the term-ids to be scored onto their
                string repr and cosine score.
            top_parent_scores: List of tuples. Each Tuple is a
                parent-term-id and its cos score.
        """
        incl_scores = {}
        parent_docs = self.get_parent_docs(top_parent_scores)
        for term_id in terms_scores:
            term_docs = set(self.df[term_id])
            common_docs = parent_docs.intersection(term_docs)
            df_C_h = len(common_docs)
            df_h = len(term_docs)
            incl = df_C_h / (df_h + 1)
            incl_scores[term_id] = incl
        return incl_scores


class HyponymScorer:

    def __init__(self, path_out: str) -> None:
        self.path_out = path_out
        self.classifier = HypernymClassifier(path_out)
        self.classifier.load()
        self.path_emb = os.path.join(
            path_out, 'embeddings/embs_token_global_Word2Vec.vec')
        self.term_id_to_emb = KeyedVectors.load(self.path_emb)

    def score_topic_terms(self,
                          terms_scores: Dict[int, Tuple[str, float]],
                          top_parent_scores: List[Tuple[int, float]]
                          ) -> Dict[int, float]:
        """Get the hyponym score for all terms in a topic.

        Args:
            terms_scores: Maps the term-ids to be scored onto their
                string repr and cosine score.
            top_parent_scores: List of tuples. Each Tuple is a
                parent-term-id and its cos score.
        """
        # Get parent topic embedding.
        parent_term_embs = [self.term_id_to_emb.wv[str(term_id)]
                            for term_id, score in top_parent_scores]
        hyper_emb = np.mean(parent_term_embs, axis=0)
        # Get the relation embedding.
        num_dims = len(hyper_emb)
        rel_embs = np.zeros((len(terms_scores), num_dims))
        # Map dict to entries in matrix.
        i_to_term_id = {i: term_id for i, term_id in enumerate(terms_scores)}
        term_id_to_i = {term_id: i for i, term_id in i_to_term_id.items()}
        # Fill matrix.
        for term_id in terms_scores:
            hypo_emb = self.term_id_to_emb.wv[str(term_id)]
            rel_embs[term_id_to_i[term_id]] = hypo_emb - hyper_emb
        # Get hyponym scores.
        scores_in_array = self.classifier.classify_prob(rel_embs)
        hypo_scores = {}
        for i in range(len(scores_in_array)):
            term_id = i_to_term_id[i]
            hypo_scores[term_id] = scores_in_array[i]
        return hypo_scores
