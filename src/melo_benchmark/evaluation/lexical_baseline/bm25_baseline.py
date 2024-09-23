from typing import List

from rank_bm25 import BM25Okapi

from melo_benchmark.evaluation.scorer import LexicalBaselineScorer


class BM25BaselineScorer(LexicalBaselineScorer):

    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        tokenized_corpus = [
            self._preprocess(x).split(" ")
            for x
            in c_surface_forms
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        scores = []

        for q_surface_form in q_surface_forms:
            preproc_query = self._preprocess(q_surface_form)
            tokenized_query = preproc_query.split(" ")
            scores_q = bm25.get_scores(tokenized_query)
            scores.append(scores_q)

        return scores
