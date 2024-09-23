import random
from typing import List

from melo_benchmark.evaluation.scorer import BaseScorer


class RandomBaselineScorer(BaseScorer):

    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        scores = []

        for _ in q_surface_forms:
            scores_q = []
            for _ in c_surface_forms:
                s = random.random()
                scores_q.append(s)

            scores.append(scores_q)

        return scores
