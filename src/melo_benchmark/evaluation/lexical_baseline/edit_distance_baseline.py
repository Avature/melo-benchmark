from typing import List

from rapidfuzz import fuzz

from melo_benchmark.evaluation.scorer import BaseScorer


class EditDistanceBaselineScorer(BaseScorer):

    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        scores = []

        for q_surface_form in q_surface_forms:
            scores_q = []
            for c_surface_form in c_surface_forms:
                s = fuzz.ratio(
                    q_surface_form.lower(),
                    c_surface_form.lower()
                )
                scores_q.append(s)

            scores.append(scores_q)

        return scores
