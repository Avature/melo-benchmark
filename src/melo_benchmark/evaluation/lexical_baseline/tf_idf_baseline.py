from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from melo_benchmark.evaluation.scorer import LexicalBaselineScorer


# noinspection DuplicatedCode
class WordTfIdfBaselineScorer(LexicalBaselineScorer):

    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        processed_corpus = [
            self._preprocess(x)
            for x
            in c_surface_forms
        ]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)

        scores = []

        for q_surface_form in q_surface_forms:
            processed_query = self._preprocess(q_surface_form)
            query_vector = vectorizer.transform([processed_query])
            scores_q = cosine_similarity(query_vector, tfidf_matrix).flatten()
            scores.append(scores_q)

        return scores


# noinspection DuplicatedCode
class CharTfIdfBaselineScorer(LexicalBaselineScorer):

    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        processed_corpus = [
            self._preprocess(x)
            for x
            in c_surface_forms
        ]

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)

        scores = []

        for q_surface_form in q_surface_forms:
            processed_query = self._preprocess(q_surface_form)
            query_vector = vectorizer.transform([processed_query])
            scores_q = cosine_similarity(query_vector, tfidf_matrix).flatten()
            scores.append(scores_q)

        return scores
