import abc
import os
from typing import (
    Dict,
    List
)
import unicodedata

import numpy as np
from numpy.typing import NDArray
import scipy

from melo_benchmark.utils.lemmatizer import Lemmatizer


class BaseScorer(abc.ABC):

    @abc.abstractmethod
    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        raise NotImplementedError()


class LexicalBaselineScorer(BaseScorer, abc.ABC):

    def __init__(
                self,
                lemmatizer: Lemmatizer = None,
                ascii_normalization: bool = True,
            ):

        self.lemmatizer = lemmatizer
        self.ascii_normalization = ascii_normalization

    def _preprocess(self, text_element: str) -> str:
        text_element = text_element.lower()
        if self.lemmatizer is not None:
            text_element = self.lemmatizer.preprocess(text_element)
        if self.ascii_normalization:
            text_element = unicodedata.normalize('NFKD', text_element)
            text_element = text_element.encode('ASCII', 'ignore')
            text_element = text_element.decode("utf-8")
        return text_element


class BiEncoderScorer(BaseScorer, abc.ABC):

    def __init__(
                self,
                prompt_template: str,
                representation_cache_path: str,
                lowercase: bool = False,
                ascii_normalization: bool = True,
            ):

        self.prompt_template_text = prompt_template
        self.repr_mapping_cache_path = representation_cache_path
        self.lowercase = lowercase
        self.ascii_normalization = ascii_normalization

    def compute_scores(
                self,
                q_ids: List[str],
                q_surface_forms: List[str],
                c_ids: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        all_surface_forms = list(set(q_surface_forms + c_surface_forms))

        sf_repr_mapping = self._build_surface_form_representation_mapping(
            all_surface_forms
        )

        scores = []

        for q_surface_form in q_surface_forms:
            scores_q = []
            for c_surface_form in c_surface_forms:
                q_emb = sf_repr_mapping[q_surface_form]
                c_emb = sf_repr_mapping[c_surface_form]
                s = 1 - scipy.spatial.distance.cosine(q_emb, c_emb)
                scores_q.append(s)

            scores.append(scores_q)

        return scores


    def _build_surface_form_representation_mapping(
                self,
                surface_forms: List[str]
            ) -> Dict[str, NDArray[np.float_]]:

        # Check if the representations mapping cache file already exists
        if os.path.exists(self.repr_mapping_cache_path):
            print(f"Loading representations from cache file...")
            sf_repr_mapping = self._load_mapping_from_cache_file()
            for surface_form in surface_forms:
                if surface_form not in sf_repr_mapping.keys():
                    m = f"Invalid cache file {self.repr_mapping_cache_path}." \
                        + " Delete it before proceeding."
                    raise KeyError(m)
        else:
            # Load and process categories and occupations
            sf_repr_mapping = self._compute_representations(
                surface_forms
            )

        return sf_repr_mapping

    @abc.abstractmethod
    def _compute_representations(
                self,
                surface_forms: List[str]
            ) -> Dict[str, NDArray[np.float_]]:

        raise NotImplementedError()

    def _preprocess(self, text_element: str) -> str:
        if self.lowercase:
            text_element = text_element.lower()
        if self.ascii_normalization:
            text_element = unicodedata.normalize('NFKD', text_element)
            text_element = text_element.encode('ASCII', 'ignore')
            text_element = text_element.decode("utf-8")
        return text_element

    def _render_template(self, **kwargs) -> str:
        rendered_template = self.prompt_template_text
        for kwarg_name, kwarg_val in kwargs.items():
            rendered_template = rendered_template.replace(
                "{{" + kwarg_name + "}}",
                kwarg_val
            )
        return rendered_template

    def _load_mapping_from_cache_file(self) -> Dict[str, NDArray[np.float_]]:
        embeddings_mapping = {}

        with open(self.repr_mapping_cache_path) as f_emb:
            for i, line in enumerate(f_emb):
                t = line.split("\t")
                job_title_name = t[0]
                emb = t[1:]
                embedding = np.array([float(x) for x in emb])
                embeddings_mapping[job_title_name] = embedding

        return embeddings_mapping
