import abc
import csv
import os
import tempfile
from typing import (
    Dict,
    List
)
import unicodedata

import numpy as np
from numpy.typing import NDArray
import scipy

try:
    # noinspection PyUnresolvedReferences
    import tensorflow as tf
    tf_is_installed = True
except ImportError:
    pass

try:
    # noinspection PyUnresolvedReferences
    import torch
    torch_is_installed = True
except ImportError:
    pass

from melo_benchmark.utils.lemmatizer import Lemmatizer


class BaseScorer(abc.ABC):

    @abc.abstractmethod
    def compute_scores(
                self,
                q_surface_forms: List[str],
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
                representation_cache_path: str = None,
                lowercase: bool = False,
                ascii_normalization: bool = True,
                batch_size: int = 32,
            ):

        self.prompt_template_text = prompt_template
        self.rep_mapping_cache_path = representation_cache_path
        self.lowercase = lowercase
        self.ascii_normalization = ascii_normalization
        self.batch_size = batch_size

    def register_representation_cache(self, repr_mapping_cache_path: str):
        self.rep_mapping_cache_path = repr_mapping_cache_path

    def compute_scores(
                self,
                q_surface_forms: List[str],
                c_surface_forms: List[str]
            ) -> List[List[float]]:

        all_surface_forms = list(set(q_surface_forms + c_surface_forms))

        sf_repr_mapping = self._build_surface_form_representation_mapping(
            all_surface_forms
        )

        if tf_is_installed and len(tf.config.list_physical_devices('GPU')) > 0:
            return self._compute_scores_gpu_tf(
                q_surface_forms,
                c_surface_forms,
                sf_repr_mapping
            )

        if torch_is_installed and torch.cuda.is_available():
            return self._compute_scores_gpu_torch(
                q_surface_forms,
                c_surface_forms,
                sf_repr_mapping
            )

        # Fallback to computing scores with CPU
        return self._compute_scores_cpu(
            q_surface_forms,
            c_surface_forms,
            sf_repr_mapping
        )

    def pre_compute_embeddings(self, surface_forms: List[str]):
        n = len(surface_forms)
        print(f"Pre-computing embeddings for {n} surface forms...")
        _ = self._build_surface_form_representation_mapping(surface_forms)

    @staticmethod
    def _compute_scores_cpu(
                q_surface_forms: List[str],
                c_surface_forms: List[str],
                embeddings_mapping: Dict[str, NDArray[np.float_]]
            ) -> List[List[float]]:

        scores = []

        for q_surface_form in q_surface_forms:
            scores_q = []
            for c_surface_form in c_surface_forms:
                q_emb = embeddings_mapping[q_surface_form]
                c_emb = embeddings_mapping[c_surface_form]
                s = 1 - scipy.spatial.distance.cosine(q_emb, c_emb)
                scores_q.append(s)

            scores.append(scores_q)

        return scores

    @staticmethod
    def _compute_scores_gpu_torch(
                q_surface_forms: List[str],
                c_surface_forms: List[str],
                embeddings_mapping: Dict[str, NDArray[np.float_]]
            ) -> List[List[float]]:

        assert torch_is_installed

        raise NotImplementedError("Not implemented yet.")

    @staticmethod
    def _compute_scores_gpu_tf(
                q_surface_forms: List[str],
                c_surface_forms: List[str],
                embeddings_mapping: Dict[str, NDArray[np.float_]]
            ) -> List[List[float]]:

        assert tf_is_installed

        q_embs = [
            embeddings_mapping[surface_form]
            for surface_form in q_surface_forms
        ]

        c_embs = [
            embeddings_mapping[surface_form]
            for surface_form in c_surface_forms
        ]

        c_embs_tf = tf.convert_to_tensor(np.array(c_embs), dtype=tf.float32)
        c_embs_norm = tf.nn.l2_normalize(c_embs_tf, axis=1)
        q_embs_tf = tf.convert_to_tensor(np.array(q_embs), dtype=tf.float32)
        q_embs_norm = tf.nn.l2_normalize(q_embs_tf, axis=1)
        scores_all = tf.matmul(q_embs_norm, c_embs_norm, transpose_b=True)

        return scores_all.numpy().tolist()

    def _build_surface_form_representation_mapping(
                self,
                surface_forms: List[str]
            ) -> Dict[str, NDArray[np.float_]]:

        if self.rep_mapping_cache_path is None:
            self.rep_mapping_cache_path = os.path.join(
                tempfile.mkdtemp(),
                "repr_cache.tsv"
            )

        # Check if the representations mapping cache file already exists
        if os.path.exists(self.rep_mapping_cache_path):
            print(f"Loading representations from cache file...")
            sf_repr_mapping = self._load_mapping_from_cache_file(
                surface_forms
            )
        else:
            # Load and process categories and occupations
            sf_repr_mapping = self._compute_representations(
                surface_forms
            )

        return sf_repr_mapping

    @abc.abstractmethod
    def _compute_embeddings(
                    self,
                    rendered_prompts: List[str]
            ) -> List[List[float]]:

        raise NotImplementedError()

    def _compute_representations(
                self,
                surface_forms: List[str]
            ) -> Dict[str, NDArray[np.float_]]:

        rendered_prompts = []
        for surface_form in surface_forms:
            rendered_prompt = self._render_template(
                job_title=surface_form
            )
            rendered_prompts.append(rendered_prompt)

        embeddings = self._compute_embeddings(rendered_prompts)

        sf_repr_mapping = {}

        with open(self.rep_mapping_cache_path, "a", encoding="utf-8") as f_out:
            tsv_writer = csv.writer(f_out, delimiter='\t')

            for surface_form, embedding in zip(surface_forms, embeddings):
                tsv_writer.writerow(
                    [surface_form] + [str(x) for x in embedding]
                )
                embedding = np.array([float(x) for x in embedding])
                sf_repr_mapping[surface_form] = embedding

        return sf_repr_mapping

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

    def _load_mapping_from_cache_file(
                self,
                surface_forms: List[str]
            ) -> Dict[str, NDArray[np.float_]]:

        """
        The cached mapping is required to have embeddings for all the requested
        surface forms, but it is allowed to have embeddings for other surface
        forms. In this way, embeddings for the whole set of surface forms in
        MELO can be pre-computed, in order to minimize the amount of inference
        calls.
        """

        embeddings_mapping = {}
        target_surface_forms = set(surface_forms)

        with open(self.rep_mapping_cache_path, encoding="utf-8") as f_emb:
            tsv_reader = csv.reader(f_emb, delimiter='\t')

            for row in tsv_reader:
                job_title_name = row[0]
                if job_title_name in target_surface_forms:
                    emb = row[1:]
                    embedding = np.array([float(x) for x in emb])
                    embeddings_mapping[job_title_name] = embedding

        existent_surface_forms = set(embeddings_mapping.keys())

        for surface_form in surface_forms:
            if surface_form not in existent_surface_forms:
                m = f"Invalid cache file {self.rep_mapping_cache_path}." \
                    + " Delete it before proceeding."
                raise KeyError(m)

        return embeddings_mapping
