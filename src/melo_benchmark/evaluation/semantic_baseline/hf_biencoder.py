import logging
from typing import List

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast
)

from melo_benchmark.evaluation.scorer import BiEncoderScorer
import melo_benchmark.utils.logging_config as melo_logging


melo_logging.setup_logging()
logger = logging.getLogger(__name__)


class HuggingFaceBiEncoderScorer(BiEncoderScorer):

    def __init__(
                self,
                model_name: str,
                prompt_template: str,
                representation_cache_path: str = None,
                lowercase: bool = False,
                ascii_normalization: bool = False,
                batch_size: int = 32,
                low_cpu_mem_usage: bool = True,
                load_in_8bit: bool = False,
            ):

        # This class assumes that the prompt template includes a variable
        #    with key {{job_title}}
        super().__init__(
            prompt_template=prompt_template,
            representation_cache_path=representation_cache_path,
            lowercase=lowercase,
            ascii_normalization=ascii_normalization,
            batch_size=batch_size
        )

        self.model_name = model_name

        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.load_in_8bit = load_in_8bit

        # TODO: support batch_size > 1 for E5 and LLM-based encoders
        if self.batch_size != 1:
            # Override batch_size
            warning_message = (
                f"Batch size for model {self.model_name} loaded with the "
                + f"`HuggingFaceBiEncoderScorer` was set to {self.batch_size} "
                + "but will be overwritten to 1"
            )
            logging.warning(warning_message)
            self.batch_size = 1

        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()

    def _get_tokenizer(self) -> PreTrainedTokenizerFast:
        return AutoTokenizer.from_pretrained(self.model_name)

    def _get_model(self) -> PreTrainedModel:
        return AutoModel.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            load_in_8bit=self.load_in_8bit
        )

    def _free_model_resources(self):
        del self.model

    def _compute_embeddings(
                self,
                rendered_prompts: List[str]
            ) -> List[List[float]]:

        results = []
        for rendered_prompt in rendered_prompts:

            max_length = 4096
            batch_dict = self.tokenizer(
                [rendered_prompt],
                max_length=max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True
            )
            batch_dict['input_ids'] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in batch_dict['input_ids']
            ]
            tokenized_prompt = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            model_device = self.model.device
            tokenized_prompt.to(model_device)
            with torch.no_grad():
                model_output = self.model(**tokenized_prompt)

            prompt_embedding = self._get_last_token_pool(
                model_output.last_hidden_state,
                tokenized_prompt['attention_mask']
            )

            # normalize embeddings
            normalized_prompt_embedding = F.normalize(
                prompt_embedding,
                p=2,
                dim=1
            )

            results.append(normalized_prompt_embedding.squeeze().tolist())

        return results

    @staticmethod
    def _get_last_token_pool(
                last_hidden_states: torch.Tensor,
                attention_mask: torch.Tensor
            ) -> torch.Tensor:

        left_padding = (
                attention_mask[:, -1].sum() == attention_mask.shape[0]
        )
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        batch_size = torch.arange(batch_size, device=last_hidden_states.device)
        result = last_hidden_states[batch_size, sequence_lengths]

        return result
