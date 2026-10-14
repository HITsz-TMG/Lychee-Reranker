from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from .prompts import (
    DEFAULT_INSTRUCTION,
    DEFAULT_SYSTEM_INSTRUCTION,
    build_decoder_prompt,
    build_document,
)


class LycheeReranker:
    """Score query-document relevance with a KaLM encoder-decoder reranker.

    Scores are ``P(yes)`` from a two-class softmax over the model's lowercase
    ``yes`` and ``no`` token logits.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 32,
        query_max_length: int = 512,
        max_length: int = 1024,
        chunk_size: Optional[int] = 4,
        instruction: str = DEFAULT_INSTRUCTION,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError("model_name_or_path must be a non-empty string")
        if batch_size <= 0 or query_max_length <= 0 or max_length <= 0:
            raise ValueError("batch and maximum lengths must be positive")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None")
        if not isinstance(instruction, str) or not isinstance(system_instruction, str):
            raise TypeError("instruction and system_instruction must be strings")
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype, self.device)
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.instruction = instruction
        self.system_instruction = system_instruction

        common = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        common = {key: value for key, value in common.items() if value is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must define a pad token or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            dtype=self.dtype,
            **common,
            **model_kwargs,
        )
        # Preserve floating buffers (notably RoPE inv_freq) in checkpoint dtype.
        for parameter in self.model.parameters():
            if parameter.is_floating_point() and parameter.dtype != self.dtype:
                parameter.data = parameter.data.to(dtype=self.dtype)
        self.model.to(device=self.device)
        self.model.eval()
        self.yes_token_id = self._answer_token_id("yes")
        self.no_token_id = self._answer_token_id("no")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return resolved

    @staticmethod
    def _resolve_dtype(
        dtype: Optional[Union[str, torch.dtype]], device: torch.device
    ) -> torch.dtype:
        if dtype is None:
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        if not isinstance(dtype, str):
            raise TypeError("dtype must be a torch.dtype or string")
        values = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        normalized = dtype.lower().removeprefix("torch.")
        if normalized not in values:
            raise ValueError(f"unsupported dtype: {dtype!r}")
        return values[normalized]

    def _answer_token_id(self, answer: str) -> int:
        token_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"failed to tokenize answer {answer!r}")
        return token_ids[-1]

    def _get_encoder(self):
        if hasattr(self.model, "get_encoder"):
            return self.model.get_encoder()
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        raise AttributeError(f"cannot find encoder on {type(self.model).__name__}")

    @staticmethod
    def _pool_encoder_chunks(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        pad_length = num_chunks * chunk_size - sequence_length
        if pad_length:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
            attention_mask = F.pad(attention_mask, (0, pad_length))
        hidden_states = hidden_states.view(batch_size, num_chunks, chunk_size, hidden_size)
        chunk_mask = attention_mask.view(batch_size, num_chunks, chunk_size)
        expanded_mask = chunk_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * expanded_mask).sum(dim=2)
        pooled = pooled / chunk_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        pooled_mask = (chunk_mask.sum(dim=2) > 0).to(attention_mask.dtype)
        return pooled, pooled_mask

    def _decoder_text(self, query: str, instruction: str) -> str:
        query_ids = self.tokenizer(
            query,
            add_special_tokens=False,
            truncation=True,
            max_length=self.query_max_length,
        )["input_ids"]
        query = self.tokenizer.decode(
            query_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return build_decoder_prompt(query, instruction, self.system_instruction)

    @staticmethod
    def _validate_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise TypeError("pairs must be a sequence of (query, document) pairs")
        result: List[Tuple[str, str]] = []
        for index, pair in enumerate(pairs):
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"pairs[{index}] must contain exactly two strings")
            query, document = pair
            if not isinstance(query, str) or not isinstance(document, str):
                raise TypeError(f"pairs[{index}] must contain exactly two strings")
            result.append((query, document))
        return result

    @torch.inference_mode()
    def _predict_batch(self, pairs: Sequence[Tuple[str, str]], instruction: str) -> List[float]:
        encoder_texts = [build_document(document) for _, document in pairs]
        decoder_texts = [self._decoder_text(query, instruction) for query, _ in pairs]
        encoder_batch = self.tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        decoder_batch = self.tokenizer(
            decoder_texts,
            padding=True,
            pad_to_multiple_of=8,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        decoder_attention_mask = decoder_batch["attention_mask"]
        positions = decoder_attention_mask.long().sum(dim=1) - 1
        if (positions < 0).any():
            raise ValueError("decoder batches must contain at least one unpadded token")
        # T5Gemma2 accepts a tensor of decoder positions and returns only those
        # logits.  Deduplicating the positions keeps the output shape small for
        # right-padded batches while preserving the per-example lookup below.
        keep_positions, inverse = torch.unique(positions, sorted=True, return_inverse=True)
        if self.chunk_size is None:
            outputs = self.model(
                input_ids=encoder_batch["input_ids"],
                attention_mask=encoder_batch["attention_mask"],
                decoder_input_ids=decoder_batch["input_ids"],
                decoder_attention_mask=decoder_attention_mask,
                logits_to_keep=keep_positions,
                use_cache=False,
                return_dict=True,
            )
        else:
            encoded = self._get_encoder()(
                input_ids=encoder_batch["input_ids"],
                attention_mask=encoder_batch["attention_mask"],
                return_dict=True,
            )
            pooled, pooled_mask = self._pool_encoder_chunks(
                encoded.last_hidden_state,
                encoder_batch["attention_mask"],
                self.chunk_size,
            )
            outputs = self.model(
                encoder_outputs=BaseModelOutput(last_hidden_state=pooled),
                attention_mask=pooled_mask,
                decoder_input_ids=decoder_batch["input_ids"],
                decoder_attention_mask=decoder_attention_mask,
                logits_to_keep=keep_positions,
                use_cache=False,
                return_dict=True,
            )
        if outputs.logits.ndim != 3 or outputs.logits.shape[0] != len(pairs):
            raise RuntimeError(
                "reranker model must return [batch, selected_decoder_positions, vocab] "
                f"when logits_to_keep is used; got {tuple(outputs.logits.shape)}"
            )
        batch_indices = torch.arange(len(pairs), device=outputs.logits.device)
        last_logits = outputs.logits[batch_indices, inverse.to(outputs.logits.device)]
        yes_no = torch.stack(
            (last_logits[:, self.yes_token_id], last_logits[:, self.no_token_id]), dim=-1
        ).float()
        if not torch.isfinite(yes_no).all():
            count = (~torch.isfinite(yes_no).all(dim=-1)).sum().item()
            raise RuntimeError(f"model produced non-finite yes/no logits for {count} inputs")
        return torch.softmax(yes_no, dim=-1)[:, 0].cpu().tolist()

    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        *,
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Return ``P(yes)`` in the same order as ``pairs``."""
        validated = self._validate_pairs(pairs)
        if not validated:
            return []
        effective_instruction = self.instruction if instruction is None else instruction
        if not isinstance(effective_instruction, str):
            raise TypeError("instruction must be a string or None")
        effective_batch = self.batch_size if batch_size is None else batch_size
        if not isinstance(effective_batch, int) or effective_batch <= 0:
            raise ValueError("batch_size must be a positive integer")

        sorted_indices = np.argsort(
            [-(len(q) + len(d)) for q, d in validated], kind="stable"
        )
        sorted_pairs = [validated[index] for index in sorted_indices]
        tested_batch = effective_batch
        first_scores: Optional[List[float]] = None
        while tested_batch > 1:
            try:
                first_scores = self._predict_batch(
                    sorted_pairs[: min(len(sorted_pairs), tested_batch)], effective_instruction
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tested_batch = max(1, tested_batch * 3 // 4)
        scores: List[float] = [] if first_scores is None else list(first_scores)
        start_at = 0 if first_scores is None else tested_batch
        try:
            for start in range(start_at, len(sorted_pairs), tested_batch):
                scores.extend(
                    self._predict_batch(
                        sorted_pairs[start : start + tested_batch], effective_instruction
                    )
                )
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory; retry with a smaller batch_size or max_length"
            ) from error
        inverse = np.argsort(sorted_indices)
        return [scores[index] for index in inverse]

    def rank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Union[int, float]]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of strings")
        if any(not isinstance(document, str) for document in documents):
            raise TypeError("every document must be a string")
        if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        scores = self.predict(
            [(query, document) for document in documents],
            instruction=instruction,
            batch_size=batch_size,
        )
        ranking: List[Dict[str, Union[int, float]]] = [
            {"corpus_id": index, "score": score} for index, score in enumerate(scores)
        ]
        ranking.sort(key=lambda item: (-float(item["score"]), int(item["corpus_id"])))
        return ranking if top_k is None else ranking[:top_k]


__all__ = ["LycheeReranker"]
