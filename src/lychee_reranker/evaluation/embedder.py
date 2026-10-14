from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DenseEmbedder:
    """Minimal normalized mean-pooling dense embedder."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path:
            raise ValueError("model_name_or_path must be a non-empty string")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        if dtype is not None and not isinstance(dtype, str):
            raise TypeError("dtype must be a string or None")
        dtype_name = (dtype or ("bfloat16" if self.device.type == "cuda" else "float32")).lower()
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name.removeprefix("torch."))
        if self.dtype is None:
            raise ValueError(f"unsupported dtype: {dtype!r}")
        self.batch_size = batch_size
        self.max_length = max_length
        common = {"revision": revision, "trust_remote_code": trust_remote_code}
        common = {key: value for key, value in common.items() if value is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common)
        self.model = AutoModel.from_pretrained(model_name_or_path, dtype=self.dtype, **common)
        self.model.to(self.device).eval()
        # KaLM's public Qwen2 checkpoint is an encoder-style, bidirectional
        # model even though native Transformers constructs Qwen2 attention
        # modules with ``is_causal=True``.  A plain 2-D padding mask therefore
        # takes the native causal path and silently changes its representations.
        # Passing a prepared 4-D mask makes Transformers treat it as already
        # materialized and preserves the checkpoint's non-causal contract.
        self._use_bidirectional_mask = (
            str(getattr(self.model.config, "model_type", "")).casefold() == "qwen2"
            and getattr(self.model.config, "is_causal", True) is False
        )

    @staticmethod
    def _bidirectional_attention_mask(
        attention_mask: torch.Tensor, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Build an additive 4-D mask for a padded bidirectional sequence.

        The mask blocks both padded keys and padded queries.  Valid tokens can
        attend to every other valid token, including tokens to their right.
        Native Qwen2 accepts this prepared mask without rebuilding a causal
        mask, which is needed for KaLM's ``is_causal=False`` checkpoint.
        """
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if not dtype.is_floating_point:
            raise TypeError("bidirectional attention mask dtype must be floating point")
        valid = attention_mask.to(torch.bool)
        allowed = valid[:, None, :, None] & valid[:, None, None, :]
        mask = torch.zeros(
            (*allowed.shape[:1], 1, *allowed.shape[2:]),
            dtype=dtype,
            device=attention_mask.device,
        )
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled.float(), p=2, dim=-1)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a sequence of strings")
        batches: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            model_inputs = dict(encoded)
            if self._use_bidirectional_mask:
                model_inputs["attention_mask"] = self._bidirectional_attention_mask(
                    encoded["attention_mask"], dtype=self.dtype
                )
                model_inputs["use_cache"] = False
            outputs = self.model(**model_inputs, return_dict=True)
            pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
            if not torch.isfinite(pooled).all():
                raise RuntimeError("embedder produced non-finite embeddings")
            batches.append(pooled.cpu().numpy())
        if not batches:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def encode_queries(
        self, queries: Sequence[str], instruction: Optional[str] = None
    ) -> np.ndarray:
        if instruction:
            queries = [f"Instruct: {instruction}\nQuery: {query}" for query in queries]
        return self.encode(queries)

    def encode_corpus(self, documents: Sequence[str]) -> np.ndarray:
        return self.encode(documents)
