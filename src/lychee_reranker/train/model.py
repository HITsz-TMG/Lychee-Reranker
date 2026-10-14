from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


def mean_pool_chunks(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, chunk_size: int
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
    mask = attention_mask.view(batch_size, num_chunks, chunk_size)
    pooled = (hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)).sum(dim=2)
    pooled = pooled / mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
    pooled_mask = (mask.sum(dim=2) > 0).to(attention_mask.dtype)
    return pooled, pooled_mask


def _group_size_for_pairs(
    pair_count: int,
    *,
    train_group_size: Optional[int],
    train_batch_size: Optional[int],
    num_query_groups: Optional[int] = None,
) -> int:
    """Resolve the query-group shape without assuming a fixed device batch.

    ``train_batch_size`` is retained as a compatibility argument for callers of
    the first extraction.  New training code supplies the fixed number of
    passages per query (``train_group_size``) and infers the actual query count
    from the flattened batch.
    """
    if pair_count <= 0:
        raise ValueError("the flattened batch must contain at least one pair")
    if train_group_size is not None:
        group_size = int(train_group_size)
        if group_size < 2:
            raise ValueError("train_group_size must be at least 2")
        if pair_count % group_size:
            raise ValueError("pair count must be divisible by train_group_size")
        actual_groups = pair_count // group_size
    elif train_batch_size is not None:
        batch_size = int(train_batch_size)
        if batch_size <= 0 or pair_count % batch_size:
            raise ValueError("pair count must be divisible by train_batch_size")
        group_size = pair_count // batch_size
        if group_size < 2:
            raise ValueError("each query must contain at least one negative passage")
        actual_groups = batch_size
    else:
        raise ValueError("supply train_group_size when computing the training loss")
    if num_query_groups is not None and int(num_query_groups) != actual_groups:
        raise ValueError(
            f"collator reported {int(num_query_groups)} query groups, "
            f"but the flattened batch contains {actual_groups}"
        )
    return group_size


def sft_loss(
    yes_no_logits: torch.Tensor,
    train_batch_size: Optional[int] = None,
    *,
    train_group_size: Optional[int] = None,
    num_query_groups: Optional[int] = None,
) -> torch.Tensor:
    """Compute the release SFT objective as binary BCE on yes-minus-no logits.

    Every flattened query group has its first passage marked positive and all
    remaining passages marked negative.  The explicit group size keeps the
    objective correct when the last dataloader batch is smaller than the
    configured per-device batch.
    """
    if yes_no_logits.ndim != 2 or yes_no_logits.shape[1] != 2:
        raise ValueError("yes_no_logits must have shape [pairs, 2] in [no, yes] order")
    group_size = _group_size_for_pairs(
        yes_no_logits.shape[0],
        train_group_size=train_group_size,
        train_batch_size=train_batch_size,
        num_query_groups=num_query_groups,
    )
    logits = yes_no_logits.float()
    labels = torch.zeros(yes_no_logits.shape[0], dtype=logits.dtype, device=yes_no_logits.device)
    labels[::group_size] = 1.0
    student_margin = logits[:, 1] - logits[:, 0]
    return F.binary_cross_entropy_with_logits(student_margin, labels)


def kd_loss(
    yes_no_logits: torch.Tensor, teacher_scores: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = yes_no_logits.float()
    student = (logits[:, 1] - logits[:, 0]).reshape(-1)
    teacher = torch.as_tensor(teacher_scores, device=student.device, dtype=torch.float32).reshape(
        -1
    )
    if student.shape != teacher.shape:
        raise ValueError(f"student shape {student.shape} does not match teacher {teacher.shape}")
    teacher_probability = torch.sigmoid(teacher.detach() / temperature)
    # Appendix D Eq. (10) is the literal soft-label BCE.  Do not subtract the
    # teacher entropy: that produces the KL-equivalent scalar and changes the
    # reported training objective even though its student gradient matches.
    return (
        F.binary_cross_entropy_with_logits(student / temperature, teacher_probability)
        * temperature**2
    )


class LycheeRerankerTrainingModel(nn.Module):
    """Minimal trainable wrapper around a seq2seq language model."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        train_group_size: Optional[int] = None,
        train_batch_size: Optional[int] = None,
        chunk_sizes: Sequence[int] = (2, 8, 32, 128),
        chunk_loss_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        knowledge_distillation: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(chunk_sizes) != len(chunk_loss_weights) or not chunk_sizes:
            raise ValueError("chunk sizes and weights must have the same non-zero length")
        if any(size <= 0 for size in chunk_sizes) or any(
            weight < 0 for weight in chunk_loss_weights
        ):
            raise ValueError("chunk sizes must be positive and weights non-negative")
        if sum(chunk_loss_weights) <= 0:
            raise ValueError("at least one chunk weight must be positive")
        self.model = model
        self.tokenizer = tokenizer
        if train_group_size is None and train_batch_size is None:
            raise ValueError("train_group_size is required for training")
        self.train_group_size = int(train_group_size) if train_group_size is not None else None
        self.train_batch_size = int(train_batch_size) if train_batch_size is not None else None
        self.chunk_sizes = tuple(int(size) for size in chunk_sizes)
        self.chunk_loss_weights = tuple(float(weight) for weight in chunk_loss_weights)
        self.knowledge_distillation = knowledge_distillation
        self.temperature = temperature
        self.yes_token_id = self._answer_token_id("yes")
        self.no_token_id = self._answer_token_id("no")

    def _answer_token_id(self, answer: str) -> int:
        ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"failed to tokenize {answer!r}")
        return ids[-1]

    def _encoder(self):
        return (
            self.model.get_encoder() if hasattr(self.model, "get_encoder") else self.model.encoder
        )

    def _yes_no_logits(
        self,
        encoder_hidden: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        pooled, pooled_mask = mean_pool_chunks(encoder_hidden, encoder_attention_mask, chunk_size)
        positions = decoder_attention_mask.long().sum(dim=1) - 1
        keep_positions, inverse = torch.unique(positions, sorted=True, return_inverse=True)
        outputs = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=pooled),
            attention_mask=pooled_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            # T5Gemma2 computes only these decoder positions, avoiding a full
            # [batch, sequence, vocabulary] training allocation.
            logits_to_keep=keep_positions,
            use_cache=False,
            return_dict=True,
        )
        rows = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
        logits = outputs.logits[rows, inverse]
        return torch.stack(
            (logits[:, self.no_token_id], logits[:, self.yes_token_id]), dim=-1
        ).contiguous()

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        teacher_scores: Optional[torch.Tensor] = None,
        num_query_groups: Optional[torch.Tensor | int] = None,
    ) -> RerankerOutput:
        encoded = self._encoder()(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        losses = []
        scores = []
        for chunk_size in self.chunk_sizes:
            logits = self._yes_no_logits(
                encoded.last_hidden_state,
                encoder_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                chunk_size,
            )
            scores.append(torch.softmax(logits.float(), dim=-1)[:, 1])
            if self.training:
                if self.knowledge_distillation:
                    if teacher_scores is None:
                        raise ValueError("teacher_scores are required for knowledge distillation")
                    losses.append(kd_loss(logits, teacher_scores, self.temperature))
                else:
                    groups = None
                    if num_query_groups is not None:
                        groups = (
                            int(num_query_groups.item())
                            if torch.is_tensor(num_query_groups)
                            else int(num_query_groups)
                        )
                    losses.append(
                        sft_loss(
                            logits,
                            self.train_batch_size,
                            train_group_size=self.train_group_size,
                            num_query_groups=groups,
                        )
                    )
        weights = torch.tensor(
            self.chunk_loss_weights, device=scores[0].device, dtype=scores[0].dtype
        )
        normalized = weights / weights.sum()
        merged_scores = sum(weight * score for weight, score in zip(normalized, scores))
        loss = None
        if self.training:
            # Deliberately not normalized: this matches the release training code.
            loss = sum(weight * chunk_loss for weight, chunk_loss in zip(weights, losses))
        return RerankerOutput(loss=loss, logits=merged_scores)

    def enable_input_require_grads(self) -> None:
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        method = getattr(self.model, "gradient_checkpointing_enable", None)
        if method is None:
            raise RuntimeError("the wrapped model does not support gradient checkpointing")
        if gradient_checkpointing_kwargs is None:
            method()
        else:
            method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.enable_input_require_grads()

    def gradient_checkpointing_disable(self) -> None:
        method = getattr(self.model, "gradient_checkpointing_disable", None)
        if method is not None:
            method()

    def save_pretrained(self, output_dir: str, **kwargs) -> None:
        self.model.save_pretrained(output_dir, **kwargs)
        self.tokenizer.save_pretrained(output_dir)
