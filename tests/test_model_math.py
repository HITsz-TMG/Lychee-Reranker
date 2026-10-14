import unittest

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from lychee_reranker.train.model import (
    LycheeRerankerTrainingModel,
    kd_loss,
    mean_pool_chunks,
    sft_loss,
)


class StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [{"no": 1, "yes": 2}.get(text, 0)]}


class StubEncoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict=True):
        self.calls += 1
        return BaseModelOutput(last_hidden_state=self.embedding(input_ids))


class StubSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 3)
        self.encoder = StubEncoder(self.embedding)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        encoder_outputs,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        **kwargs,
    ):
        context = encoder_outputs.last_hidden_state.mean(dim=1)
        logits = self.projection(context).unsqueeze(1).expand(-1, decoder_input_ids.shape[1], -1)
        return Seq2SeqLMOutput(logits=logits)


class ModelMathTests(unittest.TestCase):
    def test_masked_chunk_mean(self):
        hidden = torch.tensor([[[1.0], [3.0], [5.0], [100.0]]])
        mask = torch.tensor([[1, 1, 1, 0]])
        pooled, pooled_mask = mean_pool_chunks(hidden, mask, 2)
        self.assertTrue(torch.equal(pooled_mask, torch.tensor([[1, 1]])))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[[2.0], [5.0]]])))

    def test_sft_labels_first_passage_positive(self):
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        margins = logits[:, 1] - logits[:, 0]
        expected = F.binary_cross_entropy_with_logits(margins, torch.tensor([1.0, 0.0, 1.0, 0.0]))
        self.assertTrue(torch.allclose(sft_loss(logits, 2), expected))

    def test_sft_uses_actual_query_group_count(self):
        logits = torch.zeros(6, 2)
        expected = F.binary_cross_entropy_with_logits(
            logits[:, 1] - logits[:, 0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
        self.assertTrue(
            torch.allclose(sft_loss(logits, train_group_size=3, num_query_groups=2), expected)
        )
        with self.assertRaisesRegex(ValueError, "query groups"):
            sft_loss(logits, train_group_size=3, num_query_groups=1)

    def test_sft_casts_bfloat16_margin_before_bce(self):
        logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.bfloat16)
        expected = F.binary_cross_entropy_with_logits(
            logits.float()[:, 1] - logits.float()[:, 0], torch.tensor([1.0, 0.0])
        )
        self.assertTrue(torch.allclose(sft_loss(logits, train_group_size=2), expected))

    def test_kd_is_literal_soft_label_bce(self):
        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]], requires_grad=True)
        loss = kd_loss(logits, torch.tensor([2.0, -2.0]))
        expected = F.binary_cross_entropy_with_logits(
            logits[:, 1] - logits[:, 0], torch.sigmoid(torch.tensor([2.0, -2.0]))
        )
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_wrapper_forward_backward(self):
        base = StubSeq2Seq()
        wrapper = LycheeRerankerTrainingModel(
            base,
            StubTokenizer(),
            train_group_size=2,
            chunk_sizes=(2, 8),
            chunk_loss_weights=(1.0, 1.0),
        )
        output = wrapper(
            encoder_input_ids=torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 0], [1, 3, 0]]),
            encoder_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
            decoder_input_ids=torch.tensor([[1, 2], [1, 2], [1, 2], [1, 2]]),
            decoder_attention_mask=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]]),
            num_query_groups=torch.tensor(2),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(tuple(output.logits.shape), (4,))
        self.assertEqual(base.encoder.calls, 1)
        output.loss.backward()
        self.assertIsNotNone(base.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
