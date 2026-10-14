import unittest

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from lychee_reranker.train.model import LycheeRerankerTrainingModel, kd_loss, mean_pool_chunks, sft_loss


class StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [{"no": 1, "yes": 2}.get(text, 0)]}


class StubEncoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, attention_mask, return_dict=True):
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
        expected = F.cross_entropy(logits, torch.tensor([1, 0, 1, 0]))
        self.assertTrue(torch.allclose(sft_loss(logits, 2), expected))

    def test_kd_zero_when_logits_match(self):
        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0]], requires_grad=True)
        loss = kd_loss(logits, torch.tensor([2.0, -2.0]))
        self.assertLess(abs(loss.item()), 1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_wrapper_forward_backward(self):
        wrapper = LycheeRerankerTrainingModel(
            StubSeq2Seq(),
            StubTokenizer(),
            train_batch_size=1,
            chunk_sizes=(2,),
            chunk_loss_weights=(1.0,),
        )
        output = wrapper(
            encoder_input_ids=torch.tensor([[1, 2, 0], [3, 4, 0]]),
            encoder_attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]),
            decoder_input_ids=torch.tensor([[1, 2], [1, 2]]),
            decoder_attention_mask=torch.tensor([[1, 1], [1, 1]]),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(tuple(output.logits.shape), (2,))
        output.loss.backward()
        self.assertIsNotNone(wrapper.model.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
