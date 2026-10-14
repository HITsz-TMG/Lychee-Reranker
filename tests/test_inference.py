from types import SimpleNamespace
import unittest

import torch

from lychee_reranker.inference import LycheeReranker


class _Batch(dict):
    def to(self, device):
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        if kwargs.get("return_tensors") is None:
            return {"input_ids": [1, 2]}
        if kwargs.get("padding") is True and texts and str(texts[0]).startswith("<Document>"):
            return _Batch(
                {
                    "input_ids": torch.tensor([[4, 5, 0], [4, 5, 6]]),
                    "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                }
            )
        return _Batch(
            {
                "input_ids": torch.tensor([[7, 8, 0, 0], [7, 8, 9, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            }
        )

    def decode(self, ids, **kwargs):
        return "query"


class _Model:
    def __init__(self):
        self.keep = None

    def __call__(self, **kwargs):
        self.keep = kwargs["logits_to_keep"].detach().cpu().tolist()
        # The selected positions are [1, 2].  The first example should use
        # position 1 and the second should use position 2.
        logits = torch.zeros((2, 2, 5), dtype=torch.float32)
        logits[0, 0, 2] = 2.0
        logits[0, 0, 3] = 0.0
        logits[1, 1, 2] = 0.0
        logits[1, 1, 3] = 2.0
        return SimpleNamespace(logits=logits)


class InferenceTests(unittest.TestCase):
    def test_last_unpadded_logits_are_requested_and_gathered(self):
        reranker = object.__new__(LycheeReranker)
        reranker.tokenizer = _Tokenizer()
        reranker.device = torch.device("cpu")
        reranker.query_max_length = 32
        reranker.max_length = 32
        reranker.chunk_size = None
        reranker.yes_token_id = 2
        reranker.no_token_id = 3
        reranker.system_instruction = "system"
        model = _Model()
        reranker.model = model

        scores = reranker._predict_batch([("q1", "d1"), ("q2", "d2")], "instruction")

        self.assertEqual(model.keep, [1, 2])
        self.assertAlmostEqual(scores[0], float(torch.softmax(torch.tensor([2.0, 0.0]), dim=0)[0]))
        self.assertAlmostEqual(scores[1], float(torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[0]))

    def test_chunk_pooling_respects_right_padding(self):
        hidden = torch.tensor([[[1.0], [3.0], [9.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled, pooled_mask = LycheeReranker._pool_encoder_chunks(hidden, mask, 2)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[[2.0], [0.0]]])))
        self.assertTrue(torch.equal(pooled_mask, torch.tensor([[1, 0]])))


if __name__ == "__main__":
    unittest.main()
