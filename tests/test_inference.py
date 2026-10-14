from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
    def _scorer(self, callback):
        reranker = object.__new__(LycheeReranker)
        reranker.batch_size = 5
        reranker.instruction = "instruction"
        reranker._predict_batch = callback
        return reranker

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

    def test_oom_backoff_to_one_preserves_every_input_and_order(self):
        pairs = [("q" * length, str(index)) for index, length in enumerate((1, 5, 2, 4, 3))]
        attempts = []

        def score(batch, instruction):
            attempts.append(len(batch))
            if len(batch) > 1:
                raise torch.cuda.OutOfMemoryError("synthetic capacity limit")
            return [int(document) / 10 for _, document in batch]

        reranker = self._scorer(score)
        with patch("torch.cuda.is_available", return_value=False):
            result = reranker.predict(pairs)
        self.assertEqual(result, [0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertEqual(attempts, [5, 3, 2, 1, 1, 1, 1, 1])

    def test_successful_probe_is_not_scored_twice(self):
        pairs = [("q" * length, str(index)) for index, length in enumerate((1, 5, 2, 4, 3))]
        visited = []

        def score(batch, instruction):
            visited.extend(document for _, document in batch)
            return [int(document) / 10 for _, document in batch]

        self.assertEqual(self._scorer(score).predict(pairs), [0.0, 0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(visited), len(set(visited)))

    def test_model_errors_are_not_misclassified_as_oom(self):
        attempts = []

        def score(batch, instruction):
            attempts.append(batch)
            raise RuntimeError("invalid attention shape")

        with self.assertRaisesRegex(RuntimeError, "invalid attention shape"):
            self._scorer(score).predict([("query", "document")])
        self.assertEqual(len(attempts), 1)

    def test_rank_ties_preserve_corpus_indices_and_top_k(self):
        reranker = self._scorer(lambda batch, instruction: [0.5] * len(batch))
        result = reranker.rank("q", ["long document", "short", "another document"], top_k=2)
        self.assertEqual([row["corpus_id"] for row in result], [0, 1])
        self.assertEqual(reranker.rank("q", []), [])
        self.assertEqual(reranker.rank("q", ["document"], top_k=0), [])

    def test_invalid_public_inputs_are_rejected_before_scoring(self):
        def score(batch, instruction):
            self.fail("invalid inputs reached the model")

        reranker = self._scorer(score)
        with self.assertRaises(TypeError):
            reranker.predict("query document")
        with self.assertRaises(ValueError):
            reranker.predict([("only query",)])
        with self.assertRaises(TypeError):
            reranker.predict([("query", None)])
        with self.assertRaises(ValueError):
            reranker.predict([("query", "document")], batch_size=0)
        with self.assertRaises(ValueError):
            reranker.rank("q", ["document"], top_k=-1)


if __name__ == "__main__":
    unittest.main()
