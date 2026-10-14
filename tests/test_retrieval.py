import unittest
from unittest.mock import patch

import numpy as np

from lychee_reranker.evaluation.retrieval import dense_search, dense_search_restricted, rerank


class RetrievalTests(unittest.TestCase):
    def test_dense_search_and_deterministic_tie(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = dense_search(queries, documents, ["q"], ["b", "a", "c"], top_k=2)
        self.assertEqual(list(result["q"]), ["a", "b"])

    def test_restricted_search(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
        result = dense_search_restricted(
            queries, documents, ["q"], ["a", "b"], {"q": ["b"]}, top_k=100
        )
        self.assertEqual(list(result["q"]), ["b"])

    def test_numpy_fallback_is_available_without_faiss(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
        with patch.dict("sys.modules", {"faiss": None}):
            result = dense_search(queries, documents, ["q"], ["a", "b"], top_k=2)
        self.assertEqual(list(result["q"]), ["a", "b"])

    def test_ignore_identical_ids_fetches_extra_then_trims(self):
        result = dense_search(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.9, 0.0], [0.1, 1.0]], dtype=np.float32),
            ["q"],
            ["q", "d1", "d2"],
            top_k=2,
            ignore_identical_ids=True,
        )
        self.assertEqual(list(result["q"]), ["d1", "d2"])

    def test_ignore_identical_ids_applies_to_rerank_candidate_reuse(self):
        class FakeReranker:
            def predict(self, pairs, instruction=None):
                return [1.0] * len(pairs)

        task = type(
            "Task",
            (),
            {
                "name": "fiqa",
                "queries": {"q": "query"},
                "corpus": {"q": "self", "d": "document"},
                "candidates": None,
            },
        )()
        result = rerank(
            task,
            {"q": {"q": 2.0, "d": 1.0}},
            FakeReranker(),
            top_k=2,
            ignore_identical_ids=True,
        )
        self.assertEqual(result, {"q": {"d": 1.0}})

    def test_restricted_search_requires_every_query(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "coverage"):
            dense_search_restricted(queries, documents, ["q"], ["a"], {}, top_k=1)


if __name__ == "__main__":
    unittest.main()
