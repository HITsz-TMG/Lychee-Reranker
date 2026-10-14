import unittest
from unittest.mock import patch

import numpy as np

from lychee_reranker.evaluation.retrieval import dense_search, dense_search_restricted, rerank


class RetrievalTests(unittest.TestCase):
    def test_global_search_preserves_real_faiss_ties_and_numpy_untied_parity(self):
        try:
            import faiss
        except ImportError:
            self.skipTest("FAISS is optional")

        query = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
        doc_ids = ["b", "a", "c"]
        index = faiss.IndexFlatIP(2)
        index.add(documents)
        _, indices = index.search(query, 3)
        actual = dense_search(query, documents, ["q"], doc_ids, top_k=3)
        self.assertEqual(list(actual["q"]), [doc_ids[i] for i in indices[0]])

        documents = np.array([[0.75, 0.0], [0.25, 0.0], [1.0, 0.0]], dtype=np.float32)
        expected = dense_search(query, documents, ["q"], doc_ids, top_k=2)
        with patch.dict("sys.modules", {"faiss": None}):
            fallback = dense_search(query, documents, ["q"], doc_ids, top_k=2)
        self.assertEqual(list(expected["q"]), ["c", "b"])
        self.assertEqual(expected, fallback)

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

    def test_restricted_search_keeps_source_candidate_order_on_ties(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = dense_search_restricted(
            queries,
            documents,
            ["q"],
            ["a", "b", "c"],
            {"q": ["c", "b", "a"]},
            top_k=1,
        )
        self.assertEqual(list(result["q"]), ["c"])

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

    def test_rerank_keeps_source_input_order_on_tied_scores(self):
        class FakeReranker:
            def predict(self, pairs, instruction=None):
                return [1.0] * len(pairs)

        task = type(
            "Task",
            (),
            {
                "name": "fiqa",
                "queries": {"q": "query"},
                "corpus": {"a": "A", "b": "B"},
                "candidates": None,
            },
        )()
        result = rerank(
            task,
            {"q": {"b": 0.5, "a": 0.5}},
            FakeReranker(),
            top_k=2,
        )
        self.assertEqual(list(result["q"]), ["b", "a"])

    def test_restricted_search_requires_every_query(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "coverage"):
            dense_search_restricted(queries, documents, ["q"], ["a"], {}, top_k=1)

    def test_restricted_search_validates_embedding_shape_and_counts(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            dense_search_restricted(
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                ["q"],
                ["d"],
                {"q": ["d"]},
                top_k=1,
            )
        with self.assertRaisesRegex(ValueError, "embedding counts"):
            dense_search_restricted(
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                ["q"],
                ["d"],
                {"q": ["d"]},
                top_k=1,
            )

    def test_rerank_rejects_unknown_documents_before_truncation(self):
        task = type(
            "Task",
            (),
            {
                "name": "fiqa",
                "queries": {"q": "query"},
                "corpus": {"d": "document"},
                "candidates": None,
            },
        )()
        with self.assertRaisesRegex(ValueError, "unknown docs"):
            rerank(task, {"q": {"d": 1.0, "missing": 0.0}}, object(), top_k=1)


if __name__ == "__main__":
    unittest.main()
