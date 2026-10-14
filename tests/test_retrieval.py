import unittest
from unittest.mock import patch

import numpy as np

from lychee_reranker.evaluation.retrieval import dense_search, dense_search_restricted


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

    def test_restricted_search_requires_every_query(self):
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        documents = np.array([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "coverage"):
            dense_search_restricted(queries, documents, ["q"], ["a"], {}, top_k=1)


if __name__ == "__main__":
    unittest.main()
