import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lychee_reranker.evaluation.cli import _aggregate_rows, run


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FakeEmbedder:
    model_name_or_path = "fake-embedder"
    revision = "test"

    def encode_queries(self, texts, instruction=None):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_corpus(self, texts):
        return np.array(
            [[1.0, 0.0] if "relevant" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class FakeReranker:
    model_name_or_path = "fake-reranker"
    revision = "test"
    chunk_size = 4

    def predict(self, pairs, instruction=None):
        return [0.9 if "relevant" in document else 0.1 for _, document in pairs]


class EvaluationPipelineTests(unittest.TestCase):
    def test_all_then_candidate_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "fiqa"
            write_jsonl(
                data / "corpus.jsonl",
                [
                    {"_id": "good", "title": "", "text": "relevant"},
                    {"_id": "bad", "title": "", "text": "noise"},
                ],
            )
            write_jsonl(data / "test_queries.jsonl", [{"_id": "q", "text": "question"}])
            write_jsonl(
                data / "test_qrels.jsonl",
                [{"query-id": "q", "corpus-id": "good", "score": 1}],
            )
            config = {
                "benchmark": "beir",
                "data_dir": str(root / "data"),
                "output_dir": str(root / "out"),
                "tasks": ["fiqa"],
                "metrics": ["ndcg@10", "recall@100"],
                "embedder": {"model_name_or_path": "fake"},
                "reranker": {"model_name_or_path": "fake"},
            }
            with (
                patch("lychee_reranker.evaluation.cli._build_embedder", return_value=FakeEmbedder()),
                patch("lychee_reranker.evaluation.cli._build_reranker", return_value=FakeReranker()),
            ):
                first = run(config, "all")
                second = run(config, "rerank", candidate_dir=str(root / "out" / "candidates"))
            self.assertEqual(first, second)
            self.assertEqual(first["fiqa"]["ndcg@10"], 1.0)

    def test_cqadupstack_is_one_aggregate_dataset(self):
        rows = {
            "cqadupstack/a": {"ndcg@10": 0.2},
            "cqadupstack/b": {"ndcg@10": 0.4},
            "fiqa": {"ndcg@10": 0.5},
        }
        aggregate = _aggregate_rows(rows)
        self.assertEqual(set(aggregate), {"cqadupstack", "fiqa"})
        self.assertAlmostEqual(aggregate["cqadupstack"]["ndcg@10"], 0.3)


if __name__ == "__main__":
    unittest.main()
