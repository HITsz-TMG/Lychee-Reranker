import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lychee_reranker.evaluation.cli import (
    _aggregate_lmeb_groups,
    _aggregate_rows,
    _instruction,
    parse_args,
    run,
)
from lychee_reranker.evaluation.data import EvaluationTask
from lychee_reranker.train.config import load_config


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
    def test_beir_stage_prompts_match_source_maps(self):
        task = EvaluationTask(
            name="msmarco",
            benchmark="beir",
            split="dev",
            corpus={},
            queries={},
            qrels={},
        )
        self.assertEqual(
            _instruction(task, {}, stage="retrieve"),
            "Given a web search query, retrieve relevant passages that answer the query.",
        )
        self.assertEqual(
            _instruction(task, {}, stage="rerank"),
            "Given a query, retrieve documents that answer the query.",
        )
        task.name = "cqadupstack/android"
        self.assertEqual(
            _instruction(task, {}, stage="retrieve"),
            "Given a question, retrieve detailed question descriptions from Stackexchange "
            "that are duplicates to the given question.",
        )
        self.assertEqual(_instruction(task, {"instruction": "custom"}, stage="retrieve"), "custom")

    def test_beir_config_uses_msmarco_dev_and_identical_id_contract(self):
        config = load_config(Path(__file__).parents[1] / "configs" / "eval" / "beir.yaml")
        self.assertEqual(config["split"], "test")
        self.assertEqual(config["splits"], {"msmarco": "dev"})
        self.assertIs(config["ignore_identical_ids"], True)

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
                patch(
                    "lychee_reranker.evaluation.cli._build_embedder", return_value=FakeEmbedder()
                ),
                patch(
                    "lychee_reranker.evaluation.cli._build_reranker", return_value=FakeReranker()
                ),
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

    def test_lmeb_group_macro_is_separate_from_leaf_rows(self):
        rows = {
            "Dialogue/ConvoMem/easy": {"ndcg@10": 0.2},
            "Dialogue/LoCoMo/case": {"ndcg@10": 0.6},
            "Semantic/SciFact/test": {"ndcg@10": 1.0},
        }
        groups = _aggregate_lmeb_groups(rows)
        self.assertAlmostEqual(groups["ConvoMem"]["ndcg@10"], 0.2)
        self.assertAlmostEqual(groups["LoCoMo"]["ndcg@10"], 0.6)
        self.assertAlmostEqual(groups["SciFact"]["ndcg@10"], 1.0)

    def test_cli_model_overrides(self):
        args = parse_args(
            [
                "--config",
                "eval.yaml",
                "--reranker-name-or-path",
                "local/reranker",
                "--embedder-name-or-path",
                "local/embedder",
            ]
        )
        self.assertEqual(args.reranker_name_or_path, "local/reranker")
        self.assertEqual(args.embedder_name_or_path, "local/embedder")

    def test_lmeb_leaf_reuse_and_task_group_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "ConvoMem"
            write_jsonl(
                data / "corpus.jsonl",
                [
                    {"id": "good", "text": "relevant"},
                    {"id": "bad", "text": "noise"},
                ],
            )
            write_jsonl(
                data / "candidates.jsonl",
                [
                    {"scene_id": "q", "candidate_doc_ids": ["good", "bad"]},
                ],
            )
            for leaf in ("demo", "second"):
                write_jsonl(data / leaf / "queries.jsonl", [{"id": "q", "text": "question"}])
                (data / leaf / "qrels.tsv").write_text("qid\tdocid\tscore\nq\tgood\t2\n")
            (root / "task_instructions.json").write_text(
                json.dumps(
                    {
                        "ConvoMem/demo": {"query": "Remember this."},
                        "ConvoMem/second": {"query": "Remember this."},
                    }
                )
            )
            config = {
                "benchmark": "lmeb",
                "data_dir": str(root / "data"),
                "output_dir": str(root / "out"),
                "tasks": ["ConvoMem"],
                "metrics": ["ndcg@10"],
                "embedder": {"model_name_or_path": "fake"},
                "reranker": {"model_name_or_path": "fake"},
            }
            with (
                patch(
                    "lychee_reranker.evaluation.cli._build_embedder", return_value=FakeEmbedder()
                ),
                patch(
                    "lychee_reranker.evaluation.cli._build_reranker", return_value=FakeReranker()
                ),
            ):
                first = run(config, "all")
                second = run(config, "rerank", candidate_dir=str(root / "out" / "candidates"))
            self.assertEqual(first, second)
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(set(summary["group_aggregate"]), {"ConvoMem"})
            self.assertAlmostEqual(summary["macro_average"]["ndcg@10"], 1.0)


if __name__ == "__main__":
    unittest.main()
