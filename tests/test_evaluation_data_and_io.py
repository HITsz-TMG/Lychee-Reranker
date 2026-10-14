import json
import tempfile
import unittest
from pathlib import Path

from lychee_reranker.evaluation.data import load_beir, load_lmeb, load_miracl
from lychee_reranker.evaluation.io import read_run, stable_digest, write_run


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvaluationDataTests(unittest.TestCase):
    def test_beir_and_candidate_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "fiqa"
            write_jsonl(base / "corpus.jsonl", [{"_id": "d1", "title": "T", "text": "body"}])
            write_jsonl(base / "test_queries.jsonl", [{"_id": "q1", "text": "query"}])
            write_jsonl(
                base / "test_qrels.jsonl", [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
            )
            task = load_beir(root, "fiqa")[0]
            self.assertEqual(task.corpus["d1"], "T\nbody")
            run_path = root / "run.json"
            write_run(
                run_path,
                task=task,
                run={"q1": {"d1": 0.7}},
                kind="candidates",
                model={"id": "toy"},
                config_digest=stable_digest({"x": 1}),
            )
            self.assertEqual(read_run(run_path, expected_task=task), {"q1": {"d1": 0.7}})

    def test_miracl(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "en"
            write_jsonl(base / "corpus.jsonl", [{"id": "d", "title": "", "text": "doc"}])
            write_jsonl(base / "dev_queries.jsonl", [{"id": "q", "text": "query"}])
            write_jsonl(base / "dev_qrels.jsonl", [{"qid": "q", "docid": "d", "relevance": 1}])
            task = load_miracl(Path(tmp), "en")
            self.assertEqual(task.qrels, {"q": {"d": 1.0}})

    def test_lmeb_filters_qrels_to_candidate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = root / "Dialogue" / "ConvoMem" / "case"
            write_jsonl(
                root / "Dialogue" / "ConvoMem" / "corpus.jsonl",
                [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}],
            )
            write_jsonl(
                root / "Dialogue" / "ConvoMem" / "candidates.jsonl",
                [{"scene_id": "s", "candidate_doc_ids": ["d1"]}],
            )
            write_jsonl(leaf / "queries.jsonl", [{"id": "q", "scene_id": "s", "text": "query"}])
            leaf.mkdir(parents=True, exist_ok=True)
            (leaf / "qrels.tsv").write_text("q\td1\t1\nq\td2\t1\n", encoding="utf-8")
            tasks = load_lmeb(root, ["ConvoMem"])
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].qrels, {"q": {"d1": 1.0}})
            self.assertEqual(tasks[0].candidates, {"q": ["d1"]})


if __name__ == "__main__":
    unittest.main()
