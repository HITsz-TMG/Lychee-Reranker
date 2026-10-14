import json
import math
import tempfile
import unittest
from pathlib import Path

from lychee_reranker.evaluation.data import load_beir, load_lmeb, load_miracl
from lychee_reranker.evaluation.io import read_run, stable_digest, write_run


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvaluationDataTests(unittest.TestCase):
    def _write_lmeb_leaf(self, root, leaf_name="easy", *, with_qrels=True):
        task_root = root / "Dialogue" / "ConvoMem"
        leaf = task_root / leaf_name
        write_jsonl(task_root / "corpus.jsonl", [{"id": "d1", "text": "one"}])
        write_jsonl(task_root / "candidates.jsonl", [{"scene_id": "s", "candidate_doc_ids": ["d1"]}])
        write_jsonl(leaf / "queries.jsonl", [{"id": "q", "scene_id": "s", "text": "query"}])
        if with_qrels:
            (leaf / "qrels.tsv").write_text("q\td1\t1\n", encoding="utf-8")

    def test_lmeb_requested_group_and_leaf_coverage_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_lmeb_leaf(root)
            with self.assertRaisesRegex(FileNotFoundError, "LoCoMo"):
                load_lmeb(root, ["ConvoMem", "LoCoMo"])
            tasks = load_lmeb(root, ["Dialogue/ConvoMem/easy"])
            self.assertEqual([task.name for task in tasks], ["Dialogue/ConvoMem/easy"])

    def test_lmeb_requested_leaf_without_qrels_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_lmeb_leaf(root, with_qrels=False)
            with self.assertRaisesRegex(FileNotFoundError, "qrels.tsv"):
                load_lmeb(root, ["ConvoMem"])

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
            self.assertEqual(task.corpus["d1"], "T body")
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

    def test_standard_beir_queries_and_tsv_qrels_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "fiqa"
            write_jsonl(base / "corpus.jsonl", [{"_id": "d1", "text": "body"}])
            write_jsonl(base / "queries.jsonl", [{"_id": "q1", "text": "query"}])
            (base / "qrels").mkdir()
            (base / "qrels" / "test.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t2\n", encoding="utf-8"
            )
            task = load_beir(tmp, "fiqa")[0]
            self.assertEqual(task.qrels, {"q1": {"d1": 2.0}})

    def test_cqadupstack_incomplete_children_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "cqadupstack" / "android"
            write_jsonl(child / "corpus.jsonl", [{"_id": "d", "text": "doc"}])
            write_jsonl(child / "test_queries.jsonl", [{"_id": "q", "text": "query"}])
            write_jsonl(child / "test_qrels.jsonl", [{"query-id": "q", "corpus-id": "d", "score": 1}])
            with self.assertRaisesRegex(ValueError, "expected 12 child"):
                load_beir(Path(tmp), "cqadupstack")

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

    def test_lmeb_resolves_standard_scene_query_and_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "Dialogue" / "ConvoMem"
            leaf = task_root / "easy"
            write_jsonl(
                task_root / "corpus.jsonl",
                [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}],
            )
            write_jsonl(
                task_root / "candidates.jsonl",
                [{"scene_id": "scene_0", "candidate_doc_ids": ["d1"]}],
            )
            write_jsonl(leaf / "queries.jsonl", [{"id": "scene_0_q_0", "text": "query"}])
            (leaf / "qrels.tsv").write_text(
                "scene_0_q_0\td1\t1\nscene_0_q_0\td2\t1\n", encoding="utf-8"
            )
            (root / "task_instructions.json").write_text(
                json.dumps({"ConvoMem": {"easy": "Remember the conversation."}}),
                encoding="utf-8",
            )
            task = load_lmeb(root, ["Dialogue"])[0]
            self.assertEqual(task.candidates, {"scene_0_q_0": ["d1"]})
            self.assertEqual(task.instructions, {"instruction": "Remember the conversation."})
            self.assertEqual(task.qrels, {"scene_0_q_0": {"d1": 1.0}})

    def test_lmeb_rejects_unknown_candidate_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "Dialogue" / "ConvoMem"
            leaf = task_root / "case"
            write_jsonl(task_root / "corpus.jsonl", [{"id": "d1", "text": "one"}])
            write_jsonl(
                task_root / "candidates.jsonl",
                [{"scene_id": "s", "candidate_doc_ids": ["missing"]}],
            )
            write_jsonl(leaf / "queries.jsonl", [{"id": "q", "scene_id": "s", "text": "query"}])
            (leaf / "qrels.tsv").write_text("q\td1\t1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate pool references unknown"):
                load_lmeb(root, ["ConvoMem"])

    def test_run_rejects_nonfinite_scores_and_tampered_candidate_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "fiqa"
            write_jsonl(base / "corpus.jsonl", [{"_id": "d", "text": "doc"}])
            write_jsonl(base / "test_queries.jsonl", [{"_id": "q", "text": "query"}])
            write_jsonl(base / "test_qrels.jsonl", [{"query-id": "q", "corpus-id": "d", "score": 1}])
            task = load_beir(root, "fiqa")[0]
            run_path = root / "run.json"
            with self.assertRaises(ValueError):
                write_run(
                    run_path,
                    task=task,
                    run={"q": {"d": math.inf}},
                    kind="candidates",
                    model={"id": "toy"},
                    config_digest="x",
                )
            write_run(
                run_path,
                task=task,
                run={"q": {"d": 0.7}},
                kind="candidates",
                model={"id": "toy"},
                config_digest="x",
            )
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            payload["candidate_manifest_digest"] = "tampered"
            run_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate pool manifest"):
                read_run(run_path, expected_task=task)

    def test_candidate_policy_is_checked_when_reusing_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "fiqa"
            write_jsonl(base / "corpus.jsonl", [{"_id": "d", "text": "doc"}])
            write_jsonl(base / "test_queries.jsonl", [{"_id": "q", "text": "query"}])
            write_jsonl(base / "test_qrels.jsonl", [{"query-id": "q", "corpus-id": "d", "score": 1}])
            task = load_beir(root, "fiqa")[0]
            run_path = root / "run.json"
            write_run(
                run_path,
                task=task,
                run={"q": {"d": 0.7}},
                kind="candidates",
                model={"id": "toy", "ignore_identical_ids": True},
                config_digest="x",
            )
            self.assertEqual(
                read_run(
                    run_path,
                    expected_task=task,
                    expected_ignore_identical_ids=True,
                ),
                {"q": {"d": 0.7}},
            )
            with self.assertRaisesRegex(ValueError, "policy mismatch"):
                read_run(
                    run_path,
                    expected_task=task,
                    expected_ignore_identical_ids=False,
                )


if __name__ == "__main__":
    unittest.main()
