import unittest

from lychee_reranker.evaluation.metrics import evaluate_run, ndcg_at_k, ranked_document_ids
from lychee_reranker.prompts import DEFAULT_SYSTEM_INSTRUCTION, build_decoder_prompt, build_document


class PromptAndMetricTests(unittest.TestCase):
    def test_exact_prompt(self):
        value = build_decoder_prompt("q", "i", DEFAULT_SYSTEM_INSTRUCTION)
        self.assertEqual(
            value,
            "<bos><start_of_turn>user\n"
            "Judge whether the Document meets the requirements based on the Query and "
            'the Instruct provided. Note that the answer can only be "yes" or "no".\n\n'
            "<Instruct>: i\n<Query>: q<end_of_turn>\n<start_of_turn>model\n\n\n\n",
        )
        self.assertEqual(build_document("d"), "<Document>: d")

    def test_metrics_and_tie_breaking(self):
        qrels = {"q": {"d1": 1.0, "d2": 1.0, "d3": 0.0}}
        run = {"q": {"d2": 0.5, "d1": 0.5, "d3": 0.1}}
        self.assertEqual(ranked_document_ids(run["q"]), ["d1", "d2", "d3"])
        values = evaluate_run(qrels, run, ("ndcg@10", "recall@1", "mrr@10"))
        self.assertAlmostEqual(values["ndcg@10"], 1.0)
        self.assertAlmostEqual(values["recall@1"], 0.5)
        self.assertAlmostEqual(values["mrr@10"], 1.0)

    def test_missing_run_counts_as_zero(self):
        values = evaluate_run({"a": {"d": 1}, "b": {"x": 1}}, {"a": {"d": 2}}, ("recall@1",))
        self.assertEqual(values["recall@1"], 0.5)

    def test_graded_ndcg_uses_standard_gain(self):
        value = ndcg_at_k({"d1": 2.0, "d2": 1.0}, ["d2", "d1"], 2)
        expected = (1.0 + 2.0 / 1.584962500721156) / (2.0 + 1.0 / 1.584962500721156)
        self.assertAlmostEqual(value, expected)

    def test_graded_ndcg_matches_pytrec_eval_when_available(self):
        try:
            import pytrec_eval
        except ImportError:
            self.skipTest("pytrec_eval is optional")
        qrels = {"q": {"d1": 2, "d2": 1}}
        run = {"q": {"d2": 0.9, "d1": 0.8}}
        reference = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.2"}).evaluate(run)
        actual = evaluate_run(qrels, run, ("ndcg@2",))["ndcg@2"]
        self.assertAlmostEqual(actual, reference["q"]["ndcg_cut_2"])


if __name__ == "__main__":
    unittest.main()
