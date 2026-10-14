import json
import math
import tempfile
import unittest
from pathlib import Path

from lychee_reranker.train.data import discover_data_files, load_hf_dataset, validate_record


VALID = {
    "query": "q",
    "pos": ["p"],
    "neg": ["n1", "n2"],
    "prompt": "instruction",
    "pos_scores": [2.0],
    "neg_scores": [-1.0, -2.0],
}


class DataContractTests(unittest.TestCase):
    def test_validation_and_multiplicity(self):
        self.assertEqual(validate_record(VALID, knowledge_distillation=True)["neg"], ["n1", "n2"])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_record({**VALID, "neg": []}, knowledge_distillation=False)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_record({**VALID, "neg_scores": [1.0]}, knowledge_distillation=True)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_record({**VALID, "pos_scores": [math.nan]}, knowledge_distillation=True)

    def test_jsonl_parquet_equivalence(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "a.jsonl"
            jsonl.write_text(json.dumps(VALID) + "\n", encoding="utf-8")
            parquet = root / "b.parquet"
            pq.write_table(pa.Table.from_pylist([VALID]), parquet)
            files = discover_data_files([root])
            self.assertEqual([path.name for path in files], ["a.jsonl", "b.parquet"])
            json_ds = load_hf_dataset([jsonl])
            parquet_ds = load_hf_dataset([parquet])
            self.assertEqual(
                validate_record(json_ds[0], knowledge_distillation=True),
                validate_record(parquet_ds[0], knowledge_distillation=True),
            )

    def test_repeated_input_roots_preserve_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text(json.dumps(VALID) + "\n", encoding="utf-8")
            self.assertEqual(discover_data_files([path, path]), [path, path])
            self.assertEqual(len(load_hf_dataset([path, path])), 2)


if __name__ == "__main__":
    unittest.main()
