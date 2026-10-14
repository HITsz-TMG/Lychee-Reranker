import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from lychee_reranker.train.data import (
    LycheeTrainCollator,
    discover_data_files,
    load_hf_dataset,
    validate_dataset_records,
    validate_record,
)


VALID = {
    "query": "q",
    "pos": ["p"],
    "neg": ["n1", "n2"],
    "prompt": "instruction",
    "pos_scores": [2.0],
    "neg_scores": [-1.0, -2.0],
}


class DataContractTests(unittest.TestCase):
    def test_dataset_rows_are_validated_before_training(self):
        self.assertEqual(validate_dataset_records([VALID], knowledge_distillation=True), 1)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_dataset_records([{"query": "q"}], knowledge_distillation=False)

    def test_collator_exposes_actual_query_group_count(self):
        class PadTokenizer:
            pad_token_id = 0

            def pad(self, rows, **kwargs):
                width = max(len(row["input_ids"]) for row in rows)
                ids = [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows]
                masks = [
                    row["attention_mask"] + [0] * (width - len(row["attention_mask"]))
                    for row in rows
                ]
                return {
                    "input_ids": torch.tensor(ids),
                    "attention_mask": torch.tensor(masks),
                }

        pair = {
            "encoder_input_ids": [1, 2],
            "encoder_attention_mask": [1, 1],
            "decoder_input_ids": [3],
            "decoder_attention_mask": [1],
        }
        batch = LycheeTrainCollator(PadTokenizer())(
            [([pair, pair], [2.0, -1.0]), ([pair, pair], [1.0, -2.0])]
        )
        self.assertEqual(batch["num_query_groups"].item(), 2)
        self.assertEqual(tuple(batch["teacher_scores"].shape), (4,))

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
