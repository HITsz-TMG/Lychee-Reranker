import copy
import random
import tempfile
import unittest
from pathlib import Path

import torch

from lychee_reranker.train.cli import ensure_safe_output_dir, parse_args
from lychee_reranker.train.config import (
    apply_path_overrides,
    load_config,
    seed_everything,
    validate_training_config,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def test_release_recipes_match_paper_optimizer_contract(self):
        recipes = sorted((ROOT / "configs" / "train").glob("*.yaml"))
        self.assertEqual(len(recipes), 9)
        for path in recipes:
            config = load_config(path)
            validate_training_config(config)
            self.assertEqual(config["model"]["chunk_sizes"], [2, 8, 32, 128])
            self.assertEqual(config["training"]["lr_scheduler_type"], "cosine")
            self.assertEqual(config["training"]["optim"], "adamw_torch")
            self.assertNotIn("deepspeed", config["training"])

    def test_unsupported_distributed_save_path_fails_early(self):
        config = load_config(ROOT / "configs" / "train" / "nano_sft.yaml")
        config = copy.deepcopy(config)
        config["training"]["deepspeed"] = "configs/deepspeed/stage3.json"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_training_config(config)

        config["training"].pop("deepspeed")
        config["training"]["learning_rate"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_training_config(config)

    def test_model_override_is_effective_without_mutating_source(self):
        source = load_config(ROOT / "configs" / "train" / "nano_sft.yaml")
        effective = apply_path_overrides(
            source,
            data=["public/train.jsonl"],
            output_dir="outputs/test",
            model_name_or_path="google/t5gemma-2-270m-270m",
        )
        self.assertEqual(effective["model"]["name_or_path"], "google/t5gemma-2-270m-270m")
        self.assertEqual(effective["data"]["train_data"], ["public/train.jsonl"])
        self.assertNotEqual(source["data"]["train_data"], effective["data"]["train_data"])
        args = parse_args(
            ["--config", "recipe.yaml", "--model-name-or-path", "local/model", "--print-config"]
        )
        self.assertEqual(args.model_name_or_path, "local/model")

    def test_output_guard_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            data.write_text("{}\n", encoding="utf-8")
            output = root / "output"
            self.assertEqual(ensure_safe_output_dir(output, protected_paths=[data]), output)
            (output / "sentinel").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_safe_output_dir(output, protected_paths=[data])
            with self.assertRaises(ValueError):
                ensure_safe_output_dir(root / "nested", protected_paths=[root])

        seed_everything(17)
        first = (random.random(), torch.rand(1).item())
        seed_everything(17)
        second = (random.random(), torch.rand(1).item())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
