import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
except ImportError:
    safe_open = None

from lychee_reranker.checkpoint.soup import soup_models


class SoupTests(unittest.TestCase):
    def _model(self, root: Path, name: str, floating, integer):
        path = root / name
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps({"model_type": "toy", "is_encoder_decoder": True}), encoding="utf-8"
        )
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        save_file({"float": floating, "integer": integer}, path / "model.safetensors")
        return path

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_average_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0, 3.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([3.0, 5.0]), torch.tensor([1]))
            output = soup_models([first, second], [0.5, 0.5], root / "out", max_shard_size_gb=1)
            with safe_open(output / "model.safetensors", framework="pt") as handle:
                self.assertTrue(torch.equal(handle.get_tensor("float"), torch.tensor([2.0, 4.0])))
            manifest = json.loads((output / "soup_manifest.json").read_text())
            self.assertEqual(manifest["weights"], [0.5, 0.5])
            with self.assertRaises(FileExistsError):
                soup_models([first, second], [0.5, 0.5], output, max_shard_size_gb=1)
            self.assertTrue(output.is_dir())

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_rejects_nonfloating_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([2.0]), torch.tensor([2]))
            with self.assertRaisesRegex(ValueError, "non-floating"):
                soup_models([first, second], [0.5, 0.5], root / "out")

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_sharded_output_has_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0, 3.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([3.0, 5.0]), torch.tensor([1]))
            output = soup_models([first, second], [0.5, 0.5], root / "out", max_shard_size_gb=1e-9)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(set(index["weight_map"]), {"float", "integer"})
            self.assertEqual(len(set(index["weight_map"].values())), 2)

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_accepts_indexed_safetensors_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([3.0]), torch.tensor([1]))
            for path in (first, second):
                tensors = load_file(path / "model.safetensors")
                (path / "model.safetensors").unlink()
                save_file({"float": tensors["float"]}, path / "model-00001-of-00002.safetensors")
                save_file(
                    {"integer": tensors["integer"]}, path / "model-00002-of-00002.safetensors"
                )
                (path / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"total_size": 12},
                            "weight_map": {
                                "float": "model-00001-of-00002.safetensors",
                                "integer": "model-00002-of-00002.safetensors",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            output = soup_models([first, second], [0.5, 0.5], root / "out")
            self.assertTrue((output / "model.safetensors").is_file())

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_rejects_duplicate_models_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([2.0]), torch.tensor([1]))
            for models, weights in (
                ([first, first], [0.5, 0.5]),
                ([first, second], [math.nan, 0.5]),
                ([first, second], [math.inf, 0.5]),
                ([first, second], [-0.5, 1.5]),
                ([first, second], [0.0, 0.0]),
            ):
                with self.assertRaises(ValueError):
                    soup_models(models, weights, root / "out")

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_rejects_incompatible_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([2.0]), torch.tensor([1]))
            config = json.loads((second / "config.json").read_text(encoding="utf-8"))
            config["is_encoder_decoder"] = False
            (second / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "encoder_decoder"):
                soup_models([first, second], [0.5, 0.5], root / "config-out")

            config["is_encoder_decoder"] = True
            (second / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (second / "tokenizer.json").write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                soup_models([first, second], [0.5, 0.5], root / "tokenizer-out")

            (second / "tokenizer.json").unlink()
            (second / "tokenizer.py").write_text("# tokenizer implementation\n", encoding="utf-8")
            (second / "generation_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer payload"):
                soup_models([first, second], [0.5, 0.5], root / "missing-payload-out")

    @unittest.skipIf(safe_open is None, "safetensors is not installed")
    def test_rejects_nonfinite_tensor_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._model(root, "a", torch.tensor([1.0]), torch.tensor([1]))
            second = self._model(root, "b", torch.tensor([2.0]), torch.tensor([1]))
            tensors = {"float": torch.tensor([float("nan")]), "integer": torch.tensor([1])}
            save_file(tensors, first / "model.safetensors")
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                soup_models([first, second], [0.5, 0.5], output)
            self.assertFalse(output.exists())
            self.assertFalse((root / ".out.lock").exists())


if __name__ == "__main__":
    unittest.main()
