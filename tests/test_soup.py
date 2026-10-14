import json
import tempfile
import unittest
from pathlib import Path

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    safe_open = None

from lychee_reranker.checkpoint.soup import soup_models


class SoupTests(unittest.TestCase):
    def _model(self, root: Path, name: str, floating, integer):
        path = root / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "toy"}), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
