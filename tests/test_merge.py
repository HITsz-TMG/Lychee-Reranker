import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        AutoModelForSeq2SeqLM,
        PreTrainedTokenizerFast,
        SiglipVisionConfig,
        T5Config,
        T5ForConditionalGeneration,
        T5Gemma2Config,
        T5Gemma2DecoderConfig,
        T5Gemma2EncoderConfig,
        T5Gemma2ForConditionalGeneration,
        T5Gemma2TextConfig,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from lychee_reranker.checkpoint.merge import merge_lora


@unittest.skipUnless(PEFT_AVAILABLE, "transformers, peft, safetensors, and tokenizers are required")
class MergeTests(unittest.TestCase):
    def _base_and_adapter(self, root: Path) -> tuple[Path, Path]:
        base = root / "base"
        base.mkdir()
        config = T5Config(
            vocab_size=32,
            d_model=16,
            d_kv=8,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            dropout_rate=0.0,
            tie_word_embeddings=False,
        )
        torch.manual_seed(7)
        T5ForConditionalGeneration(config).save_pretrained(base, safe_serialization=True)

        vocabulary = {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="<pad>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        fast.save_pretrained(base)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.float32)
        peft_model = get_peft_model(
            base_model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q", "v"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            ),
        )
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.05)
        adapter = root / "adapter"
        peft_model.save_pretrained(adapter)
        return base, adapter

    def _t5gemma2_base_and_adapter(self, root: Path) -> tuple[Path, Path]:
        base = root / "t5gemma2-base"
        base.mkdir()
        torch.manual_seed(123)
        text = dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
            max_position_embeddings=256,
            query_pre_attn_scalar=8,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )
        config = T5Gemma2Config(
            encoder=T5Gemma2EncoderConfig(
                text_config=T5Gemma2TextConfig(**text),
                vision_config=SiglipVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=28,
                    patch_size=14,
                ),
                image_token_index=31,
                mm_tokens_per_image=4,
            ),
            decoder=T5Gemma2DecoderConfig(**text),
            vocab_size=32,
            image_token_index=31,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=2,
        )
        T5Gemma2ForConditionalGeneration(config).save_pretrained(base)

        tokenizer = Tokenizer(
            models.WordLevel(
                {
                    "[PAD]": 0,
                    "[EOS]": 1,
                    "<bos>": 2,
                    "[UNK]": 3,
                    "yes": 4,
                    "no": 5,
                    "question": 6,
                    "answer": 7,
                    "unrelated": 8,
                },
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            eos_token="[EOS]",
            bos_token="<bos>",
            unk_token="[UNK]",
        ).save_pretrained(base)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.float32)
        peft_model = get_peft_model(
            base_model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.SEQ_2_SEQ_LM,
            ),
        )
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.05)
        adapter = root / "t5gemma2-adapter"
        peft_model.save_pretrained(adapter)
        return base, adapter

    def test_tiny_t5_merge_reload_matches_peft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = self._base_and_adapter(root)
            base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
            base_config["dtype"] = "bfloat16"
            (base / "config.json").write_text(json.dumps(base_config), encoding="utf-8")
            output = merge_lora(str(base), str(adapter), root / "merged", dtype="float32")

            expected = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.float32)
            expected = PeftModel.from_pretrained(expected, adapter, is_trainable=False)
            expected = expected.merge_and_unload(safe_merge=True)
            actual = AutoModelForSeq2SeqLM.from_pretrained(output, dtype=torch.float32)
            self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
            for key, tensor in expected.state_dict().items():
                self.assertTrue(torch.equal(tensor, actual.state_dict()[key]), key)

            self.assertTrue((output / "tokenizer.json").is_file())
            self.assertTrue((output / "generation_config.json").is_file())
            saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["dtype"], "float32")
            manifest = json.loads((output / "merge_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["operation"], "merge_lora")
            self.assertEqual(manifest["model_type"], "t5")

    def test_tiny_t5gemma2_bfloat16_merge_reload_matches_peft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = self._t5gemma2_base_and_adapter(root)
            output = merge_lora(str(base), str(adapter), root / "merged", dtype="bfloat16")

            expected = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.bfloat16)
            expected = PeftModel.from_pretrained(expected, adapter, is_trainable=False)
            expected = expected.merge_and_unload(safe_merge=True)
            actual = AutoModelForSeq2SeqLM.from_pretrained(output, dtype=torch.bfloat16)
            self.assertEqual(set(expected.state_dict()), set(actual.state_dict()))
            for key, tensor in expected.state_dict().items():
                self.assertEqual(tensor.dtype, actual.state_dict()[key].dtype, key)
                self.assertTrue(torch.equal(tensor, actual.state_dict()[key]), key)

            saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["dtype"], "bfloat16")
            self.assertEqual(saved_config["decoder"]["dtype"], "bfloat16")
            self.assertEqual(saved_config["encoder"]["dtype"], "bfloat16")

    def test_rejects_adapter_for_different_local_base_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = self._base_and_adapter(root)
            wrong_base = root / "wrong-base"
            shutil.copytree(base, wrong_base)
            with self.assertRaisesRegex(ValueError, "different base model"):
                merge_lora(str(wrong_base), str(adapter), root / "unused", dtype="float32")

            existing = root / "existing"
            existing.mkdir()
            marker = existing / "marker"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                merge_lora(str(base), str(adapter), existing, dtype="float32")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_rejects_nonfinite_adapter_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = self._base_and_adapter(root)
            adapter_file = adapter / "adapter_model.safetensors"
            tensors = load_file(adapter_file)
            first = next(iter(tensors))
            tensors[first].view(-1)[0] = float("nan")
            save_file(tensors, adapter_file)
            output = root / "merged"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                merge_lora(str(base), str(adapter), output, dtype="float32")
            self.assertFalse(output.exists())
            self.assertFalse((root / ".merged.lock").exists())


if __name__ == "__main__":
    unittest.main()
