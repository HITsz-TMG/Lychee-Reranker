"""Real CPU distributed training: no downloaded model, data, or GPU required."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    SiglipVisionConfig,
    T5Gemma2Config,
    T5Gemma2DecoderConfig,
    T5Gemma2EncoderConfig,
    T5Gemma2ForConditionalGeneration,
    T5Gemma2TextConfig,
)


def _tiny_base(path):
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
    T5Gemma2ForConditionalGeneration(config).save_pretrained(path)
    tokenizer = Tokenizer(
        WordLevel(
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
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="<bos>",
        unk_token="[UNK]",
    ).save_pretrained(path)


def test_tiny_t5gemma2_two_rank_training_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base"
    _tiny_base(base)
    data = tmp_path / "data.jsonl"
    record = {
        "query": "question",
        "prompt": "Retrieve an answer.",
        "pos": ["answer"],
        "neg": ["unrelated", "unrelated answer"],
    }
    data.write_text("".join(json.dumps(record) + "\n" for _ in range(8)))
    config = yaml.safe_load((root / "configs/train/nano_sft.yaml").read_text())
    config["model"].update(name_or_path=str(base), dtype="float32")
    config["model"]["lora"].update(rank=2, alpha=1)
    config["data"].update(train_data=[str(data)], train_group_size=2)
    output = tmp_path / "output"
    config["training"].update(
        output_dir=str(output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=2,
        save_steps=1,
        bf16=False,
        use_cpu=True,
        disable_tqdm=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    config["training"].pop("deepspeed", None)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(config))
    env = os.environ.copy()
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(tmp_path / "hf"),
        HF_DATASETS_CACHE=str(tmp_path / "cache"),
        PYTHONPATH=str(root / "src"),
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "--module",
        "lychee_reranker.train.cli",
        "--config",
        str(recipe),
    ]
    original_digest = None
    for resume in (False, True):
        args = command + (
            ["--resume-from-checkpoint", str(output / "checkpoint-1")] if resume else []
        )
        result = subprocess.run(
            args,
            env=env,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout[-18000:]
        state = json.loads((output / "trainer_state.json").read_text())
        assert state["global_step"] == 2
        adapter = output / "adapter_model.safetensors"
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        if resume:
            assert digest == original_digest
        else:
            original_digest = digest
    adapter_config = json.loads((output / "adapter_config.json").read_text())
    assert "vision_tower" in str(adapter_config.get("exclude_modules", "")) or all(
        "vision_tower" not in name and "." in name for name in adapter_config["target_modules"]
    )
