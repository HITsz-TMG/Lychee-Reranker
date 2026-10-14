from __future__ import annotations

from pathlib import Path

import torch
from transformers import Trainer
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINING_ARGS_NAME,
    TRAINER_STATE_NAME,
)


class LycheeTrainer(Trainer):
    """Trainer that preserves PEFT adapter and optimizer resume semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.args, "deepspeed", None) is not None:
            raise ValueError(
                "DeepSpeed is unsupported by LycheeTrainer; use native torchrun/DDP "
                "for adapter checkpoint save/resume"
            )
        if getattr(self.args, "fsdp", None):
            raise ValueError(
                "FSDP is unsupported by LycheeTrainer; use native torchrun/DDP "
                "for adapter checkpoint save/resume"
            )

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if state_dict is not None:
            raise ValueError(
                "LycheeTrainer cannot serialize an externally gathered full state dict; "
                "FSDP/DeepSpeed save paths are unsupported"
            )
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "save_pretrained"):
            raise TypeError("Lychee training model must expose save_pretrained for adapter saves")
        model.save_pretrained(str(output), safe_serialization=True)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(output))
        torch.save(self.args, output / TRAINING_ARGS_NAME)
        # Checkpoint saves normally write this after _save in Trainer's
        # _save_checkpoint.  Keeping a state copy here also makes an explicit
        # trainer.save_model() directory self-describing.
        self.state.save_to_json(str(output / TRAINER_STATE_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """Load the adapter stored by :meth:`_save` before Trainer's full-model path.

        Transformers recognizes adapter checkpoints only when the top-level
        object is a ``PeftModel``.  LycheeTrainer wraps that object to provide
        the multi-ratio loss, so delegate the adapter load to the inner PEFT
        model and let the parent restore optimizer/scheduler/RNG state.
        """
        target = model if model is not None else self.model
        peft_model = getattr(target, "model", None)
        adapter_files = (
            Path(resume_from_checkpoint) / "adapter_model.safetensors",
            Path(resume_from_checkpoint) / "adapter_model.bin",
        )
        if peft_model is not None and hasattr(peft_model, "load_adapter") and any(
            path.is_file() for path in adapter_files
        ):
            active_adapters = getattr(peft_model, "active_adapters", ["default"])
            if isinstance(active_adapters, str):
                active_adapters = [active_adapters]
            adapter_name = active_adapters[0] if active_adapters else "default"
            peft_model.load_adapter(
                resume_from_checkpoint,
                adapter_name=adapter_name,
                is_trainable=True,
            )
            if hasattr(peft_model, "set_adapter"):
                peft_model.set_adapter(adapter_name)
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """Map CPU DDP optimizer tensors to generic CPU storage on resume."""
        device = getattr(self.args, "device", None)
        device_type = getattr(device, "type", str(device))
        if checkpoint is not None and self.args.world_size > 1 and device_type == "cpu":
            optimizer_path = Path(checkpoint) / OPTIMIZER_NAME
            scheduler_path = Path(checkpoint) / SCHEDULER_NAME
            if optimizer_path.is_file() and scheduler_path.is_file():
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location="cpu", weights_only=True)
                )
                self.lr_scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location="cpu", weights_only=True)
                )
                return
        super()._load_optimizer_and_scheduler(checkpoint)
