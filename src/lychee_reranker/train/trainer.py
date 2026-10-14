from __future__ import annotations

from pathlib import Path

from transformers import Trainer


class LycheeTrainer(Trainer):
    """Trainer that saves the PEFT model and tokenizer through the wrapper."""

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        # The Trainer state dict uses wrapper-prefixed keys; PEFT owns the
        # authoritative adapter serialization and must collect it directly.
        model.save_pretrained(output_dir, safe_serialization=True)
        self.state.save_to_json(str(Path(output_dir) / "trainer_state.json"))
