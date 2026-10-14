import unittest

import torch
from transformers import Qwen2Config, Qwen2Model

from lychee_reranker.evaluation.embedder import DenseEmbedder


class DenseEmbedderAttentionTests(unittest.TestCase):
    def test_native_qwen2_noncausal_mask_uses_future_tokens_and_blocks_padding(self):
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            is_causal=False,
            _attn_implementation="eager",
        )
        model = Qwen2Model(config).eval()

        # Transformers 5.3 still constructs native Qwen2 attention as causal;
        # the prepared mask is the contract that bypasses that implementation
        # detail for this checkpoint.
        self.assertFalse(config.is_causal)
        self.assertTrue(model.layers[0].self_attn.is_causal)

        first = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 9, 4]])
        full_mask = DenseEmbedder._bidirectional_attention_mask(torch.ones(1, 4, dtype=torch.long))
        output = model(
            input_ids=first,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        changed_output = model(
            input_ids=changed_future,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        self.assertGreater(
            float((output[:, 0] - changed_output[:, 0]).abs().max().detach()), 1e-5
        )

        padded = torch.tensor([[1, 2, 3, 0]])
        changed_padding = torch.tensor([[1, 2, 3, 7]])
        padding_mask = DenseEmbedder._bidirectional_attention_mask(
            torch.tensor([[1, 1, 1, 0]])
        )
        padded_output = model(
            input_ids=padded,
            attention_mask=padding_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        changed_padding_output = model(
            input_ids=changed_padding,
            attention_mask=padding_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        self.assertTrue(
            torch.allclose(padded_output[:, :3], changed_padding_output[:, :3], atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
