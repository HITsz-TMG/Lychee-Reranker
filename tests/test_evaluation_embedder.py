import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2Model

from lychee_reranker.evaluation.embedder import DenseEmbedder


class DenseEmbedderAttentionTests(unittest.TestCase):
    def test_encode_matches_flag_global_length_sort_and_restores_order(self):
        lengths = {"short": 2, "long": 5, "mid": 3, "tiny": 1}
        token_ids = {text: index + 1 for index, text in enumerate(lengths)}

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            pad_orders = []

            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": [
                        [token_ids[text]] * lengths[text] for text in texts
                    ],
                    "attention_mask": [[1] * lengths[text] for text in texts],
                }

            def pad(self, records, **kwargs):
                self.pad_orders.append([record["input_ids"][0] for record in records])
                max_length = max(len(record["input_ids"]) for record in records)
                input_ids = [
                    record["input_ids"] + [0] * (max_length - len(record["input_ids"]))
                    for record in records
                ]
                attention_mask = [
                    record["attention_mask"]
                    + [0] * (max_length - len(record["attention_mask"]))
                    for record in records
                ]
                return FakeBatch(
                    {
                        "input_ids": torch.tensor(input_ids),
                        "attention_mask": torch.tensor(attention_mask),
                    }
                )

        class FakeModel:
            config = SimpleNamespace(hidden_size=2)

            def __call__(self, input_ids, attention_mask, **kwargs):
                first = input_ids[:, :1].float()
                hidden = torch.cat(
                    [first.expand(-1, input_ids.shape[1]).unsqueeze(-1),
                     torch.ones_like(input_ids, dtype=torch.float32).unsqueeze(-1)],
                    dim=-1,
                )
                return SimpleNamespace(last_hidden_state=hidden)

        embedder = object.__new__(DenseEmbedder)
        embedder.tokenizer = FakeTokenizer()
        embedder.model = FakeModel()
        embedder.device = torch.device("cpu")
        embedder.dtype = torch.float32
        embedder.batch_size = 2
        embedder.max_length = 16
        embedder._use_bidirectional_mask = False

        actual = embedder.encode(["short", "long", "mid", "tiny"])
        self.assertEqual(embedder.tokenizer.pad_orders, [[2, 3], [1, 4]])
        expected_ids = [token_ids[text] for text in ("short", "long", "mid", "tiny")]
        expected = torch.nn.functional.normalize(
            torch.tensor([[float(value), 1.0] for value in expected_ids]), p=2, dim=-1
        ).numpy()
        self.assertTrue((abs(actual - expected) < 1e-7).all())
        self.assertEqual(embedder.encode([]).shape, (0, 2))

    def test_pool_matches_flag_mean_pooling_for_low_precision_hidden_states(self):
        hidden_fp32 = torch.tensor(
            [
                [[1.25, -0.75], [0.5, 2.25], [9.0, 9.0]],
                [[-1.5, 0.25], [2.0, -3.0], [4.0, 5.0]],
            ],
            dtype=torch.float32,
        )
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        for dtype in (torch.float16, torch.bfloat16):
            low_precision_hidden = hidden_fp32.to(dtype)
            expected = torch.nn.functional.normalize(
                (
                    low_precision_hidden * attention_mask.unsqueeze(-1).float()
                ).sum(dim=1)
                / attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1),
                p=2,
                dim=-1,
            )
            actual = DenseEmbedder._pool(low_precision_hidden, attention_mask)
            self.assertTrue(torch.equal(actual, expected))

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
        self.assertGreater(float((output[:, 0] - changed_output[:, 0]).abs().max().detach()), 1e-5)

        padded = torch.tensor([[1, 2, 3, 0]])
        changed_padding = torch.tensor([[1, 2, 3, 7]])
        padding_mask = DenseEmbedder._bidirectional_attention_mask(torch.tensor([[1, 1, 1, 0]]))
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
