from .airllm_base import AirLLMBaseModel


class AirLLMGemma4(AirLLMBaseModel):
    """
    AirLLM adapter for Gemma 4 models.

    Gemma 4 architecture:
    - 60 layers, hidden_size=5376, head_dim=256
    - GQA: 32 Q heads, 16 KV heads, 4 global KV heads
    - Sliding window attention (1024) with full attention every 6th layer
    - RMSNorm (eps=1e-06)
    - RoPE with proportional type for full attention, default for sliding
    """

    def set_layer_names_dict(self):
        """
        Layer names dictionary for Gemma 4.
        Maps logical components to their names in the model.
        """
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }

    def run_norm(self, layer, seq):
        """
        Run RMSNorm for Gemma 4.
        Gemma 4 uses RMSNorm with eps=1e-06.
        """
        return layer(seq)

    def run_lm_head(self, layer, seq):
        """
        Run language model head for Gemma 4.
        """
        if self.config.tie_word_embeddings:
            # Gemma 4 ties embeddings, need to use embed_tokens weight
            embed_tokens = self.model.model.embed_tokens
            seq = seq @ embed_tokens.weight.T
        else:
            seq = layer(seq)
        return seq
