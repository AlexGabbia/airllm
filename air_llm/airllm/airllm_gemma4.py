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

    Layer pattern (from config.json layer_types):
    [sliding, sliding, sliding, sliding, sliding, full_attention] x 10
    Layers 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 are full attention
    All others are sliding window (1024 tokens)
    """

    def set_layer_names_dict(self):
        """
        Layer names dictionary for Gemma 4.
        Maps logical components to their names in the model.

        Gemma 4 is a multimodal model (Gemma4ForConditionalGeneration) where
        the language model weights are nested under model.language_model.*
        instead of directly under model.*.
        """
        self.layer_names_dict = {
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
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

        Gemma 4 multimodal model has language model nested under
        model.language_model, so embed_tokens is at
        model.language_model.embed_tokens (not model.model.embed_tokens).
        """
        if self.config.tie_word_embeddings:
            # Gemma4ForConditionalGeneration nests the text model
            # under model.language_model
            if hasattr(self.model, 'language_model'):
                embed_tokens = self.model.language_model.embed_tokens
            else:
                embed_tokens = self.model.model.embed_tokens
            seq = seq @ embed_tokens.weight.T
        else:
            seq = layer(seq)
        return seq

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """
        Handle sliding window attention mask for Gemma 4.
        Sliding window layers use a causal mask limited to 1024 tokens.
        Full attention layers (every 6th) use the full causal mask.
        """
        return {
            "attention_mask": full_attention_mask[
                :, :, -len_p - len_s :, -len_p - len_s :
            ]
        }

    def get_pos_emb_args(self, len_p, len_s):
        """
        Position embedding arguments for Gemma 4.
        Gemma 4 uses RoPE with different configs for sliding vs full attention.
        """
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        """
        Past key value arguments for Gemma 4.
        """
        return {"past_key_value": (k_cache, v_cache)}

    def _is_full_attention_layer(self, layer_idx):
        """
        Check if a layer uses full attention (every 6th layer).
        Gemma 4 pattern: 5 sliding + 1 full, repeated.
        Full attention layers: 5, 11, 17, 23, 29, 35, 41, 47, 53, 59
        """
        return (layer_idx + 1) % 6 == 0

    def _is_boundary_layer(self, layer_idx):
        """
        Check if layer is a boundary layer (should not be compressed).
        Full attention layers (every 6th) are always protected because they
        aggregate global context and are more sensitive to compression artifacts.
        """
        is_boundary = (
            self.boundary_layers > 0
            and (
                layer_idx < self.boundary_layers
                or layer_idx >= self.n_layers - self.boundary_layers
            )
        )
        # Full attention layers are always protected regardless of boundary_layers
        return is_boundary or self._is_full_attention_layer(layer_idx)

    def get_kv_compressor(self, layer_idx):
        """
        Return the appropriate KV compressor for this layer.
        Full attention layers use global_head_dim, sliding layers use head_dim.
        """
        if self._is_full_attention_layer(layer_idx) and self.kv_compressor_global is not None:
            return self.kv_compressor_global
        return self.kv_compressor
