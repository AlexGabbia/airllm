import torch

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

    Multimodal support:
    - Vision tower: 27 layers, hidden_size=1152, head_dim=72
    - Patch embedding + spatial pooling (2520 patches → 280 soft tokens)
    - Multi-modal embedder: projects vision features from 1152 → 5376
    - Bidirectional attention for vision tokens within <boi>...<eoi> blocks

    Layer pattern (from config.json layer_types):
    [sliding, sliding, sliding, sliding, sliding, full_attention] x 10
    Layers 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 are full attention
    All others are sliding window (1024 tokens)
    """

    def set_layer_names_dict(self):
        """
        Layer names dictionary for Gemma 4.

        Gemma 4 is a multimodal model (Gemma4ForConditionalGeneration) where
        the language model weights are nested under model.language_model.*
        and the vision tower is under model.vision_tower.*
        """
        self.layer_names_dict = {
            # Language model components
            "embed": "model.language_model.embed_tokens",
            "layer_prefix": "model.language_model.layers",
            "norm": "model.language_model.norm",
            "lm_head": "lm_head",
            # Vision components (for multimodal inference)
            "vision_patch_embedder": "model.vision_tower.patch_embedder",
            "vision_layer_prefix": "model.vision_tower.encoder.layers",
            "vision_std": "model.vision_tower",
            "embed_vision": "model.embed_vision",
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

    def get_image_token_id(self):
        """Return the image token ID for Gemma 4."""
        return getattr(self.config, 'image_token_id', 258880)

    def get_boi_token_id(self):
        """Return the beginning-of-image token ID."""
        return getattr(self.config, 'boi_token_id', 255999)

    def get_eoi_token_id(self):
        """Return the end-of-image token ID."""
        return getattr(self.config, 'eoi_token_id', 258882)

    def forward_vision(self, pixel_values, pixel_position_ids=None,
                       padding_mask=None, num_soft_tokens=None):
        """
        Process images through the vision tower layer-by-layer.

        Runs the vision encoder (27 layers) one layer at a time on GPU,
        then applies spatial pooling and projection to text hidden dimension.

        Args:
            pixel_values: Patchified image tensor (batch, max_patches, patch_dim)
            pixel_position_ids: 2D position IDs for patches (batch, max_patches, 2)
            padding_mask: Boolean mask for valid patches (True = padding position)
            num_soft_tokens: Number of soft tokens per image (default: 280)

        Returns:
            image_features: Projected image features (num_valid_tokens, text_hidden_size)
        """
        device = self.running_device
        vision_config = self._get_vision_config()
        if vision_config is None:
            return None

        if num_soft_tokens is None:
            num_soft_tokens = getattr(self.config, 'vision_soft_tokens_per_image', 280)

        # Step 1: Patch embedding
        # Load and run patch_embedder
        patch_embedder = self._get_vision_module('vision_patch_embedder')
        if patch_embedder is None:
            return None

        hidden_states = patch_embedder(
            pixel_values.to(device=device, dtype=self.running_dtype),
            pixel_position_ids=pixel_position_ids.to(device=device) if pixel_position_ids is not None else None,
            padding_positions=padding_mask.to(device=device) if padding_mask is not None else None,
        )

        # Step 2: Run vision encoder layers one by one
        attention_mask = None
        if padding_mask is not None:
            # Bidirectional attention for all vision tokens
            # padding_mask: True = valid, we invert for attention mask
            attention_mask = padding_mask.to(device=device, dtype=hidden_states.dtype)
            # Expand for multi-head attention: (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert: valid positions = 0, padding = large negative
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        n_vision_layers = self.n_vision_layers
        for i in range(n_vision_layers):
            layer_name = f"{self.layer_names_dict['vision_layer_prefix']}.{i}"
            vision_layer = self._get_vision_layer(i)

            # Load layer weights to GPU
            state_dict = self.load_layer_to_cpu(layer_name)
            self.move_layer_to_device(state_dict)

            # Run vision encoder layer
            layer_outputs = vision_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            # Offload layer
            vision_layer.to("meta")
            torch.cuda.empty_cache()

        # Step 3: Spatial pooling (reduce patches to soft tokens)
        pooler = self._get_vision_module('pooler')
        if pooler is not None:
            # Pooler expects hidden_states and position info
            hidden_states = pooler(
                hidden_states,
                pixel_position_ids=pixel_position_ids.to(device=device) if pixel_position_ids is not None else None,
                padding_positions=padding_mask.to(device=device) if padding_mask is not None else None,
                output_length=num_soft_tokens,
            )

        # Step 4: Standardization (std_bias and std_scale)
        vision_std = self._get_vision_std()
        if vision_std is not None:
            std_scale = getattr(vision_std, 'std_scale', None)
            std_bias = getattr(vision_std, 'std_bias', None)
            if std_scale is not None:
                hidden_states = hidden_states * std_scale.to(device=device, dtype=hidden_states.dtype)
            if std_bias is not None:
                hidden_states = hidden_states + std_bias.to(device=device, dtype=hidden_states.dtype)

        # Step 5: Project to text hidden dimension
        embed_vision = self._get_vision_module('embed_vision')
        if embed_vision is not None:
            hidden_states = embed_vision(hidden_states.to(device=device))

        # Remove padding tokens if any
        if padding_mask is not None:
            # Keep only valid (non-padding) tokens
            valid_mask = ~padding_mask.to(device=device)
            if valid_mask.dim() > 1:
                valid_mask = valid_mask.squeeze(0)
            hidden_states = hidden_states[valid_mask]

        return hidden_states

    def _get_vision_config(self):
        """Get the vision configuration from the model config."""
        vision_config = getattr(self.config, 'vision_config', None)
        if vision_config is None:
            # Try nested in model config
            text_config = getattr(self.config, 'text_config', None)
            if text_config is not None:
                vision_config = getattr(text_config, 'vision_config', None)
        return vision_config

    def _get_vision_module(self, module_name):
        """Get a vision module from the model by navigating its structure."""
        model = self.model
        # Navigate model.vision_tower.* or model.embed_vision
        parts = self.layer_names_dict.get(module_name, '').split('.')
        # Remove 'model.' prefix since we start from self.model
        if parts[0] == 'model':
            parts = parts[1:]
        module = model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def _get_vision_layer(self, idx):
        """Get a vision encoder layer by index."""
        parts = self.layer_names_dict['vision_layer_prefix'].split('.')
        if parts[0] == 'model':
            parts = parts[1:]
        module = self.model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                raise ValueError(f"Could not navigate to vision encoder layers: {'.'.join(parts)}")
        return module[idx]

    def _get_vision_std(self):
        """Get the vision standardization module (std_bias, std_scale)."""
        return self._get_vision_module('vision_std')

    def merge_image_embeddings(self, inputs_embeds, image_features, input_ids):
        """
        Merge image features into text embeddings by replacing image token positions.

        Args:
            inputs_embeds: Text embeddings tensor (batch, seq_len, hidden_size)
            image_features: Projected image features (num_image_tokens, hidden_size)
            input_ids: Input token IDs (batch, seq_len)

        Returns:
            inputs_embeds with image tokens replaced by image features
        """
        image_token_id = self.get_image_token_id()
        # Find positions of image tokens
        image_mask = (input_ids == image_token_id)
        if image_mask.any():
            # Scatter image features into embeddings
            inputs_embeds = inputs_embeds.clone()
            # Flatten for scatter operation
            image_features_flat = image_features.to(inputs_embeds.dtype).to(inputs_embeds.device)
            inputs_embeds[image_mask] = image_features_flat[:image_mask.sum()]
        return inputs_embeds