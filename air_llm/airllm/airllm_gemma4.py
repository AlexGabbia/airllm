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

    KV Cache: Gemma4 uses transformers Cache objects (DynamicCache with
    sliding window layers) instead of returning (k, v) tuples from the
    decoder layer. The decoder layer only returns hidden_states and
    updates the Cache object in-place.
    """

    # Gemma4 decoder layers use Cache objects, not tuple returns
    uses_cache_object = True

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

        Gemma4ForConditionalGeneration has structure:
          model.language_model.embed_tokens
        Accessing embed_tokens requires navigating through the nested structure.
        """
        if self.config.tie_word_embeddings:
            # Navigate the nested model structure to find embed_tokens
            # Try: model.language_model.embed_tokens (Gemma4ForConditionalGeneration)
            # Then: model.model.language_model.embed_tokens (if model is wrapped)
            # Then: model.model.embed_tokens (standard Llama-like structure)
            embed_tokens = None
            if hasattr(self.model, 'language_model'):
                embed_tokens = self.model.language_model.embed_tokens
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
                embed_tokens = self.model.model.language_model.embed_tokens
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_tokens = self.model.model.embed_tokens
            if embed_tokens is None:
                raise AttributeError("Cannot find embed_tokens for tied lm_head")
            seq = seq @ embed_tokens.weight.T
        else:
            seq = layer(seq)
        return seq

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        """
        Handle sliding window attention mask for Gemma 4.

        During prefill (len_p=0): Q and K both have len_s tokens,
        so mask shape is [batch, heads, len_s, len_s].

        During generation (len_p>0): Q has len_s tokens (typically 1),
        K has len_p+len_s tokens. Mask shape should be
        [batch, heads, len_s, len_p+len_s].
        """
        if len_p > 0:
            # Generation: query length = len_s, key length = len_p + len_s
            return {
                "attention_mask": full_attention_mask[
                    :, :, -len_s:, -len_p - len_s:
                ]
            }
        # Prefill: full causal mask for all tokens
        return {
            "attention_mask": full_attention_mask[
                :, :, -len_p - len_s :, -len_p - len_s :
            ]
        }

    def get_pos_emb_args(self, len_p, len_s):
        """
        Position embedding arguments for Gemma 4.
        Returns position_embeddings (cos, sin) tuple for the current decoder layer.

        During prefill (len_p=0), uses precomputed position_embeddings from
        compute_position_embeddings(). During generation (len_p>0, len_s=1),
        recomputes position_embeddings for the single new token position.
        """
        if len_p > 0:
            # Generation mode: recompute position_embeddings for the new token
            decoder_idx = getattr(self, '_current_decoder_idx', -1)
            if decoder_idx >= 0:
                return self._compute_pos_emb_for_generation(len_p, len_s, decoder_idx)
            return {}

        # Prefill mode: use precomputed position_embeddings
        if hasattr(self, '_position_embeddings_data') and self._position_embeddings_data is not None:
            decoder_idx = getattr(self, '_current_decoder_idx', -1)
            if decoder_idx >= 0:
                return self.get_position_embeddings_for_layer(
                    self._position_embeddings_data, decoder_idx
                )
        return {}

    def _compute_pos_emb_for_generation(self, len_p, len_s, layer_idx):
        """
        Compute position_embeddings for a single generation step.
        len_p = number of past tokens, len_s = 1 (new token).
        Position ID for the new token is len_p.
        """
        text_model = self.model.model.language_model
        rotary_emb = getattr(text_model, 'rotary_emb', None)
        if rotary_emb is None:
            return {}

        text_config = getattr(self.config, 'text_config', None) or self.config
        layer_types = getattr(text_config, 'layer_types', None)
        if layer_types is None or layer_idx >= len(layer_types):
            return {}

        layer_type = layer_types[layer_idx]

        # Create position_ids for the new token only
        device = self.running_device
        position_ids = torch.arange(len_p, len_p + len_s, dtype=torch.long, device=device)[None, :]

        # We need hidden_states just for dtype/device; use a dummy
        # rotary_emb only uses position_ids shape for the output shape
        text_config = getattr(self.config, 'text_config', None) or self.config
        hidden_size = getattr(text_config, 'hidden_size', 5376)
        dummy_hidden = torch.zeros(1, len_s, hidden_size, device=device, dtype=self.running_dtype)
        cos, sin = rotary_emb(dummy_hidden, position_ids, layer_type=layer_type)

        return {"position_embeddings": (cos, sin)}

    def compute_position_embeddings(self, hidden_states, position_ids):
        """
        Precompute position embeddings (cos, sin) for each layer type.

        Gemma4 has different RoPE parameters for sliding_attention and
        full_attention layers. The model's rotary_emb module computes
        different cos/sin for each type.

        Returns a dict: {layer_type: (cos, sin)}
        """
        text_model = self.model.model.language_model
        rotary_emb = getattr(text_model, 'rotary_emb', None)
        if rotary_emb is None:
            return {}

        # Get layer types from config
        text_config = getattr(self.config, 'text_config', None) or self.config
        layer_types = getattr(text_config, 'layer_types', None)
        if layer_types is None:
            return {}

        # Compute position embeddings for each unique layer type
        unique_types = set(layer_types)
        position_embeddings = {}
        for layer_type in unique_types:
            cos, sin = rotary_emb(hidden_states, position_ids, layer_type=layer_type)
            position_embeddings[layer_type] = (cos, sin)

        return position_embeddings, layer_types

    def get_position_embeddings_for_layer(self, position_embeddings_data, layer_idx):
        """
        Get the position_embeddings (cos, sin) for a specific decoder layer.
        """
        if position_embeddings_data is None:
            return {}
        pos_emb_dict, layer_types = position_embeddings_data
        if layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
            if layer_type in pos_emb_dict:
                return {"position_embeddings": pos_emb_dict[layer_type]}
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        """
        Past key value arguments for Gemma 4.
        Creates a DynamicCache pre-populated with the given K/V cache.
        """
        return {"past_key_values": self.create_layer_cache(0, k_cache, v_cache)}

    def create_layer_cache(self, layer_idx, k_cache=None, v_cache=None):
        """
        Create a DynamicCache for a single decoder layer call.

        Gemma4 uses DynamicCache with sliding window support.
        We create a minimal cache with only the layer we need,
        pre-populate it with existing K/V (if any), and return it.

        Args:
            layer_idx: The decoder layer index (0-59) that will be called
            k_cache: Existing key cache tensor (batch, kv_heads, seq, head_dim) or None
            v_cache: Existing value cache tensor (batch, kv_heads, seq, head_dim) or None

        Returns:
            DynamicCache ready to be passed to the decoder layer
        """
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicSlidingWindowLayer, DynamicLayer

        # Create cache WITHOUT config to avoid allocating all 60 layers
        cache = DynamicCache()

        # Get layer types to determine if this layer uses sliding window
        text_config = getattr(self.config, 'text_config', None) or self.config
        layer_types = getattr(text_config, 'layer_types', None)
        sliding_window = getattr(text_config, 'sliding_window', None)

        # Create the appropriate cache layer type
        if layer_types is not None and layer_idx < len(layer_types):
            if layer_types[layer_idx] == 'sliding_attention' and sliding_window is not None:
                cache_layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)
            else:
                cache_layer = DynamicLayer()
        else:
            cache_layer = DynamicLayer()

        # Pre-populate with existing K/V if provided
        if k_cache is not None and v_cache is not None:
            cache_layer.keys = k_cache.to(device=self.running_device, dtype=self.running_dtype)
            cache_layer.values = v_cache.to(device=self.running_device, dtype=self.running_dtype)
            # Mark as initialized so update() doesn't overwrite with empty tensors
            cache_layer.is_initialized = True
            cache_layer.dtype = self.running_dtype
            cache_layer.device = self.running_device
            # Update cumulative length for sliding window layers
            if hasattr(cache_layer, 'cumulative_length'):
                cache_layer.cumulative_length = k_cache.shape[2]

        # Add the cache layer at the correct index
        # Pad with None layers up to layer_idx so the index is correct
        while len(cache.layers) < layer_idx:
            cache.layers.append(None)
        if len(cache.layers) == layer_idx:
            cache.layers.append(cache_layer)
        else:
            cache.layers[layer_idx] = cache_layer

        return cache

    def extract_kv_from_cache(self, cache, layer_idx):
        """
        Extract K/V tensors from a DynamicCache after a decoder layer call.

        Args:
            cache: The DynamicCache that was passed to the decoder layer
            layer_idx: The decoder layer index to extract from

        Returns:
            (k_cache, v_cache) tuple of tensors
        """
        if layer_idx < len(cache.layers):
            cache_layer = cache.layers[layer_idx]
            if cache_layer is not None and hasattr(cache_layer, 'keys'):
                return cache_layer.keys.clone(), cache_layer.values.clone()
        return None, None

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