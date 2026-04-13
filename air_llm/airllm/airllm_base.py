from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    GenerationMixin,
    LlamaForCausalLM,
    GenerationConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler

try:
    from optimum.bettertransformer import BetterTransformer

    bettertransformer_installed = True
except ImportError:
    BetterTransformer = None
    bettertransformer_installed = False

from .utils import clean_memory, load_layer, find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print(">>>> bitsandbytes installed")
except ImportError:
    bitsandbytes_installed = False


try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print(">>>> cache_utils installed")
except ImportError:
    cache_utils_installed = False


class AirLLMBaseModel(GenerationMixin):
    # customize layer names here

    # No-op for HuggingFace GenerationMixin compatibility
    # (called by _optimize_model_for_decode during generation)
    def set_experts_implementation(self, implementation):
        pass

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }

    def __init__(
        self,
        model_local_path_or_repo_id,
        device="cuda:0",
        dtype=torch.float16,
        max_seq_len=512,
        layer_shards_saving_path=None,
        profiling_mode=False,
        compression=None,
        hf_token=None,
        prefetching=True,
        delete_original=False,
        kv_compression=None,
        kv_compression_bits=3,
        boundary_layers=0,
    ):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM.

        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        max_seq_len : int, optional
            max seq lenght, by default 512
        layer_shards_saving_path : str, optional
            optional path to save layered shards model file, by default just save to the local cache of model, subdir named splitted_model will be saved
        profiling_mode : book, optional
            if to profile the model loading time, default to False
        compression: str, optinal
            setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
        hf_token: str, optional
            huggingface api token could be provided, by default None
        """

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError(
                    "WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`"
                )

        self.compression = compression
        self.hf_token = hf_token
        self.kv_compression = kv_compression
        self.kv_compression_bits = kv_compression_bits
        self.boundary_layers = boundary_layers
        self.kv_compressor = None
        self.kv_compressor_global = None

        # Save parameters

        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = (
            find_or_create_local_splitted_path(
                model_local_path_or_repo_id,
                layer_shards_saving_path,
                compression=compression,
                layer_names=self.layer_names_dict,
                hf_token=hf_token,
                delete_original=delete_original,
            )
        )
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        # Create model
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, token=hf_token, trust_remote_code=True
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_local_path, trust_remote_code=True
            )

        self.generation_config = self.get_generation_config()
        # print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)
        self.processor = self.get_processor(hf_token=hf_token)

        self.init_model()

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)

        self.layer_names = (
            [self.layer_names_dict["embed"]]
            + [
                f"{self.layer_names_dict['layer_prefix']}.{i}"
                for i in range(layers_count)
            ]
            + [self.layer_names_dict["norm"], self.layer_names_dict["lm_head"]]
        )

        # Add vision tower layers if present
        self.n_vision_layers = 0
        self.has_vision = False
        if 'vision_layer_prefix' in self.layer_names_dict:
            vision_attr = self.model
            for attr_name in self.layer_names_dict["vision_layer_prefix"].split("."):
                vision_attr = getattr(vision_attr, attr_name, None)
                if vision_attr is None:
                    break
            if vision_attr is not None:
                self.n_vision_layers = len(vision_attr)
                self.has_vision = self.n_vision_layers > 0

        if self.has_vision:
            vision_layer_names = (
                [self.layer_names_dict['vision_patch_embedder']]
                + [f"{self.layer_names_dict['vision_layer_prefix']}.{i}" for i in range(self.n_vision_layers)]
                + [self.layer_names_dict['vision_std'], self.layer_names_dict['embed_vision']]
            )
            self.layer_names = vision_layer_names + self.layer_names
            print(f"Vision tower enabled: {self.n_vision_layers} vision layers detected")

        # Track how many non-decoder layers are before and after the decoder layers
        # This is used for KV cache list indexing
        # layer_names layout: [vision_prefixes...] + [embed, layer.0..layer.N-1, norm, lm_head]
        self._embed_idx = self.layer_names.index(self.layer_names_dict["embed"])
        self._norm_idx = self.layer_names.index(self.layer_names_dict["norm"])
        self._lm_head_idx = self.layer_names.index(self.layer_names_dict["lm_head"])

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(
                f"not support prefetching for compression for now. loading with no prepetching mode."
            )

        # this operation should run only if gpu is available
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        # Initialize KV cache compressor if requested
        self.n_layers = layers_count
        if kv_compression is not None:
            from .rotorquant_cache import RotorQuantKVCache

            # For multimodal models (e.g. Gemma4), text-level attributes
            # may be in config.text_config rather than config directly
            text_config = getattr(self.config, "text_config", None)

            head_dim = getattr(self.config, "head_dim", None)
            if head_dim is None and text_config is not None:
                head_dim = getattr(text_config, "head_dim", None)
            if head_dim is None:
                config_for_dims = text_config if text_config is not None else self.config
                hidden_size = getattr(config_for_dims, "hidden_size", None)
                num_attention_heads = getattr(config_for_dims, "num_attention_heads", None)
                if hidden_size is not None and num_attention_heads is not None:
                    head_dim = hidden_size // num_attention_heads
                else:
                    head_dim = 256  # fallback

            self.kv_compressor = RotorQuantKVCache(
                mode=kv_compression,
                bits=kv_compression_bits,
                head_dim=head_dim,
                device=self.running_device,
            )

            # Handle models with mixed head dimensions (e.g., Gemma 4 global attention)
            self.kv_compressor_global = None
            # global_head_dim may be in text_config for multimodal models
            global_head_dim = getattr(self.config, "global_head_dim", None)
            if global_head_dim is None and text_config is not None:
                global_head_dim = getattr(text_config, "global_head_dim", None)
            if global_head_dim is not None and global_head_dim != head_dim:
                self.kv_compressor_global = RotorQuantKVCache(
                    mode=kv_compression,
                    bits=kv_compression_bits,
                    head_dim=global_head_dim,
                    device=self.running_device,
                )
                print(
                    f"KV cache compression enabled: {kv_compression} ({kv_compression_bits}-bit), "
                    f"boundary_layers={boundary_layers}, "
                    f"head_dim={head_dim}, global_head_dim={global_head_dim}"
                )
            else:
                print(
                    f"KV cache compression enabled: {kv_compression} ({kv_compression_bits}-bit), "
                    f"boundary_layers={boundary_layers}, "
                    f"head_dim={head_dim}"
                )

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(
                self.model_local_path, token=hf_token, trust_remote_code=True
            )
        else:
            return AutoTokenizer.from_pretrained(
                self.model_local_path, trust_remote_code=True
            )

    def get_processor(self, hf_token=None):
        """Load the multimodal processor (tokenizer + image processor) if available."""
        try:
            from transformers import AutoProcessor
            if hf_token is not None:
                processor = AutoProcessor.from_pretrained(
                    self.model_local_path, token=hf_token, trust_remote_code=True
                )
            else:
                processor = AutoProcessor.from_pretrained(
                    self.model_local_path, trust_remote_code=True
                )
            # Verify it actually has image processing capabilities
            if hasattr(processor, 'image_processor') and processor.image_processor is not None:
                return processor
            return None
        except (ImportError, Exception):
            return None

    def get_use_better_transformer(self):
        return bettertransformer_installed

    def _is_boundary_layer(self, layer_idx):
        """Check if layer is a boundary layer (should not be compressed)."""
        if self.boundary_layers == 0:
            return False
        n_layers = self.n_layers
        return (
            layer_idx < self.boundary_layers
            or layer_idx >= n_layers - self.boundary_layers
        )

    def get_kv_compressor(self, layer_idx):
        """Return the KV compressor for this layer. Override in subclasses for mixed head dimensions."""
        return self.kv_compressor

    def init_model(self):
        # try way 1 better transformers...
        # Load meta model (no memory used)
        self.model = None

        if self.get_use_better_transformer():
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(
                        self.config, trust_remote_code=True
                    )
                    self.model = BetterTransformer.transform(
                        self.model
                    )  # enable flash attention
            except ValueError as ve:
                del self.model
                clean_memory()
                self.model = None

            if self.model is None:
                # try way 2.
                try:
                    print(
                        f"new version of transfomer, no need to use BetterTransformer, try setting attn impl to sdpa..."
                    )
                    self.config.attn_implementation = "sdpa"

                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(
                            self.config,
                            attn_implementation="sdpa",
                            trust_remote_code=True,
                        )
                    print(f"attn imp: {type(self.model.model.layers[3].self_attn)}")

                except TypeError as ve:
                    del self.model
                    clean_memory()
                    self.model = None

        # fallback to original way
        if self.model is None:
            print(
                f"either BetterTransformer or attn_implementation='sdpa' is available, creating model directly"
            )
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True
                )

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(
                quantization_config, pre_quantized=True
            )
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model,
                buffer_name,
                self.running_device,
                value=buffer,
                dtype=self.running_dtype,
            )

        if "rotary_pos_emb" in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

    def set_layers_from_layer_names(self):
        self.layers = []

        # Add vision tower modules first (if present)
        if 'vision_patch_embedder' in self.layer_names_dict:
            model_attr = self.model
            for attr_name in self.layer_names_dict["vision_patch_embedder"].split("."):
                model_attr = getattr(model_attr, attr_name)
            self.layers.append(model_attr)

        if 'vision_layer_prefix' in self.layer_names_dict:
            model_attr = self.model
            for attr_name in self.layer_names_dict["vision_layer_prefix"].split("."):
                model_attr = getattr(model_attr, attr_name)
            self.layers.extend(list(model_attr))

        if 'vision_std' in self.layer_names_dict:
            model_attr = self.model
            for attr_name in self.layer_names_dict["vision_std"].split("."):
                model_attr = getattr(model_attr, attr_name)
            self.layers.append(model_attr)

        if 'embed_vision' in self.layer_names_dict:
            model_attr = self.model
            for attr_name in self.layer_names_dict["embed_vision"].split("."):
                model_attr = getattr(model_attr, attr_name)
            self.layers.append(model_attr)

        # Language model modules
        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(
            self.checkpoint_path, self.layer_names_dict["rotary_pos_emb"]
        )
        self.move_layer_to_device(state_dict)

    def load_layer_to_cpu(self, layer_name):
        t = time.time()

        load_layer_output = load_layer(
            self.checkpoint_path, layer_name, self.profiling_mode
        )
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time

            self.profiler.add_profiling_time("load_safe_tensor", disk_loading_time)

            self.profiler.add_profiling_time("compression_time", compression_time)
        else:
            state_dict = load_layer_output

        # pin memory:
        if self.prefetching:
            t = time.time()
            if torch.cuda.is_available():  # Check if CUDA is available
                for k in state_dict.keys():
                    state_dict[k].pin_memory()
            else:
                # For CPU, no action is needed, but you could optionally add a log or message
                print(
                    "Prefetching is enabled, but no pin_memory operation is needed for CPU."
                )

            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time(
                    "pin_memory_to_trigger_load", elapsed_time
                )

        return state_dict

    def move_layer_to_device(self, state_dict):
        layers = []
        for param_name, param in state_dict.items():
            if self.hf_quantizer is None:
                layers.append(param_name)
            else:
                if ".weight" in param_name:
                    layer_name = param_name[
                        : param_name.index(".weight") + len(".weight")
                    ]
                    if layer_name not in layers:
                        layers.append(layer_name)

        for param_name in layers:
            if self.hf_quantizer is None or not self.hf_quantizer.check_quantized_param(
                self.model, param_value=None, param_name=param_name, state_dict={}
            ):
                set_module_tensor_to_device(
                    self.model,
                    param_name,
                    self.running_device,
                    value=state_dict[param_name],
                    dtype=self.running_dtype,
                )
            else:
                torch_dtype = self.hf_quantizer.update_torch_dtype(None)
                self.hf_quantizer.create_quantized_param(
                    self.model,
                    state_dict[param_name],
                    param_name,
                    self.running_device,
                    state_dict,
                )
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # Treat empty cache objects as None (transformers 5.x passes empty DynamicCache)
            # DynamicCache has get_seq_length() method; our custom format is a list
            is_empty_cache = False
            if hasattr(past_key_values, 'get_seq_length'):
                # DynamicCache or similar cache object from transformers
                is_empty_cache = past_key_values.get_seq_length() == 0
            if is_empty_cache:
                past_key_values = None
            else:
                past_length = self.get_past_key_values_cache_seq_len(
                    past_key_values
                )  # [0][0].shape[2]

                # Some generation methods already pass only the last input ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        # Pass vision inputs only during prefill (first generation step)
        if past_key_values is None:
            pixel_values = kwargs.get("pixel_values", None)
            pixel_position_ids = kwargs.get("pixel_position_ids", None)
            padding_mask = kwargs.get("padding_mask", None)
            num_soft_tokens = kwargs.get("num_soft_tokens", None)
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
            if pixel_position_ids is not None:
                model_inputs["pixel_position_ids"] = pixel_position_ids
            if padding_mask is not None:
                model_inputs["padding_mask"] = padding_mask
            if num_soft_tokens is not None:
                model_inputs["num_soft_tokens"] = num_soft_tokens

        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        # Handle DynamicCache from transformers >= 4.36
        if hasattr(past_key_values, '_seen_tokens'):
            return past_key_values._seen_tokens
        # Handle Cache objects with key_cache attribute
        if hasattr(past_key_values, 'key_cache'):
            if len(past_key_values.key_cache) > 0:
                return past_key_values.key_cache[0].shape[2]
            return 0
        entry = past_key_values[0]
        if isinstance(entry, dict):
            # Compressed or uncompressed dict format
            if entry.get("is_compressed", False):
                return entry["k_shape"][2]
            else:
                return entry["k"].shape[2]
        # Legacy tuple format: (k_cache, v_cache)
        return entry[0].shape[2]

    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_pos_emb_args(self, len_p, len_s):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {"past_key_value": (k_cache, v_cache)}

    def create_layer_cache(self, layer_idx, k_cache=None, v_cache=None):
        """
        Create a Cache object for models that use Cache-based KV management
        (e.g., Gemma4). Default implementation returns None; subclasses
        that use Cache objects should override this.
        """
        return None

    def extract_kv_from_cache(self, cache, layer_idx):
        """
        Extract K/V tensors from a Cache object after a decoder layer call.
        Default implementation returns None; subclasses that use Cache
        objects should override this.
        """
        return None, None

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {"attention_mask": full_attention_mask[:, :, -len_s:, -len_p - len_s :]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        return {"position_ids": full_position_ids[:, len_p : len_p + len_s]}

    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def _is_vision_layer(self, layer_name):
        """Check if a layer name belongs to the vision tower."""
        if not self.has_vision:
            return False
        vision_prefixes = [
            self.layer_names_dict.get('vision_patch_embedder', ''),
            self.layer_names_dict.get('vision_layer_prefix', ''),
            self.layer_names_dict.get('vision_std', ''),
            self.layer_names_dict.get('embed_vision', ''),
        ]
        for prefix in vision_prefixes:
            if prefix and layer_name.startswith(prefix):
                return True
        return False

    def _is_decoder_layer(self, layer_name):
        """Check if a layer name is a decoder (transformer) layer."""
        return layer_name.startswith(self.layer_names_dict['layer_prefix'])

    def _layer_idx_to_decoder_idx(self, layer_loop_idx):
        """Map a loop index to a decoder layer index (0-based, only decoder layers).

        Decoder layers are indexed starting from 0 (first transformer layer).
        This maps from the position in self.layer_names to the decoder layer number.
        """
        return layer_loop_idx - self._embed_idx - 1

    def merge_image_embeddings(self, inputs_embeds, image_features, input_ids):
        """
        Merge image features into text embeddings by replacing image token positions.
        Override in subclasses for model-specific behavior.
        """
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        num_soft_tokens: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if cache_utils_installed and self.kv_compressor is None:
            # Only disable use_cache if we don't have KV compression.
            # With KV compression, we can handle past_key_values in compressed form.
            use_cache = False

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()

        batch = [
            input_ids_unit.to(self.running_device).unsqueeze(0)
            for input_ids_unit in input_ids
        ]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(
            self.max_seq_len, dtype=torch.long, device=self.running_device
        )[None, :]

        kv_cache_list = [None] * len(self.layers) if use_cache else None
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        # Gemma4 uses shared_kv_states dict for KV sharing between
        # sliding window and full attention layers. Initialized empty,
        # layers populate it during forward pass.
        shared_kv_states = {}

        with torch.inference_mode(), ThreadPoolExecutor() as executor:
            # Find the first layer that actually needs loading (skip tied lm_head etc.)
            first_load_idx = 0
            while first_load_idx < len(self.layer_names):
                first_skip = (
                    self.layer_names[first_load_idx] == self.layer_names_dict["lm_head"]
                    and getattr(self.config, "tie_word_embeddings", False)
                )
                if not first_skip:
                    break
                first_load_idx += 1

            # Load first layer
            if self.prefetching and first_load_idx < len(self.layer_names):
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[first_load_idx])

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                desc=f"running layers({self.running_device})",
                total=len(self.layers),
            ):
                # Skip loading weights for lm_head when tie_word_embeddings=True
                # (run_lm_head uses embed_tokens.weight.T directly)
                skip_loading = (
                    layer_name == self.layer_names_dict["lm_head"]
                    and getattr(self.config, "tie_word_embeddings", False)
                )

                if self.prefetching:
                    if skip_loading:
                        state_dict = {}
                        moved_layers = []
                    else:
                        if self.profiling_mode:
                            t = time.time()
                        # Load current layer and prepare next layer
                        state_dict = future.result()
                        # torch.cuda.current_stream().wait_stream(self.stream)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time(
                                "load_safe_tensor_cpu_wait", elapsed_time
                            )

                        if self.profiling_mode:
                            t = time.time()
                        moved_layers = self.move_layer_to_device(state_dict)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time(
                                "create_layer_from_state_dict", elapsed_time
                            )

                    # kick off next layer loading (skip layers that don't need loading)
                    next_loading_idx = i + 1
                    while next_loading_idx < len(self.layer_names):
                        next_name = self.layer_names[next_loading_idx]
                        next_skip = (
                            next_name == self.layer_names_dict["lm_head"]
                            and getattr(self.config, "tie_word_embeddings", False)
                        )
                        if not next_skip:
                            break
                        next_loading_idx += 1

                    if next_loading_idx < len(self.layer_names):
                        if self.profiling_mode:
                            t = time.time()
                        future = executor.submit(
                            self.load_layer_to_cpu, self.layer_names[next_loading_idx]
                        )
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time(
                                "kick_off_load_cpu", elapsed_time
                            )

                else:
                    if skip_loading:
                        state_dict = {}
                        moved_layers = []
                    else:
                        state_dict = self.load_layer_to_cpu(layer_name)
                        if self.profiling_mode:
                            t = time.time()
                        moved_layers = self.move_layer_to_device(state_dict)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time(
                                "create_layer_from_safe_tensor", elapsed_time
                            )

                # Run layer

                for j, seq in enumerate(batch):
                    # Vision tower layers (only process during prefill when pixel_values provided)
                    if self.has_vision and self._is_vision_layer(layer_name):
                        if pixel_values is not None and past_key_values is None:
                            # Vision layers are processed by forward_vision() in bulk,
                            # not individually here. Skip them in the main loop.
                            pass
                        continue
                    elif layer_name == self.layer_names_dict["embed"]:
                        batch[j] = layer(seq)
                        # Compute position embeddings for Gemma4-style models
                        # that require (cos, sin) per layer type
                        # Use actual sequence length, not max_seq_len
                        if hasattr(self, 'compute_position_embeddings'):
                            actual_seq_len = batch[j].shape[1]
                            self._position_embeddings_data = self.compute_position_embeddings(
                                batch[j], position_ids[:, :actual_seq_len]
                            )
                        # Merge image embeddings after text embedding if vision is available
                        if self.has_vision and pixel_values is not None and past_key_values is None:
                            if hasattr(self, 'forward_vision'):
                                image_features = self.forward_vision(
                                    pixel_values, pixel_position_ids,
                                    padding_mask, num_soft_tokens
                                )
                                if image_features is not None:
                                    batch[j] = self.merge_image_embeddings(
                                        batch[j], image_features, seq
                                    )
                                    # Switch to inputs_embeds mode for subsequent layers
                                    # since we now have merged embeddings
                                    self._using_inputs_embeds = True
                    elif layer_name == self.layer_names_dict["norm"]:
                        # batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
                        batch[j] = self.run_norm(layer, seq)

                        if output_attentions:
                            all_hidden_states[i].append(batch[j])
                    elif layer_name == self.layer_names_dict["lm_head"]:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        # This is a decoder (transformer) layer
                        decoder_idx = self._layer_idx_to_decoder_idx(i)
                        # Store current decoder index for position embedding lookup
                        self._current_decoder_idx = decoder_idx

                        if output_attentions:
                            all_hidden_states[i].append(new_seq)

                        # Check if this model uses Cache objects (e.g., Gemma4)
                        uses_cache_obj = getattr(self, 'uses_cache_object', False)

                        if past_key_values is not None:
                            # join past kv
                            entry = past_key_values[decoder_idx]
                            if isinstance(entry, dict) and entry.get("is_compressed", False):
                                compressor = self.get_kv_compressor(decoder_idx)
                                k_cache, v_cache = compressor.decompress(entry)
                            elif isinstance(entry, dict):
                                k_cache, v_cache = entry["k"], entry["v"]
                            else:
                                k_cache, v_cache = entry
                            len_p = self.get_past_key_values_cache_seq_len(
                                past_key_values
                            )
                            len_s = self.get_sequence_len(seq)

                            position_ids_args = self.get_position_ids_args(
                                position_ids, len_p, len_s
                            )
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, len_p, len_s
                            )

                            if uses_cache_obj:
                                # Create a Cache object pre-populated with K/V
                                layer_cache = self.create_layer_cache(
                                    decoder_idx, k_cache, v_cache
                                )
                                past_key_value_args = {"past_key_values": layer_cache}
                            else:
                                past_key_value_args = self.get_past_key_value_args(
                                    k_cache, v_cache
                                )

                            kwargs = {
                                "use_cache": True,
                                "shared_kv_states": shared_kv_states,
                            }

                            pos_embed_args = self.get_pos_emb_args(len_p, len_s)
                            kwargs = {
                                **kwargs,
                                **past_key_value_args,
                                **pos_embed_args,
                                **attention_mask_args,
                                **position_ids_args,
                            }

                            if uses_cache_obj:
                                # Cache-object models: layer returns only hidden_states
                                new_seq = layer(seq, **kwargs)

                                if use_cache:
                                    k_cache, v_cache = self.extract_kv_from_cache(
                                        layer_cache, decoder_idx
                                    )
                                    if k_cache is not None:
                                        if (
                                            self.kv_compressor is not None
                                            and not self._is_boundary_layer(decoder_idx)
                                        ):
                                            compressor = self.get_kv_compressor(decoder_idx)
                                            kv_cache_list[i] = compressor.compress(
                                                k_cache, v_cache
                                            )
                                        else:
                                            kv_cache_list[i] = {
                                                "k": k_cache,
                                                "v": v_cache,
                                                "is_compressed": False,
                                            }
                            else:
                                # Traditional models: layer returns (hidden_states, attn, (k, v))
                                layer_outputs = layer(seq, **kwargs)
                                new_seq = layer_outputs[0]

                                if output_attentions:
                                    all_self_attns[i].append(layer_outputs[1])

                                if use_cache:
                                    (k_cache, v_cache) = layer_outputs[
                                        2 if output_attentions else 1
                                    ]
                                    if (
                                        self.kv_compressor is not None
                                        and not self._is_boundary_layer(decoder_idx)
                                    ):
                                        compressor = self.get_kv_compressor(decoder_idx)
                                        kv_cache_list[i] = compressor.compress(
                                            k_cache, v_cache
                                        )
                                    else:
                                        kv_cache_list[i] = {
                                            "k": k_cache,
                                            "v": v_cache,
                                            "is_compressed": False,
                                        }

                        else:
                            len_seq = self.get_sequence_len(seq)

                            pos_embed_args = self.get_pos_emb_args(0, len_seq)
                            attention_mask_args = self.get_attention_mask_args(
                                attention_mask, 0, len_seq
                            )
                            position_ids_args = self.get_position_ids_args(
                                position_ids, 0, len_seq
                            )

                            if not use_cache:
                                kwargs = {
                                    "use_cache": False,
                                    "shared_kv_states": shared_kv_states,
                                    "attention_mask": attention_mask[
                                        :, :, -len_seq:, -len_seq:
                                    ],
                                }
                                kwargs = {
                                    **kwargs,
                                    **pos_embed_args,
                                    **attention_mask_args,
                                    **position_ids_args,
                                }

                                if uses_cache_obj:
                                    new_seq = layer(seq, **kwargs)
                                else:
                                    new_seq = layer(seq, **kwargs)[0]
                            else:
                                if uses_cache_obj:
                                    # Create empty cache for prefill
                                    layer_cache = self.create_layer_cache(decoder_idx)
                                    kwargs = {
                                        "use_cache": True,
                                        "past_key_values": layer_cache,
                                        "shared_kv_states": shared_kv_states,
                                        "attention_mask": attention_mask[
                                            :, :, -len_seq:, -len_seq:
                                        ],
                                    }
                                    kwargs = {
                                        **kwargs,
                                        **pos_embed_args,
                                        **attention_mask_args,
                                        **position_ids_args,
                                    }

                                    try:
                                        new_seq = layer(seq, **kwargs)
                                    except RuntimeError as e:
                                        if 'meta' in str(e).lower() or 'device' in str(e).lower():
                                            print(f"  FAILED layer_name={layer_name}, decoder_idx={decoder_idx}")
                                            meta_params = [pn for pn, pp in layer.named_parameters() if pp.device.type == 'meta']
                                            if meta_params:
                                                print(f"  Meta params: {meta_params[:5]}...")
                                        raise

                                    # Extract K/V from cache
                                    k_cache, v_cache = self.extract_kv_from_cache(
                                        layer_cache, decoder_idx
                                    )
                                    if k_cache is not None:
                                        if (
                                            self.kv_compressor is not None
                                            and not self._is_boundary_layer(decoder_idx)
                                        ):
                                            compressor = self.get_kv_compressor(decoder_idx)
                                            kv_cache_list[i] = compressor.compress(
                                                k_cache, v_cache
                                            )
                                        else:
                                            kv_cache_list[i] = {
                                                "k": k_cache,
                                                "v": v_cache,
                                                "is_compressed": False,
                                            }
                                else:
                                    kwargs = {
                                        "use_cache": True,
                                        "attention_mask": attention_mask[
                                            :, :, -len_seq:, -len_seq:
                                        ],
                                    }
                                    kwargs = {
                                        **kwargs,
                                        **pos_embed_args,
                                        **attention_mask_args,
                                        **position_ids_args,
                                    }

                                    layer_out = layer(seq, **kwargs)

                                    new_seq, (k_cache, v_cache) = layer_out
                                    if (
                                        self.kv_compressor is not None
                                        and not self._is_boundary_layer(decoder_idx)
                                    ):
                                        compressor = self.get_kv_compressor(decoder_idx)
                                        kv_cache_list[i] = compressor.compress(
                                            k_cache, v_cache
                                        )
                                    else:
                                        kv_cache_list[i] = {
                                            "k": k_cache,
                                            "v": v_cache,
                                            "is_compressed": False,
                                        }

                        batch[j] = new_seq

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Remove previous layer from memory (including buffers)

                if self.hf_quantizer is not None:
                    for (
                        param_name
                    ) in moved_layers:  # param_name, param in state_dict.items():
                        set_module_tensor_to_device(self.model, param_name, "meta")
                else:
                    layer.to("meta")

                layer.to("meta")
                clean_memory()  # proposed by CPMP

        logits = torch.cat(batch, 0)
        if use_cache:
            # Remove non-decoder entries (vision layers, embed, norm, lm_head)
            # from the KV cache list. Only decoder transformer layers have KV cache.
            # _embed_idx is the index of embed layer, _norm_idx and _lm_head_idx
            # are the indices of norm and lm_head layers.
            # Decoder layers are between embed and norm (exclusive).
            decoder_start = self._embed_idx + 1  # first decoder layer
            decoder_end = self._norm_idx  # one past last decoder layer
            kv_cache_list = kv_cache_list[decoder_start:decoder_end]
            # KV cache entries are now dicts (compressed or uncompressed),
            # no need for concatenation since we store one entry per layer per batch item

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        if not return_dict:
            return tuple(
                v
                for v in [
                    logits,
                    tuple(kv_cache_list) if kv_cache_list is not None else None,
                    tuple(all_hidden_states) if all_hidden_states is not None else None,
                    tuple(all_self_attns) if all_self_attns is not None else None,
                ]
                if v is not None
            )
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()

            print(
                f"total infer process time(including all above plus gpu compute): {forward_elapsed_time:.04f}"
            )
            print(
                f"total infer wall time(including all above plus gpu compute): {forward_elapsed_time_wall:.04f}"
            )

            self.profiler.clear_profiling_time()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states)
            if all_hidden_states is not None
            else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )
