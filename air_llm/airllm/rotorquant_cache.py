import torch
from typing import Dict, Any, Optional, Tuple, List
from .rotorquant_core import PlanarQuantCompressor, IsoQuantCompressor


class RotorQuantKVCache:
    """
    RotorQuant KV Cache wrapper with PlanarQuant/IsoQuant compression.

    Compresses KV cache tensors during generation to save VRAM.
    Supports multiple compression modes and boundary layer protection.

    Modes:
        - 'planar3': 3-bit PlanarQuant (default, best speed/quality balance)
        - 'planar4': 4-bit PlanarQuant (better quality)
        - 'iso3': 3-bit IsoQuant (better quality than planar3)
        - 'iso4': 4-bit IsoQuant (best quality)
        - 'asym_planar3': K=planar3, V=fp16 (zero PPL loss, K-only compression)

    Compression ratio for planar3: ~5x vs fp16
    """

    def __init__(
        self,
        mode: str = "planar3",
        bits: int = 3,
        head_dim: int = 128,
        device: str = "cuda",
    ):
        """
        Args:
            mode: compression mode ('planar3', 'planar4', 'iso3', 'iso4', 'asym_planar3')
            bits: number of bits for quantization (1-4)
            head_dim: dimension of each attention head
            device: torch device
        """
        self.mode = mode
        self.head_dim = head_dim
        self.device = device

        # Initialize compressors based on mode
        if "planar" in mode:
            self.k_compressor = PlanarQuantCompressor(head_dim, bits, device)
        elif "iso" in mode:
            self.k_compressor = IsoQuantCompressor(head_dim, bits, device)
        else:
            raise ValueError(f"Unknown compression mode: {mode}")

        # V compression: asymmetric modes keep V in fp16
        if mode.startswith("asym_"):
            self.v_compressor = None
        else:
            if "planar" in mode:
                self.v_compressor = PlanarQuantCompressor(head_dim, bits, device)
            elif "iso" in mode:
                self.v_compressor = IsoQuantCompressor(head_dim, bits, device)

    def compress(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compress KV cache tensors.

        Args:
            k_cache: key cache tensor of shape (batch, n_heads, seq, head_dim)
            v_cache: value cache tensor of shape (batch, n_heads, seq, head_dim)

        Returns:
            compressed: dict with compressed k and v representations
        """
        # Reshape to (n_vectors, head_dim) for compressor
        k_shape = k_cache.shape
        v_shape = v_cache.shape

        k_flat = k_cache.reshape(-1, self.head_dim)
        v_flat = v_cache.reshape(-1, self.head_dim)

        # Compress K
        compressed_k = self.k_compressor.compress(k_flat)

        # Compress V (or keep as fp16 for asymmetric modes)
        if self.v_compressor is not None:
            compressed_v = self.v_compressor.compress(v_flat)
        else:
            compressed_v = {"tensor": v_cache.half()}

        return {
            "k": compressed_k,
            "v": compressed_v,
            "k_shape": list(k_shape),
            "v_shape": list(v_shape),
            "is_compressed": True,
        }

    def decompress(
        self,
        compressed: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress KV cache tensors.

        Args:
            compressed: dict from compress()

        Returns:
            k_cache: decompressed key cache
            v_cache: decompressed value cache
        """
        if not compressed.get("is_compressed", False):
            # Already uncompressed
            return compressed["k"], compressed["v"]

        k_shape = tuple(compressed["k_shape"])
        v_shape = tuple(compressed["v_shape"])

        # Decompress K
        k_flat = self.k_compressor.decompress(compressed["k"])
        k_cache = k_flat.reshape(k_shape)

        # Decompress V (or return as-is for asymmetric modes)
        if self.v_compressor is not None:
            v_flat = self.v_compressor.decompress(compressed["v"])
            v_cache = v_flat.reshape(v_shape)
        else:
            v_cache = compressed["v"]["tensor"]

        return k_cache, v_cache

    def memory_usage_bytes(self, n_vectors: int) -> int:
        """
        Calculate memory usage for n_vectors compressed KV pairs.

        Args:
            n_vectors: number of vectors per head

        Returns:
            bytes: total memory in bytes for K+V cache
        """
        k_bytes = self.k_compressor.memory_usage_bytes(n_vectors)

        if self.v_compressor is not None:
            v_bytes = self.v_compressor.memory_usage_bytes(n_vectors)
        else:
            v_bytes = n_vectors * self.head_dim * 2  # fp16

        return k_bytes + v_bytes

    def compression_ratio(self, n_vectors: int) -> float:
        """
        Calculate compression ratio vs fp16.

        Args:
            n_vectors: number of vectors per head

        Returns:
            ratio: fp16_bytes / compressed_bytes
        """
        fp16_bytes = n_vectors * self.head_dim * 2 * 2  # K+V
        compressed_bytes = self.memory_usage_bytes(n_vectors)
        return fp16_bytes / compressed_bytes
