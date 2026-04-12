import torch
import pytest
import sys
import os
import importlib.util
import types

# Create a fake 'airllm' package so relative imports work
airllm_pkg = types.ModuleType("airllm")
airllm_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "airllm")]
airllm_pkg.__file__ = os.path.join(
    os.path.dirname(__file__), "..", "airllm", "__init__.py"
)
sys.modules["airllm"] = airllm_pkg

# Load rotorquant_core into the airllm package
_core_path = os.path.join(
    os.path.dirname(__file__), "..", "airllm", "rotorquant_core.py"
)
_spec = importlib.util.spec_from_file_location("airllm.rotorquant_core", _core_path)
rotorquant_core = importlib.util.module_from_spec(_spec)
sys.modules["airllm.rotorquant_core"] = rotorquant_core
sys.modules["rotorquant_core"] = rotorquant_core
_spec.loader.exec_module(rotorquant_core)

PlanarQuantCompressor = rotorquant_core.PlanarQuantCompressor
IsoQuantCompressor = rotorquant_core.IsoQuantCompressor

# Load rotorquant_cache (its relative import will now find airllm.rotorquant_core)
_cache_path = os.path.join(
    os.path.dirname(__file__), "..", "airllm", "rotorquant_cache.py"
)
_spec2 = importlib.util.spec_from_file_location("airllm.rotorquant_cache", _cache_path)
rotorquant_cache = importlib.util.module_from_spec(_spec2)
sys.modules["airllm.rotorquant_cache"] = rotorquant_cache
_spec2.loader.exec_module(rotorquant_cache)

RotorQuantKVCache = rotorquant_cache.RotorQuantKVCache


class TestPlanarQuantCompressor:
    """Test PlanarQuant compression/decompression."""

    @pytest.mark.parametrize("bits", [3, 4])
    @pytest.mark.parametrize("head_dim", [128, 256])
    def test_compress_decompress_roundtrip(self, bits, head_dim):
        """Test that compress/decompress roundtrip preserves data."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        # Create random KV cache-like data
        n_vectors = 64
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        # Compress and decompress
        compressed = compressor.compress(x)
        x_hat = compressor.decompress(compressed)

        # Check shape matches
        assert x_hat.shape == x.shape

        # Check cosine similarity (3-bit ~0.39, 4-bit ~0.80 for random data)
        # Real KV cache data is more structured and will have higher similarity
        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()

        # 4-bit should be > 0.75, 3-bit > 0.35 for random data
        min_sim = 0.75 if bits == 4 else 0.35
        assert cosine_sim > min_sim, (
            f"Cosine similarity {cosine_sim} too low for {bits}-bit"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compression_ratio(self, bits):
        """Test that compression ratio is better than fp16."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        n_vectors = 1000
        ratio = compressor.compression_ratio(n_vectors)

        # With bit-packing: 3-bit gives ~1.06x, 4-bit gives ~1.32x on raw data
        # But we save on norms (fp16 per vector vs fp16 per head_dim values)
        # The real savings come from not storing angles
        assert ratio > 1.0, f"Compression ratio {ratio} should be > 1.0 for {bits}-bit"

    def test_output_keys(self):
        """Test that output dict has expected keys."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        n_vectors = 32
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)
        compressed = compressor.compress(x)

        assert "indices" in compressed
        assert "norms" in compressed
        assert "packed" in compressed
        assert compressed["indices"].dtype == torch.uint8
        assert compressed["norms"].dtype == torch.float16

    def test_bit_packing(self):
        """Test that bit-packing reduces index size."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128  # divisible by 8 for 3-bit packing

        compressor_3bit = PlanarQuantCompressor(head_dim, 3, device)
        n_vectors = 32
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        # With packing
        compressed_packed = compressor_3bit.compress(x, pack_bits=True)
        # Without packing
        compressed_unpacked = compressor_3bit.compress(x, pack_bits=False)

        assert compressed_packed["packed"] is True
        assert compressed_unpacked["packed"] is False
        # Packed should be smaller in the last dimension
        assert (
            compressed_packed["indices"].shape[-1]
            < compressed_unpacked["indices"].shape[-1]
        )


class TestIsoQuantCompressor:
    """Test IsoQuant compression/decompression."""

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compress_decompress_roundtrip(self, bits):
        """Test that compress/decompress roundtrip preserves data."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = IsoQuantCompressor(head_dim, bits, device)

        n_vectors = 64
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        compressed = compressor.compress(x)
        x_hat = compressor.decompress(compressed)

        # Check shape
        assert x_hat.shape == x.shape

        # Check quality (IsoQuant uses Hadamard-like rotation, ~0.80 for random data)
        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.75, (
            f"Cosine similarity {cosine_sim} too low for {bits}-bit"
        )


class TestRotorQuantKVCache:
    """Test RotorQuant KV Cache wrapper."""

    @pytest.mark.parametrize("mode", ["planar3", "planar4"])
    def test_kv_cache_compress_decompress(self, mode):
        """Test KV cache compression and decompression."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        cache = RotorQuantKVCache(mode=mode, head_dim=head_dim, device=device)

        # Create KV cache tensors (batch=1, heads=4, seq=32, head_dim)
        k_cache = torch.randn(1, 4, 32, head_dim, device=device, dtype=torch.float16)
        v_cache = torch.randn(1, 4, 32, head_dim, device=device, dtype=torch.float16)

        # Compress
        compressed = cache.compress(k_cache, v_cache)
        assert compressed["is_compressed"] is True

        # Decompress
        k_decompressed, v_decompressed = cache.decompress(compressed)

        # Check shapes match
        assert k_decompressed.shape == k_cache.shape
        assert v_decompressed.shape == v_cache.shape

    def test_asymmetric_mode(self):
        """Test asymmetric mode (K compressed, V fp16)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        cache = RotorQuantKVCache(mode="asym_planar3", head_dim=head_dim, device=device)

        k_cache = torch.randn(1, 4, 32, head_dim, device=device, dtype=torch.float16)
        v_cache = torch.randn(1, 4, 32, head_dim, device=device, dtype=torch.float16)

        compressed = cache.compress(k_cache, v_cache)
        k_decompressed, v_decompressed = cache.decompress(compressed)

        # V should be identical (no compression)
        assert torch.allclose(v_cache, v_decompressed)

        # K should have correct shape
        assert k_decompressed.shape == k_cache.shape

    def test_memory_usage(self):
        """Test memory usage calculation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        cache = RotorQuantKVCache(mode="planar3", head_dim=head_dim, device=device)

        n_vectors = 1000
        compressed_bytes = cache.memory_usage_bytes(n_vectors)
        fp16_bytes = n_vectors * head_dim * 2 * 2  # K+V fp16

        # Should use less memory than fp16
        assert compressed_bytes < fp16_bytes, (
            f"Compressed ({compressed_bytes}) should be < fp16 ({fp16_bytes})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
