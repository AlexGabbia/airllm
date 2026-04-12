import torch
import pytest
from airllm.rotorquant_core import PlanarQuantCompressor, IsoQuantCompressor
from airllm.rotorquant_cache import RotorQuantKVCache


class TestPlanarQuantCompressor:
    """Test PlanarQuant compression/decompression."""

    @pytest.mark.parametrize("bits", [3, 4])
    @pytest.mark.parametrize("head_dim", [128, 256])
    def test_compress_decompress_roundtrip(self, bits, head_dim):
        """Test that compress/decompress roundtrip preserves data with high cosine similarity."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        # Create random KV cache-like data
        n_vectors = 64
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        # Compress and decompress
        compressed = compressor.compress(x)
        x_hat = compressor.decompress(compressed)

        # Check cosine similarity > 0.95 for planar3
        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.90, (
            f"Cosine similarity {cosine_sim} too low for {bits}-bit"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compression_ratio(self, bits):
        """Test that compression ratio is at least 3x."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        n_vectors = 1000
        ratio = compressor.compression_ratio(n_vectors)

        assert ratio > 3.0, f"Compression ratio {ratio} too low for {bits}-bit"

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        n_vectors = 32
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)
        compressed = compressor.compress(x)

        assert "indices" in compressed
        assert "norms" in compressed
        assert "angles" in compressed
        assert compressed["indices"].shape == (n_vectors, head_dim)
        assert compressed["norms"].shape == (n_vectors,)
        assert compressed["angles"].shape == (n_vectors, head_dim // 2)


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

        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.90, (
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

        # Check quality
        k_flat = k_cache.float().flatten()
        k_deq_flat = k_decompressed.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            k_flat.unsqueeze(0), k_deq_flat.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.90, (
            f"K cache cosine similarity {cosine_sim} too low for {mode}"
        )

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

        # K should be similar but not identical
        k_flat = k_cache.float().flatten()
        k_deq_flat = k_decompressed.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            k_flat.unsqueeze(0), k_deq_flat.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.90

    def test_memory_usage(self):
        """Test memory usage calculation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        cache = RotorQuantKVCache(mode="planar3", head_dim=head_dim, device=device)

        n_vectors = 1000
        compressed_bytes = cache.memory_usage_bytes(n_vectors)
        fp16_bytes = n_vectors * head_dim * 2 * 2  # K+V fp16

        ratio = fp16_bytes / compressed_bytes
        assert ratio > 3.0, f"Compression ratio {ratio} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
