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


class TestBitPacking3Bit:
    """Test 3-bit bit-packing roundtrip correctness."""

    def test_pack_unpack_roundtrip(self):
        """Test that 3-bit pack/unpack preserves index values exactly."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128  # Must be divisible by 8
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        # Create known indices (values 0-7 for 3-bit)
        indices = torch.randint(0, 8, (64, head_dim), device=device, dtype=torch.uint8)

        # Pack and unpack
        packed = compressor._pack_indices(indices)
        unpacked = compressor._unpack_indices(packed)

        # Trim padding if any
        assert torch.equal(indices, unpacked[:, :head_dim]), (
            f"3-bit packing roundtrip failed: "
            f"max diff = {(indices[:, :head_dim].long() - unpacked[:, :head_dim].long()).abs().max()}"
        )

    def test_packed_size(self):
        """Test that 3-bit packing produces the correct output size."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        indices = torch.randint(0, 8, (64, head_dim), device=device, dtype=torch.uint8)
        packed = compressor._pack_indices(indices)

        # 128 indices * 3 bits = 384 bits = 48 bytes
        expected_packed_len = head_dim * 3 // 8
        assert packed.shape[-1] == expected_packed_len, (
            f"Packed size {packed.shape[-1]} != expected {expected_packed_len}"
        )

    def test_all_zeros(self):
        """Test 3-bit packing with all-zero indices."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        indices = torch.zeros(32, head_dim, device=device, dtype=torch.uint8)
        packed = compressor._pack_indices(indices)
        unpacked = compressor._unpack_indices(packed)

        assert torch.equal(indices, unpacked[:, :head_dim])

    def test_all_max_values(self):
        """Test 3-bit packing with all max values (7)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        indices = torch.full((32, head_dim), 7, device=device, dtype=torch.uint8)
        packed = compressor._pack_indices(indices)
        unpacked = compressor._unpack_indices(packed)

        assert torch.equal(indices, unpacked[:, :head_dim])

    def test_non_divisible_head_dim(self):
        """Test 3-bit packing with head_dim not divisible by 8 (requires padding)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 100  # Not divisible by 8, requires 4 padding values
        compressor = PlanarQuantCompressor(head_dim, 3, device)

        indices = torch.randint(0, 8, (32, head_dim), device=device, dtype=torch.uint8)
        packed = compressor._pack_indices(indices)
        unpacked = compressor._unpack_indices(packed)

        assert torch.equal(indices, unpacked[:, :head_dim])


class TestBitPacking4Bit:
    """Test 4-bit bit-packing roundtrip correctness."""

    def test_pack_unpack_roundtrip(self):
        """Test that 4-bit pack/unpack preserves index values exactly."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 4, device)

        indices = torch.randint(0, 16, (64, head_dim), device=device, dtype=torch.uint8)

        packed = compressor._pack_indices(indices)
        unpacked = compressor._unpack_indices(packed)

        assert torch.equal(indices, unpacked), "4-bit packing roundtrip failed"

    def test_packed_size(self):
        """Test that 4-bit packing produces the correct output size."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, 4, device)

        indices = torch.randint(0, 16, (64, head_dim), device=device, dtype=torch.uint8)
        packed = compressor._pack_indices(indices)

        # 128 indices * 4 bits = 512 bits = 64 bytes
        expected_packed_len = head_dim // 2
        assert packed.shape[-1] == expected_packed_len


class TestPlanarQuantCompressor:
    """Test PlanarQuant compression/decompression."""

    @pytest.mark.parametrize("bits", [3, 4])
    @pytest.mark.parametrize("head_dim", [128, 256])
    def test_compress_decompress_roundtrip(self, bits, head_dim):
        """Test that compress/decompress roundtrip preserves data."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        n_vectors = 64
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        compressed = compressor.compress(x)
        x_hat = compressor.decompress(compressed)

        assert x_hat.shape == x.shape

        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()

        min_sim = 0.75 if bits == 4 else 0.35
        assert cosine_sim > min_sim, (
            f"Cosine similarity {cosine_sim} too low for {bits}-bit"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compress_with_packing_roundtrip(self, bits):
        """Test full compress/decompress with bit-packing enabled."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        n_vectors = 64
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        compressed = compressor.compress(x, pack_bits=True)
        assert compressed["packed"] is True

        x_hat = compressor.decompress(compressed)
        assert x_hat.shape == x.shape

        # Verify packing actually reduced size
        uncompressed = compressor.compress(x, pack_bits=False)
        assert compressed["indices"].numel() < uncompressed["indices"].numel()

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compression_ratio(self, bits):
        """Test that compression ratio is better than fp16."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = PlanarQuantCompressor(head_dim, bits, device)

        n_vectors = 1000
        ratio = compressor.compression_ratio(n_vectors)
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

        assert x_hat.shape == x.shape

        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.75, (
            f"Cosine similarity {cosine_sim} too low for {bits}-bit"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compress_with_packing_roundtrip(self, bits):
        """Test IsoQuant compress/decompress with bit-packing."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = IsoQuantCompressor(head_dim, bits, device)

        n_vectors = 64
        x = torch.randn(n_vectors, head_dim, device=device, dtype=torch.float16)

        compressed = compressor.compress(x, pack_bits=True)
        assert compressed["packed"] is True

        x_hat = compressor.decompress(compressed)
        assert x_hat.shape == x.shape

        # Verify quality with packing
        x_flat = x.float().flatten()
        x_hat_flat = x_hat.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)
        ).item()
        assert cosine_sim > 0.75, (
            f"Cosine similarity {cosine_sim} too low for {bits}-bit with packing"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_compression_ratio(self, bits):
        """Test IsoQuant compression ratio."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        compressor = IsoQuantCompressor(head_dim, bits, device)

        n_vectors = 1000
        ratio = compressor.compression_ratio(n_vectors)
        assert ratio > 1.0, f"Compression ratio {ratio} should be > 1.0 for {bits}-bit"


class TestRotorQuantKVCache:
    """Test RotorQuant KV Cache wrapper."""

    @pytest.mark.parametrize("mode", ["planar3", "planar4", "iso3", "iso4"])
    def test_kv_cache_compress_decompress(self, mode):
        """Test KV cache compression and decompression."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = 128
        bits = int(mode[-1])
        cache = RotorQuantKVCache(mode=mode, head_dim=head_dim, device=device)

        k_cache = torch.randn(1, 4, 32, head_dim, device=device, dtype=torch.float16)
        v_cache = torch.randn(1, 4, 32, head_dim, device=device, dtype=torch.float16)

        compressed = cache.compress(k_cache, v_cache)
        assert compressed["is_compressed"] is True

        k_decompressed, v_decompressed = cache.decompress(compressed)

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

        assert compressed_bytes < fp16_bytes, (
            f"Compressed ({compressed_bytes}) should be < fp16 ({fp16_bytes})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])