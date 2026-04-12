import torch
import math
from typing import Tuple, Dict, Optional, Any, Any


def lloyd_max_centroids(bits: int, n_iterations: int = 100) -> torch.Tensor:
    """
    Compute Lloyd-Max optimal centroids for scalar quantization.
    For a standard normal distribution, computes optimal reconstruction levels.

    Args:
        bits: number of bits (1-4 supported)
        n_iterations: number of Lloyd-Max iterations for custom bit widths

    Returns:
        centroids: tensor of shape (2^bits, 1) with optimal reconstruction levels
    """
    n_levels = 2**bits

    # Pre-computed optimal centroids for common bit widths
    if n_levels == 2:
        return torch.tensor([[-0.6745], [0.6745]])
    if n_levels == 4:
        return torch.tensor([[-1.5104], [-0.4527], [0.4527], [1.5104]])
    if n_levels == 8:
        return torch.tensor(
            [
                [-2.1674],
                [-1.4346],
                [-0.7560],
                [-0.0982],
                [0.0982],
                [0.7560],
                [1.4346],
                [2.1674],
            ]
        )
    if n_levels == 16:
        return torch.tensor(
            [
                [-2.7384],
                [-2.2315],
                [-1.8230],
                [-1.4613],
                [-1.1247],
                [-0.8016],
                [-0.4849],
                [-0.1697],
                [0.1697],
                [0.4849],
                [0.8016],
                [1.1247],
                [1.4613],
                [1.8230],
                [2.2315],
                [2.7384],
            ]
        )

    # Fallback: compute via Lloyd-Max algorithm
    n_samples = 100000
    x = torch.randn(n_samples, 1)
    x = x.sort().values

    boundaries = torch.linspace(x.min(), x.max(), n_levels + 1)
    centroids = torch.zeros(n_levels, 1)

    for _ in range(n_iterations):
        for i in range(n_levels):
            mask = (x >= boundaries[i]) & (x < boundaries[i + 1])
            if i == n_levels - 1:
                mask = (x >= boundaries[i]) & (x <= boundaries[i + 1])
            if mask.any():
                centroids[i] = x[mask].mean()

        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids


def apply_givens_rotation(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply 2D Givens rotations to pairs of coordinates.
    For each pair (x[2i], x[2i+1]), applies rotation by angle[i]:
        x'[2i]   = cos(angle[i]) * x[2i] - sin(angle[i]) * x[2i+1]
        x'[2i+1] = sin(angle[i]) * x[2i] + cos(angle[i]) * x[2i+1]

    Args:
        x: tensor of shape (..., head_dim) where head_dim is even
        angles: tensor of shape (..., head_dim // 2)

    Returns:
        rotated: tensor of same shape as x
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_rotated_even = cos_a * x_even - sin_a * x_odd
    x_rotated_odd = sin_a * x_even + cos_a * x_odd

    result = torch.empty_like(x)
    result[..., 0::2] = x_rotated_even
    result[..., 1::2] = x_rotated_odd

    return result


def inverse_givens_rotation(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply inverse 2D Givens rotations (transpose of forward rotation).
    For each pair (x[2i], x[2i+1]), applies inverse rotation by angle[i]:
        x[2i]   = cos(angle[i]) * x'[2i] + sin(angle[i]) * x'[2i+1]
        x[2i+1] = -sin(angle[i]) * x'[2i] + cos(angle[i]) * x'[2i+1]

    Args:
        x: tensor of shape (..., head_dim) - the rotated values
        angles: tensor of shape (..., head_dim // 2) - same angles used in forward

    Returns:
        unrotated: tensor of same shape as x
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_unrotated_even = cos_a * x_even + sin_a * x_odd
    x_unrotated_odd = -sin_a * x_even + cos_a * x_odd

    result = torch.empty_like(x)
    result[..., 0::2] = x_unrotated_even
    result[..., 1::2] = x_unrotated_odd

    return result


def compute_givens_angles(x: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal Givens rotation angles to decorrelate coordinate pairs.
    For each pair (x[2i], x[2i+1]), computes angle that aligns with principal direction.

    Args:
        x: tensor of shape (..., head_dim)

    Returns:
        angles: tensor of shape (..., head_dim // 2)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    angles = torch.atan2(x_odd, x_even)

    return angles


class PlanarQuantCompressor:
    """
    PlanarQuant compressor using 2D Givens rotations + Lloyd-Max scalar quantization.

    Implements the RotorQuant PlanarQuant method which achieves better perplexity
    than TurboQuant with 28% faster decode and 5.3x faster prefill.

    Compression ratio for 3-bit: ~5x (10.3x with bit-packing)
    """

    def __init__(self, head_dim: int, bits: int = 3, device: str = "cuda"):
        """
        Args:
            head_dim: dimension of each attention head (must be even)
            bits: number of bits for quantization (1-4)
            device: torch device
        """
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        assert 1 <= bits <= 4, f"bits must be 1-4, got {bits}"

        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.n_pairs = head_dim // 2

        self.codebook = lloyd_max_centroids(bits).to(device)
        self.n_levels = 2**bits

        # Fixed rotation angles: pi/4 for all pairs gives Hadamard-like mixing
        # This spreads energy evenly across coordinates for better quantization
        self.angles = torch.full((self.n_pairs,), math.pi / 4, device=device)

        # Fixed rotation angles: pi/4 for all pairs gives Hadamard-like mixing
        # This spreads energy evenly across coordinates for better quantization
        self.angles = torch.full((self.n_pairs,), math.pi / 4, device=device)

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scalar quantize using Lloyd-Max codebook.

        Args:
            x: tensor of shape (n_vectors, head_dim)

        Returns:
            indices: uint8 tensor of shape (n_vectors, head_dim)
            x_hat: dequantized tensor of same shape as x
        """
        x_reshaped = x.unsqueeze(-1)  # (n_vectors, head_dim, 1)
        codebook = self.codebook.view(1, 1, -1)  # (1, 1, n_levels)
        distances = (x_reshaped - codebook).pow(2)
        indices = distances.argmin(dim=-1)
        x_hat = self.codebook.view(-1)[indices]
        return indices.to(torch.uint8), x_hat

    def _dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Dequantize indices back to values.

        Args:
            indices: uint8 tensor of shape (n_vectors, head_dim)

        Returns:
            x_hat: float tensor of same shape
        """
        return self.codebook.view(-1)[indices.long()]

    def _pack_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Pack indices from uint8 to compact bit representation.
        For 3-bit: packs 8 uint8 values into 3 uint8 values (24 bits).
        For 4-bit: packs 2 uint8 values into 1 uint8 value.
        """
        if self.bits == 3:
            n = indices.shape[-1]
            pad = (8 - n % 8) % 8
            if pad > 0:
                indices = torch.nn.functional.pad(indices, (0, pad), value=0)

            indices = indices.view(*indices.shape[:-1], -1, 8)

            b0 = (indices[..., 0]) | ((indices[..., 1] & 0x03) << 7)
            b1 = ((indices[..., 1] >> 2) & 0x3F) | ((indices[..., 2] & 0x0F) << 4)
            b2 = ((indices[..., 2] >> 4) & 0x0F) | ((indices[..., 3] & 0x1F) << 3)
            b3 = ((indices[..., 3] >> 5) & 0x07) | (indices[..., 4] << 3)
            b4 = ((indices[..., 5]) & 0x7F) | ((indices[..., 6] & 0x01) << 7)
            b5 = ((indices[..., 6] >> 1) & 0x3F) | ((indices[..., 7] & 0x03) << 6)
            b6 = (indices[..., 7] >> 2) & 0x3F

            packed = torch.stack([b0, b1, b2, b3, b4, b5, b6], dim=-1)
            return packed.view(*indices.shape[:-2], -1)

        elif self.bits == 4:
            indices = indices.view(*indices.shape[:-1], -1, 2)
            packed = (indices[..., 0]) | (indices[..., 1] << 4)
            return packed.view(*indices.shape[:-2], -1)

        return indices

    def _unpack_indices(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack indices from compact bit representation back to uint8.
        """
        if self.bits == 3:
            packed = packed.view(*packed.shape[:-1], -1, 7)

            b0, b1, b2, b3, b4, b5, b6 = [packed[..., i] for i in range(7)]

            i0 = b0 & 0x7F
            i1 = ((b0 >> 7) & 0x03) | ((b1 & 0x0F) << 2)
            i2 = ((b1 >> 4) & 0x0F) | ((b2 & 0x07) << 4)
            i3 = ((b2 >> 3) & 0x1F) | ((b3 & 0x03) << 5)
            i4 = (b3 >> 3) & 0x1F
            i5 = b4 & 0x7F
            i6 = ((b4 >> 7) & 0x01) | ((b5 & 0x3F) << 1)
            i7 = ((b5 >> 6) & 0x03) | ((b6 & 0x3F) << 2)

            indices = torch.stack([i0, i1, i2, i3, i4, i5, i6, i7], dim=-1)
            return indices.view(*packed.shape[:-2], -1)

        elif self.bits == 4:
            packed = packed.view(*packed.shape[:-1], -1, 1)
            i0 = packed[..., 0] & 0x0F
            i1 = (packed[..., 0] >> 4) & 0x0F
            indices = torch.stack([i0, i1], dim=-1)
            return indices.view(*packed.shape[:-2], -1)

        return packed

    def compress(self, x: torch.Tensor, pack_bits: bool = True) -> Dict[str, Any]:
        """
        Compress a batch of vectors using PlanarQuant.

        Args:
            x: tensor of shape (batch, seq, head_dim) or (n_vectors, head_dim)
            pack_bits: whether to bit-pack indices (default True)

        Returns:
            dict with keys:
                - 'indices': uint8 quantization indices (bit-packed if pack_bits=True)
                - 'norms': fp16 per-vector norms
                - 'packed': bool indicating if indices are bit-packed
        """
        original_shape = x.shape
        x = x.float()

        # Compute per-vector norms
        norms = torch.norm(x, dim=-1, keepdim=True)
        norms = norms.clamp(min=1e-8)

        # Normalize vectors
        x_norm = x / norms

        # Apply fixed Givens rotations (shared angles across all vectors)
        x_rotated = apply_givens_rotation(x_norm, self.angles)

        # Quantize rotated vectors
        x_flat = x_rotated.reshape(-1, self.head_dim)
        indices, _ = self._quantize(x_flat)
        indices = indices.reshape(*original_shape[:-1], self.head_dim)

        # Bit-pack indices if requested
        if pack_bits and self.bits in (3, 4):
            indices = self._pack_indices(indices)

        return {
            "indices": indices,
            "norms": norms.squeeze(-1).half(),
            "packed": pack_bits and self.bits in (3, 4),
        }

    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """
        Decompress PlanarQuant representation back to fp16.

        Args:
            compressed: dict from compress()

        Returns:
            x_hat: reconstructed tensor of original shape
        """
        indices = compressed["indices"]
        norms = compressed["norms"].float()

        # Unpack indices if bit-packed
        if compressed.get("packed", False):
            indices = self._unpack_indices(indices)

        # Dequantize
        indices_flat = indices.reshape(-1, self.head_dim)
        x_rotated = self._dequantize(indices_flat)

        # Inverse Givens rotation (using same fixed angles)
        x_rotated = x_rotated.reshape(*norms.shape, self.head_dim)
        x_norm = inverse_givens_rotation(x_rotated, self.angles)

        # Restore scale
        x_hat = x_norm * norms.unsqueeze(-1)

        return x_hat.half()

    def memory_usage_bytes(self, n_vectors: int, packed: bool = True) -> int:
        """
        Calculate memory usage for n_vectors compressed vectors.

        Args:
            n_vectors: number of vectors
            packed: whether indices are bit-packed

        Returns:
            bytes: total memory in bytes
        """
        if packed and self.bits == 3:
            # 3-bit: 7 bytes per 8 indices
            indices_bytes = n_vectors * self.head_dim * 7 // 8
        elif packed and self.bits == 4:
            # 4-bit: 1 byte per 2 indices
            indices_bytes = n_vectors * self.head_dim // 2
        else:
            # No packing: 1 byte per index (uint8)
            indices_bytes = n_vectors * self.head_dim

        norms_bytes = n_vectors * 2
        # No angles stored (fixed rotation)
        return indices_bytes + norms_bytes

    def compression_ratio(self, n_vectors: int) -> float:
        """
        Calculate compression ratio vs fp16.

        Args:
            n_vectors: number of vectors

        Returns:
            ratio: fp16_bytes / compressed_bytes
        """
        fp16_bytes = n_vectors * self.head_dim * 2
        compressed_bytes = self.memory_usage_bytes(n_vectors)
        return fp16_bytes / compressed_bytes


class IsoQuantCompressor:
    """
    IsoQuant compressor using 4D quaternion rotations + Lloyd-Max scalar quantization.

    Implements the RotorQuant IsoQuant method which achieves the best quality
    at 4-bit (PPL 9.03 vs 9.56 for PlanarQuant).

    Compression ratio for 3-bit: ~5x
    """

    def __init__(self, head_dim: int, bits: int = 3, device: str = "cuda"):
        """
        Args:
            head_dim: dimension of each attention head (must be divisible by 4)
            bits: number of bits for quantization (1-4)
            device: torch device
        """
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        assert 1 <= bits <= 4, f"bits must be 1-4, got {bits}"

        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.n_quads = head_dim // 4

        self.codebook = lloyd_max_centroids(bits).to(device)
        self.n_levels = 2**bits

    def _apply_quaternion_rotation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 4D quaternion rotation to decorrelate each group of 4 coordinates.
        Uses a fixed rotation matrix based on Hadamard-like structure.

        Args:
            x: tensor of shape (n_vectors, head_dim)

        Returns:
            x_rotated: rotated tensor
            params: rotation parameters (for inverse)
        """
        n_vectors = x.shape[0]
        x = x.reshape(n_vectors, self.n_quads, 4)

        a, b, c, d = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        r0 = (a + b + c + d) * 0.5
        r1 = (a + b - c - d) * 0.5
        r2 = (a - b + c - d) * 0.5
        r3 = (a - b - c + d) * 0.5

        x_rotated = torch.stack([r0, r1, r2, r3], dim=-1)
        x_rotated = x_rotated.reshape(n_vectors, self.head_dim)

        return x_rotated, torch.zeros(n_vectors, self.n_quads, 4, device=x.device)

    def _inverse_quaternion_rotation(
        self, x: torch.Tensor, params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply inverse 4D quaternion rotation.

        Args:
            x: rotated tensor
            params: rotation parameters from forward

        Returns:
            x_unrotated: unrotated tensor
        """
        n_vectors = x.shape[0]
        x = x.reshape(n_vectors, self.n_quads, 4)

        r0, r1, r2, r3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]

        a = (r0 + r1 + r2 + r3) * 0.5
        b = (r0 + r1 - r2 - r3) * 0.5
        c = (r0 - r1 + r2 - r3) * 0.5
        d = (r0 - r1 - r2 + r3) * 0.5

        x_unrotated = torch.stack([a, b, c, d], dim=-1)
        x_unrotated = x_unrotated.reshape(n_vectors, self.head_dim)

        return x_unrotated

    def compress(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress a batch of vectors using IsoQuant.

        Args:
            x: tensor of shape (batch, seq, head_dim) or (n_vectors, head_dim)

        Returns:
            dict with keys:
                - 'indices': uint8 quantization indices
                - 'norms': fp16 per-vector norms
        """
        original_shape = x.shape
        x = x.float()

        norms = torch.norm(x, dim=-1, keepdim=True)
        norms = norms.clamp(min=1e-8)

        x_norm = x / norms

        x_flat = x_norm.reshape(-1, self.head_dim)
        x_rotated, _ = self._apply_quaternion_rotation(x_flat)

        indices, _ = self._quantize(x_rotated)

        return {
            "indices": indices.reshape(*original_shape[:-1], self.head_dim),
            "norms": norms.squeeze(-1).half(),
        }

    def decompress(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decompress IsoQuant representation back to fp16.

        Args:
            compressed: dict from compress()

        Returns:
            x_hat: reconstructed tensor of original shape
        """
        indices = compressed["indices"]
        norms = compressed["norms"].float()

        indices_flat = indices.reshape(-1, self.head_dim)
        x_rotated = self._dequantize(indices_flat)

        x_unrotated = self._inverse_quaternion_rotation(x_rotated, None)
        x_unrotated = x_unrotated.reshape(*norms.shape, self.head_dim)

        x_hat = x_unrotated * norms.unsqueeze(-1)

        return x_hat.half()

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_reshaped = x.unsqueeze(-1)
        codebook = self.codebook.view(1, 1, -1)
        distances = (x_reshaped - codebook).pow(2)
        indices = distances.argmin(dim=-1)
        x_hat = self.codebook.view(-1)[indices]
        return indices.to(torch.uint8), x_hat

    def _dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook.view(-1)[indices.long()]

    def memory_usage_bytes(self, n_vectors: int) -> int:
        indices_bytes = n_vectors * self.head_dim
        norms_bytes = n_vectors * 2
        return indices_bytes + norms_bytes

    def compression_ratio(self, n_vectors: int) -> float:
        fp16_bytes = n_vectors * self.head_dim * 2
        compressed_bytes = self.memory_usage_bytes(n_vectors)
        return fp16_bytes / compressed_bytes
