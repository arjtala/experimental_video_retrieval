"""DINOv3 encoder for video frame embeddings.

This module provides a DINOv3-based encoder that can extract:
- Global CLS token embeddings (semantic)
- Patch token features (spatial/structural)
- Attention maps (where the model looks)
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.video import frames_to_tensor


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding matching DINOv3 implementation."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.register_buffer("periods", torch.zeros(self.head_dim // 4))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, num_heads, head_dim = x.shape
        device = x.device
        dtype = x.dtype

        seq_len = h * w
        y_pos = torch.arange(h, device=device).unsqueeze(1).expand(h, w).flatten().float()
        x_pos = torch.arange(w, device=device).unsqueeze(0).expand(h, w).flatten().float()

        periods_tensor: torch.Tensor = self.periods
        rope_dim: int = periods_tensor.shape[0]
        half_rope = rope_dim // 2

        freqs_y = y_pos.unsqueeze(-1) / periods_tensor[:half_rope].unsqueeze(0)
        freqs_x = x_pos.unsqueeze(-1) / periods_tensor[half_rope:].unsqueeze(0)
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)

        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        num_special = N - seq_len
        if num_special > 0:
            x_special = x[:, :num_special]
            x_patches = x[:, num_special:]
        else:
            x_special = None
            x_patches = x

        x_rot = x_patches[..., :rope_dim]
        x_pass = x_patches[..., rope_dim:]

        x1, x2 = x_rot[..., : rope_dim // 2], x_rot[..., rope_dim // 2 :]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos1, cos2 = cos[..., : rope_dim // 2], cos[..., rope_dim // 2 :]
        sin1, sin2 = sin[..., : rope_dim // 2], sin[..., rope_dim // 2 :]

        x_rot_out = torch.cat([x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2], dim=-1)
        x_patches_out = torch.cat([x_rot_out, x_pass], dim=-1)

        if x_special is not None:
            return torch.cat([x_special, x_patches_out], dim=1)
        return x_patches_out


class Attention(nn.Module):
    """Multi-head attention with attention map extraction."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self._attn_weights: torch.Tensor | None = None  # For extraction

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE2D | None,
        h: int,
        w: int,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope is not None:
            q = rope(q, h, w)
            k = rope(k, h, w)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self._attn_weights = attn.detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x)

        if return_attention:
            return out, attn
        return out


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden, dim)
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE2D | None,
        h: int,
        w: int,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), rope, h, w, return_attention=True)
            x = x + self.ls1 * attn_out
            x = x + self.ls2 * self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.ls1 * self.attn(self.norm1(x), rope, h, w)
            return x + self.ls2 * self.mlp(self.norm2(x))


class DINOv3ViT(nn.Module):
    """DINOv3 Vision Transformer with feature extraction utilities."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_storage_tokens: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.num_storage_tokens = num_storage_tokens
        self.num_heads = num_heads

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.rope_embed = RoPE2D(embed_dim, num_heads)

        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.storage_tokens, std=0.02)

    def forward_features(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attention_layer: int = -1,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning all token features.

        Args:
            x: Input images (B, C, H, W).
            return_attention: If True, return attention weights from specified layer.
            attention_layer: Which layer's attention to return (-1 = last).

        Returns:
            Features (B, N, D) or tuple of (features, attention).
        """
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        storage_tokens = self.storage_tokens.expand(B, -1, -1)
        x = torch.cat([x, storage_tokens], dim=1)

        attn_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == (attention_layer % len(self.blocks)):
                x, attn_weights = block(x, self.rope_embed, h, w, return_attention=True)
            else:
                x = block(x, self.rope_embed, h, w)

        x = self.norm(x)

        if return_attention:
            return x, attn_weights
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get CLS token embedding."""
        features = self.forward_features(x)
        return features[:, 0]

    def get_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get patch token features (excluding CLS and storage tokens)."""
        features = self.forward_features(x)
        return features[:, 1 : -self.num_storage_tokens]

    def get_attention_maps(
        self,
        x: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """Get attention weights from a specific layer.

        Args:
            x: Input images (B, C, H, W).
            layer: Which layer (-1 = last).

        Returns:
            Attention weights (B, num_heads, N, N).
        """
        _, attn = self.forward_features(x, return_attention=True, attention_layer=layer)
        return attn


def _load_dinov3_weights(model: DINOv3ViT, weights_path: str) -> None:
    """Load DINOv3 checkpoint with key remapping."""
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith(".ls1.gamma"):
            new_k = k.replace(".ls1.gamma", ".ls1")
        elif k.endswith(".ls2.gamma"):
            new_k = k.replace(".ls2.gamma", ".ls2")
        elif ".attn.qkv.bias_mask" in k:
            continue
        elif k == "patch_embed.proj.weight":
            new_k = "patch_embed.weight"
        elif k == "patch_embed.proj.bias":
            new_k = "patch_embed.bias"
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)


class DINOv3Encoder:
    """High-level DINOv3 encoder for video frame embeddings.

    Provides methods for extracting various representations:
    - Global (CLS) embeddings
    - Patch features
    - Attention maps
    - Attention centroids (for trajectory analysis)
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | torch.device = "cuda",
        img_size: int = 224,
    ):
        """Initialize DINOv3 encoder.

        Args:
            weights_path: Path to DINOv3 checkpoint (local or will reference entity_tracking).
            device: Target device.
            img_size: Input image size.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.img_size = img_size

        # Create model
        self.model = DINOv3ViT(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            num_storage_tokens=4,
            img_size=img_size,
        )

        if weights_path:
            _load_dinov3_weights(self.model, weights_path)

        self.model = self.model.eval().to(self.device)
        self.embedding_dim = 1024

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for the model."""
        import cv2

        processed = []
        for frame in frames:
            # Resize to model input size
            resized = cv2.resize(frame, (self.img_size, self.img_size))
            # Normalize to [0, 1] and convert to tensor
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)

        batch = torch.stack(processed).to(self.device)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (batch - mean) / std

    @torch.no_grad()
    def encode_frames(
        self,
        frames: list[np.ndarray],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode frames to global (CLS) embeddings.

        Args:
            frames: List of RGB frames (H, W, 3).
            batch_size: Batch size for processing.
            normalize: L2-normalize embeddings.

        Returns:
            Embeddings tensor (N, embedding_dim).
        """
        all_embeddings = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch = self._preprocess(batch_frames)
            embeddings = self.model(batch)

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_video(
        self,
        video_path: str,
        sample_rate: int = 5,
        max_frames: int | None = None,
        batch_size: int = 32,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Encode a video to frame embeddings.

        Args:
            video_path: Path to video file.
            sample_rate: Sample every Nth frame.
            max_frames: Maximum frames to process.
            batch_size: Batch size for processing.

        Returns:
            Tuple of (embeddings tensor, metadata dict).
        """
        from ..utils.video import load_video

        frames, fps = load_video(video_path, max_frames=max_frames, sample_rate=sample_rate)

        embeddings = self.encode_frames(frames, batch_size=batch_size)

        metadata = {
            "num_frames": len(frames),
            "fps": fps,
            "sample_rate": sample_rate,
            "video_path": video_path,
        }

        return embeddings, metadata

    @torch.no_grad()
    def get_attention_centroids(
        self,
        frames: list[np.ndarray],
        layer: int = -1,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Compute attention centroid trajectory.

        For each frame, computes the spatial center-of-mass of the CLS token's
        attention over patch tokens. This trajectory encodes camera/subject motion.

        Args:
            frames: List of RGB frames.
            layer: Attention layer to use (-1 = last).
            batch_size: Batch size for processing.

        Returns:
            Centroid positions (N, 2) where columns are (x, y) in [0, 1].
        """
        all_centroids = []
        h = w = self.img_size // 16  # Number of patches per dimension

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch = self._preprocess(batch_frames)

            attn = self.model.get_attention_maps(batch, layer=layer)  # (B, heads, N, N)

            # Average over heads
            attn = attn.mean(dim=1)  # (B, N, N)

            # Get CLS attention over patches (skip CLS at 0, storage at end)
            num_storage = self.model.num_storage_tokens
            cls_attn = attn[:, 0, 1 : -num_storage]  # (B, num_patches)

            # Reshape to spatial grid
            cls_attn = cls_attn.view(-1, h, w)  # (B, h, w)

            # Normalize to probability distribution
            cls_attn = cls_attn / (cls_attn.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Compute centroids
            y_coords = torch.arange(h, device=self.device).float()
            x_coords = torch.arange(w, device=self.device).float()

            centroid_y = (cls_attn.sum(dim=2) * y_coords).sum(dim=1) / h
            centroid_x = (cls_attn.sum(dim=1) * x_coords).sum(dim=1) / w

            centroids = torch.stack([centroid_x, centroid_y], dim=1)  # (B, 2)
            all_centroids.append(centroids)

        return torch.cat(all_centroids, dim=0)

    @torch.no_grad()
    def get_patch_statistics(
        self,
        frames: list[np.ndarray],
        batch_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Compute patch-level statistics for each frame.

        Returns variance, entropy, and PCA of patch tokens - useful for
        capturing spatial texture independent of semantics.

        Args:
            frames: List of RGB frames.
            batch_size: Batch size for processing.

        Returns:
            Dict with 'variance', 'entropy', 'pca_explained' tensors.
        """
        all_variance = []
        all_entropy = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch = self._preprocess(batch_frames)

            patch_features = self.model.get_patch_features(batch)  # (B, num_patches, D)

            # Variance across patches (spatial complexity)
            variance = patch_features.var(dim=1).mean(dim=1)  # (B,)
            all_variance.append(variance)

            # Entropy of patch token distribution
            # Use softmax over patches as pseudo-probability
            probs = F.softmax(patch_features.norm(dim=2), dim=1)  # (B, num_patches)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (B,)
            all_entropy.append(entropy)

        return {
            "variance": torch.cat(all_variance, dim=0),
            "entropy": torch.cat(all_entropy, dim=0),
        }
