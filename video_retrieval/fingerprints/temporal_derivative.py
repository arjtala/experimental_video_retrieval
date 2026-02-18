"""Temporal Derivative Fingerprinting.

The key insight: two videos may share semantic content (e.g., Central Park),
but the *sequence of transitions* between frames is unique. By computing
the temporal derivative of embeddings (how the embedding changes frame-to-frame),
we capture motion and trajectory information independent of static content.

Example:
- Cyclist A: Harlem → Central Park → Financial District
- Cyclist B: Chelsea → Central Park → Queens

Both have "Central Park" embeddings, but:
- A's derivative shows: urban→park transition, then park→downtown transition
- B's derivative shows: urban→park transition, then park→residential transition

These derivative patterns are distinct fingerprints.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any


class TemporalDerivativeFingerprint:
    """Generate fingerprints from temporal derivatives of video embeddings.

    Captures how a video *moves through* embedding space, not just where it is.
    """

    def __init__(
        self,
        derivative_order: int = 1,
        window_size: int = 1,
        aggregation: str = "mean",
        include_magnitude: bool = True,
        include_direction: bool = True,
    ):
        """Initialize fingerprinter.

        Args:
            derivative_order: Order of temporal derivative (1=velocity, 2=acceleration).
            window_size: Frames to skip when computing derivatives (larger = coarser motion).
            aggregation: How to aggregate derivatives ("mean", "std", "histogram", "sequence").
            include_magnitude: Include derivative magnitudes in fingerprint.
            include_direction: Include derivative directions in fingerprint.
        """
        self.derivative_order = derivative_order
        self.window_size = window_size
        self.aggregation = aggregation
        self.include_magnitude = include_magnitude
        self.include_direction = include_direction

    def compute_derivatives(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal derivatives of embeddings.

        Args:
            embeddings: Frame embeddings (T, D).

        Returns:
            Derivatives tensor (T-window_size, D) for first order.
        """
        derivatives = embeddings

        for _ in range(self.derivative_order):
            # d[t] = embed[t + window] - embed[t]
            derivatives = derivatives[self.window_size :] - derivatives[: -self.window_size]

        return derivatives

    def compute_fingerprint(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute fingerprint from video embeddings.

        Args:
            embeddings: Frame embeddings (T, D).

        Returns:
            Dict containing fingerprint components.
        """
        derivatives = self.compute_derivatives(embeddings)

        result = {}

        if self.include_magnitude:
            # Magnitude of change at each timestep
            magnitudes = torch.norm(derivatives, dim=1)  # (T-k,)
            result["magnitudes"] = magnitudes

            if self.aggregation == "mean":
                result["mean_magnitude"] = magnitudes.mean()
                result["std_magnitude"] = magnitudes.std()
            elif self.aggregation == "histogram":
                # Histogram of magnitudes
                result["magnitude_histogram"] = torch.histc(magnitudes, bins=20)

        if self.include_direction:
            # Normalized direction vectors
            directions = F.normalize(derivatives, p=2, dim=1)  # (T-k, D)
            result["directions"] = directions

            if self.aggregation == "mean":
                # Mean direction (may cancel out for oscillatory motion)
                result["mean_direction"] = directions.mean(dim=0)
                # Variance of directions (spread of motion)
                result["direction_variance"] = directions.var(dim=0).mean()

        # Transition signature: sequence of direction changes
        if derivatives.shape[0] > 1:
            direction_changes = directions[1:] - directions[:-1]
            result["direction_changes"] = direction_changes
            result["mean_direction_change"] = torch.norm(direction_changes, dim=1).mean()

        return result

    def fingerprint_to_vector(
        self,
        fingerprint: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert fingerprint dict to a single vector for comparison.

        Args:
            fingerprint: Output from compute_fingerprint.

        Returns:
            1D tensor suitable for distance computation.
        """
        components = []

        if "mean_magnitude" in fingerprint:
            components.append(fingerprint["mean_magnitude"].unsqueeze(0))
        if "std_magnitude" in fingerprint:
            components.append(fingerprint["std_magnitude"].unsqueeze(0))
        if "mean_direction" in fingerprint:
            components.append(fingerprint["mean_direction"])
        if "direction_variance" in fingerprint:
            components.append(fingerprint["direction_variance"].unsqueeze(0))
        if "mean_direction_change" in fingerprint:
            components.append(fingerprint["mean_direction_change"].unsqueeze(0))
        if "magnitude_histogram" in fingerprint:
            # Normalize histogram
            hist = fingerprint["magnitude_histogram"]
            components.append(hist / (hist.sum() + 1e-8))

        if not components:
            raise ValueError("No components to create fingerprint vector")

        return torch.cat(components)

    def compare(
        self,
        fp1: dict[str, torch.Tensor],
        fp2: dict[str, torch.Tensor],
        metric: str = "cosine",
    ) -> float:
        """Compare two fingerprints.

        Args:
            fp1, fp2: Fingerprints from compute_fingerprint.
            metric: Distance metric ("cosine", "euclidean", "dtw").

        Returns:
            Similarity score (higher = more similar for cosine).
        """
        v1 = self.fingerprint_to_vector(fp1)
        v2 = self.fingerprint_to_vector(fp2)

        if metric == "cosine":
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        elif metric == "euclidean":
            return -torch.norm(v1 - v2).item()  # Negative so higher = more similar
        else:
            raise ValueError(f"Unknown metric: {metric}")


class MultiScaleDerivativeFingerprint:
    """Compute temporal derivatives at multiple time scales.

    Different motions operate at different temporal scales:
    - Fine-grained: camera shake, small movements (1-2 frames)
    - Medium: scene transitions (5-10 frames)
    - Coarse: narrative/journey progression (30+ frames)

    Combining multiple scales captures motion at all levels.
    """

    def __init__(
        self,
        window_sizes: list[int] = [1, 5, 15, 30],
        aggregation: str = "mean",
    ):
        """Initialize multi-scale fingerprinter.

        Args:
            window_sizes: List of window sizes for derivative computation.
            aggregation: How to aggregate per-scale derivatives.
        """
        self.window_sizes = window_sizes
        self.fingerprinters = [
            TemporalDerivativeFingerprint(
                derivative_order=1,
                window_size=ws,
                aggregation=aggregation,
            )
            for ws in window_sizes
        ]

    def compute_fingerprint(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute multi-scale fingerprint.

        Args:
            embeddings: Frame embeddings (T, D).

        Returns:
            Dict with per-scale fingerprints and combined vector.
        """
        result = {"scales": {}}
        combined_components = []

        for ws, fp in zip(self.window_sizes, self.fingerprinters):
            if embeddings.shape[0] > ws:
                scale_fp = fp.compute_fingerprint(embeddings)
                result["scales"][ws] = scale_fp
                combined_components.append(fp.fingerprint_to_vector(scale_fp))

        if combined_components:
            result["combined"] = torch.cat(combined_components)

        return result

    def compare(
        self,
        fp1: dict[str, Any],
        fp2: dict[str, Any],
    ) -> float:
        """Compare multi-scale fingerprints."""
        v1 = fp1.get("combined")
        v2 = fp2.get("combined")

        if v1 is None or v2 is None:
            return 0.0

        # Handle different lengths (from different video lengths)
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]

        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
