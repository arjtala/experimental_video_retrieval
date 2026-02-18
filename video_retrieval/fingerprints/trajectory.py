"""Trajectory-based fingerprinting using attention centroids.

The key insight: DINOv3's attention heads track salient objects and regions.
The trajectory of attention centroids over time encodes motion patterns
that are independent of what specific content is being attended to.

Example:
- Cyclist A in NYC: attention tracks cyclist moving left-to-right across frame
- Cyclist B in Paris: attention tracks cyclist moving left-to-right across frame
- Both have similar attention trajectories despite different semantic content

This makes attention trajectories a non-semantic motion fingerprint.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any


class TrajectoryFingerprint:
    """Generate fingerprints from attention centroid trajectories.

    Uses the trajectory of attention center-of-mass over time as a
    motion signature independent of semantic content.
    """

    def __init__(
        self,
        use_velocity: bool = True,
        use_acceleration: bool = True,
        use_curvature: bool = True,
        smoothing_window: int = 3,
        aggregation: str = "statistics",
    ):
        """Initialize trajectory fingerprinter.

        Args:
            use_velocity: Include velocity (first derivative) in fingerprint.
            use_acceleration: Include acceleration (second derivative) in fingerprint.
            use_curvature: Include path curvature in fingerprint.
            smoothing_window: Window for smoothing trajectory before derivatives.
            aggregation: How to aggregate ("statistics", "sequence", "histogram").
        """
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_curvature = use_curvature
        self.smoothing_window = smoothing_window
        self.aggregation = aggregation

    def smooth_trajectory(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Apply moving average smoothing to trajectory.

        Args:
            trajectory: Attention centroids (T, 2) where 2 is (x, y).

        Returns:
            Smoothed trajectory (T-window+1, 2).
        """
        if self.smoothing_window <= 1:
            return trajectory

        # Simple moving average
        kernel = torch.ones(self.smoothing_window) / self.smoothing_window
        kernel = kernel.to(trajectory.device)

        # Pad and convolve each dimension
        smoothed = []
        for dim in range(trajectory.shape[1]):
            signal = trajectory[:, dim]
            # Use 1D convolution
            padded = F.pad(signal, (self.smoothing_window // 2, self.smoothing_window // 2), mode="replicate")
            conv = F.conv1d(
                padded.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
            )
            smoothed.append(conv.squeeze())

        return torch.stack(smoothed, dim=1)[: trajectory.shape[0]]

    def compute_velocity(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity (first derivative) of trajectory.

        Args:
            trajectory: Centroids (T, 2).

        Returns:
            Velocity vectors (T-1, 2).
        """
        return trajectory[1:] - trajectory[:-1]

    def compute_acceleration(
        self,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute acceleration (second derivative).

        Args:
            velocity: Velocity vectors (T-1, 2).

        Returns:
            Acceleration vectors (T-2, 2).
        """
        return velocity[1:] - velocity[:-1]

    def compute_curvature(
        self,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> torch.Tensor:
        """Compute path curvature.

        Curvature = |v x a| / |v|^3
        High curvature = sharp turns, low curvature = straight paths.

        Args:
            velocity: Velocity vectors (T-1, 2).
            acceleration: Acceleration vectors (T-2, 2).

        Returns:
            Curvature values (T-2,).
        """
        # Use only overlapping frames
        v = velocity[:-1]  # (T-2, 2)
        a = acceleration  # (T-2, 2)

        # 2D cross product (scalar): v_x * a_y - v_y * a_x
        cross = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]

        # Speed
        speed = torch.norm(v, dim=1)

        # Curvature (with small epsilon to avoid division by zero)
        curvature = torch.abs(cross) / (speed**3 + 1e-8)

        return curvature

    def compute_fingerprint(
        self,
        centroids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute trajectory fingerprint from attention centroids.

        Args:
            centroids: Attention centroids (T, 2) from DINOv3Encoder.get_attention_centroids().

        Returns:
            Dict containing fingerprint components.
        """
        result = {}

        # Smooth trajectory
        trajectory = self.smooth_trajectory(centroids)
        result["trajectory"] = trajectory

        # Velocity
        if self.use_velocity and trajectory.shape[0] > 1:
            velocity = self.compute_velocity(trajectory)
            result["velocity"] = velocity

            # Speed (magnitude of velocity)
            speed = torch.norm(velocity, dim=1)
            result["speed"] = speed

            if self.aggregation == "statistics":
                result["mean_speed"] = speed.mean()
                result["std_speed"] = speed.std()
                result["max_speed"] = speed.max()

                # Direction statistics (using atan2 for angle)
                angles = torch.atan2(velocity[:, 1], velocity[:, 0])
                result["mean_direction"] = angles.mean()
                result["direction_variance"] = angles.var()

        # Acceleration
        if self.use_acceleration and "velocity" in result and result["velocity"].shape[0] > 1:
            acceleration = self.compute_acceleration(result["velocity"])
            result["acceleration"] = acceleration

            accel_mag = torch.norm(acceleration, dim=1)
            result["acceleration_magnitude"] = accel_mag

            if self.aggregation == "statistics":
                result["mean_acceleration"] = accel_mag.mean()
                result["std_acceleration"] = accel_mag.std()

        # Curvature
        if (
            self.use_curvature
            and "velocity" in result
            and "acceleration" in result
            and result["acceleration"].shape[0] > 0
        ):
            curvature = self.compute_curvature(result["velocity"], result["acceleration"])
            result["curvature"] = curvature

            if self.aggregation == "statistics":
                result["mean_curvature"] = curvature.mean()
                result["std_curvature"] = curvature.std()
                result["max_curvature"] = curvature.max()

        # Trajectory extent (bounding box of attention path)
        result["trajectory_extent"] = trajectory.max(dim=0).values - trajectory.min(dim=0).values

        # Total path length
        if "velocity" in result:
            result["total_path_length"] = torch.norm(result["velocity"], dim=1).sum()

        return result

    def fingerprint_to_vector(
        self,
        fingerprint: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert fingerprint to a single vector.

        Args:
            fingerprint: Output from compute_fingerprint.

        Returns:
            1D tensor for similarity computation.
        """
        components = []

        # Speed statistics
        if "mean_speed" in fingerprint:
            components.append(fingerprint["mean_speed"].unsqueeze(0))
        if "std_speed" in fingerprint:
            components.append(fingerprint["std_speed"].unsqueeze(0))
        if "max_speed" in fingerprint:
            components.append(fingerprint["max_speed"].unsqueeze(0))

        # Direction statistics
        if "mean_direction" in fingerprint:
            # Convert angle to sin/cos for better comparison
            angle = fingerprint["mean_direction"]
            components.append(torch.cos(angle).unsqueeze(0))
            components.append(torch.sin(angle).unsqueeze(0))
        if "direction_variance" in fingerprint:
            components.append(fingerprint["direction_variance"].unsqueeze(0))

        # Acceleration statistics
        if "mean_acceleration" in fingerprint:
            components.append(fingerprint["mean_acceleration"].unsqueeze(0))
        if "std_acceleration" in fingerprint:
            components.append(fingerprint["std_acceleration"].unsqueeze(0))

        # Curvature statistics
        if "mean_curvature" in fingerprint:
            components.append(fingerprint["mean_curvature"].unsqueeze(0))
        if "std_curvature" in fingerprint:
            components.append(fingerprint["std_curvature"].unsqueeze(0))
        if "max_curvature" in fingerprint:
            components.append(fingerprint["max_curvature"].unsqueeze(0))

        # Trajectory extent
        if "trajectory_extent" in fingerprint:
            components.append(fingerprint["trajectory_extent"])

        # Total path length
        if "total_path_length" in fingerprint:
            components.append(fingerprint["total_path_length"].unsqueeze(0))

        if not components:
            raise ValueError("No components to create fingerprint vector")

        return torch.cat(components)

    def compare(
        self,
        fp1: dict[str, torch.Tensor],
        fp2: dict[str, torch.Tensor],
        metric: str = "cosine",
    ) -> float:
        """Compare two trajectory fingerprints.

        Args:
            fp1, fp2: Fingerprints from compute_fingerprint.
            metric: Distance metric ("cosine", "euclidean").

        Returns:
            Similarity score (higher = more similar).
        """
        v1 = self.fingerprint_to_vector(fp1)
        v2 = self.fingerprint_to_vector(fp2)

        if metric == "cosine":
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        elif metric == "euclidean":
            return -torch.norm(v1 - v2).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")


class DTWTrajectoryMatcher:
    """Use Dynamic Time Warping to match trajectories of different lengths.

    DTW is particularly useful for comparing videos of different durations
    that show similar motion patterns at different speeds.
    """

    def __init__(
        self,
        normalize_trajectory: bool = True,
    ):
        """Initialize DTW matcher.

        Args:
            normalize_trajectory: Normalize trajectories to [0, 1] before matching.
        """
        self.normalize = normalize_trajectory

    def normalize_trajectory(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize trajectory to unit square [0, 1] x [0, 1].

        Args:
            trajectory: Centroids (T, 2).

        Returns:
            Normalized trajectory (T, 2).
        """
        min_vals = trajectory.min(dim=0).values
        max_vals = trajectory.max(dim=0).values
        range_vals = max_vals - min_vals
        range_vals = torch.clamp(range_vals, min=1e-8)  # Avoid division by zero
        return (trajectory - min_vals) / range_vals

    def dtw_distance(
        self,
        traj1: torch.Tensor,
        traj2: torch.Tensor,
    ) -> float:
        """Compute DTW distance between two trajectories.

        Args:
            traj1: First trajectory (T1, 2).
            traj2: Second trajectory (T2, 2).

        Returns:
            DTW distance (lower = more similar).
        """
        if self.normalize:
            traj1 = self.normalize_trajectory(traj1)
            traj2 = self.normalize_trajectory(traj2)

        n, m = traj1.shape[0], traj2.shape[0]

        # Cost matrix
        cost = torch.cdist(traj1, traj2)  # (n, m)

        # DTW matrix
        dtw = torch.full((n + 1, m + 1), float("inf"), device=traj1.device)
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(
                    dtw[i - 1, j].item(),
                    dtw[i, j - 1].item(),
                    dtw[i - 1, j - 1].item(),
                )

        return dtw[n, m].item()

    def compare(
        self,
        traj1: torch.Tensor,
        traj2: torch.Tensor,
    ) -> float:
        """Compare trajectories and return similarity score.

        Args:
            traj1, traj2: Trajectories to compare.

        Returns:
            Similarity score (higher = more similar).
        """
        distance = self.dtw_distance(traj1, traj2)
        # Convert distance to similarity
        return 1.0 / (1.0 + distance)
