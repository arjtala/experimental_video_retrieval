"""Video loading and frame extraction utilities."""

from pathlib import Path

import av
import numpy as np
import torch


def load_video(
    video_path: str | Path,
    max_frames: int | None = None,
    sample_rate: int = 1,
    max_resolution: int | None = 720,
) -> tuple[list[np.ndarray], float]:
    """Load video frames.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract (None = all).
        sample_rate: Sample every Nth frame.
        max_resolution: Maximum height (preserves aspect ratio). None = original.

    Returns:
        Tuple of (list of RGB frames as numpy arrays, fps).
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)

    frames = []
    frame_count = 0

    for frame in container.decode(video=0):
        if frame_count % sample_rate == 0:
            img = frame.to_ndarray(format="rgb24")

            # Resize if needed
            if max_resolution and img.shape[0] > max_resolution:
                scale = max_resolution / img.shape[0]
                new_h = max_resolution
                new_w = int(img.shape[1] * scale)
                import cv2
                img = cv2.resize(img, (new_w, new_h))

            frames.append(img)

            if max_frames and len(frames) >= max_frames:
                break

        frame_count += 1

    container.close()
    return frames, fps


def extract_frames(
    video_path: str | Path,
    frame_indices: list[int] | None = None,
    sample_rate: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Extract specific frames from a video.

    Args:
        video_path: Path to video file.
        frame_indices: Specific frame indices to extract (None = use sample_rate).
        sample_rate: Sample every Nth frame (ignored if frame_indices provided).
        max_frames: Maximum frames to return.

    Returns:
        List of RGB frames as numpy arrays.
    """
    container = av.open(str(video_path))

    if frame_indices:
        frame_set = set(frame_indices)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_set:
                frames.append(frame.to_ndarray(format="rgb24"))
                if max_frames and len(frames) >= max_frames:
                    break
    else:
        frames, _ = load_video(video_path, max_frames=max_frames, sample_rate=sample_rate)

    container.close()
    return frames


def frames_to_tensor(
    frames: list[np.ndarray],
    normalize: bool = True,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Convert frames to a batched tensor.

    Args:
        frames: List of RGB numpy arrays (H, W, 3).
        normalize: Apply ImageNet normalization.
        device: Target device.

    Returns:
        Tensor of shape (N, 3, H, W).
    """
    tensors = []
    for frame in frames:
        # HWC -> CHW, normalize to [0, 1]
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    batch = torch.stack(tensors).to(device)

    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        batch = (batch - mean) / std

    return batch
