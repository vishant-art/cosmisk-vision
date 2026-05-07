"""Per-scene frame extraction via ffmpeg — parallel extraction."""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from scripts.scenes import Scene


def format_filename(index: int, t: float) -> str:
    """`NNNN_tMM-SS.jpg`. MM may exceed 59 for videos > 1h — that's intentional
    so filenames sort naturally."""
    total = round(t)
    mm, ss = divmod(total, 60)
    return f"{index:04d}_t{mm:02d}-{ss:02d}.jpg"


def _extract_single_frame(
    video: Path,
    scene: Scene,
    index: int,
    out_dir: Path,
    width_px: int,
) -> dict:
    """Extract a single frame. Called by ThreadPoolExecutor."""
    name = format_filename(index, scene.t)
    out_path = out_dir / name
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-ss", f"{scene.t:.3f}",
        "-i", str(video),
        "-frames:v", "1",
        "-vf", f"scale={width_px}:-2",
        "-q:v", "3",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr_text = (
            e.stderr.decode("utf-8", errors="replace")
            if isinstance(e.stderr, (bytes, bytearray))
            else (e.stderr or "")
        )
        raise RuntimeError(
            f"ffmpeg failed extracting frame {index} at t={scene.t:.3f}s "
            f"(exit {e.returncode}): {stderr_text.strip()}"
        ) from e
    return {
        "index": index,
        "t": scene.t,
        "path": name,
        "kind": scene.kind,
    }


def extract_frames(
    video: Path,
    scenes: list[Scene],
    *,
    out_dir: Path,
    width_px: int = 512,
    max_workers: Optional[int] = None,
) -> list[dict]:
    """For each scene, extract a frame via ffmpeg — in parallel.

    Args:
        video: Path to the video file
        scenes: List of Scene objects with timestamps
        out_dir: Directory to write frames to
        width_px: Frame width (height auto-scaled)
        max_workers: Thread pool size (default: min(8, len(scenes)))

    Returns: [{"index": int, "t": float, "path": str (relative to out_dir), "kind": str}]
             Sorted by index (time order).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scenes:
        return []

    # Cap workers at 8 to avoid overwhelming the system
    workers = max_workers or min(8, len(scenes))

    results: list[dict] = []
    errors: list[Exception] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _extract_single_frame, video, scene, i, out_dir, width_px
            ): i
            for i, scene in enumerate(scenes, start=1)
        }

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                errors.append(e)

    if errors:
        # Re-raise first error (could collect all, but one is usually enough)
        raise errors[0]

    # Sort by index to maintain time order
    results.sort(key=lambda r: r["index"])
    return results
