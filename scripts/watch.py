"""claude-watch-pro orchestrator — runs the full pipeline with parallel extraction and auto-chunking."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# When invoked as `python scripts/watch.py` the repo root is not automatically
# on sys.path.  Insert it so that `from scripts import …` works correctly
# regardless of how the script is launched.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import library as lib
from scripts import resolve as resolve_mod
from scripts import download as download_mod
from scripts import transcribe as transcribe_mod
from scripts import scenes as scenes_mod
from scripts import frames as frames_mod
from scripts import setup as setup_mod
from scripts import whisper

# Auto-chunk threshold: videos longer than this get auto-chunked
AUTO_CHUNK_THRESHOLD_S = 30 * 60  # 30 minutes
DEFAULT_CHUNK_DURATION_S = 25 * 60  # 25 minute chunks


def _parse_ts(s: str) -> float:
    """Accept SS, MM:SS, or HH:MM:SS."""
    parts = s.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"bad timestamp: {s}")


def _format_ts(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _focus_range(args) -> tuple[float, float] | None:
    if args.start is None and args.end is None:
        return None
    s = _parse_ts(args.start) if args.start else 0.0
    e = _parse_ts(args.end) if args.end else 1e12  # clamped later by duration
    return (s, e)


def _calculate_chunks(
    duration_s: float,
    chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S,
    overlap_s: float = 30.0,
) -> list[tuple[float, float]]:
    """Calculate chunk boundaries with overlap for long videos.

    Returns: [(start_s, end_s), ...] with overlap at boundaries.
    """
    if duration_s <= chunk_duration_s:
        return [(0.0, duration_s)]

    chunks = []
    start = 0.0
    while start < duration_s:
        end = min(start + chunk_duration_s, duration_s)
        chunks.append((start, end))
        # Move start forward, but keep overlap for context
        start = end - overlap_s if end < duration_s else duration_s

    return chunks


def process_single_video(
    args,
    focus: tuple[float, float] | None,
    meta: dict,
    work: Path,
    video: Path,
    env: dict,
    chunk_info: str = "",
) -> dict:
    """Process a single video (or chunk) and return results."""

    # ---- Transcribe ----
    transcript_path = work / "transcript.json"
    cached = lib.cache_lookup(lib.slug_for(meta), meta["source_hash"])

    if cached and transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
    else:
        transcript: list[dict] = []
        if meta["is_url"]:
            vtt = transcribe_mod.fetch_native_captions(
                meta["source"],
                work / "subs",
                language=args.language,
            )
            if vtt:
                transcript = transcribe_mod.dedupe_cues(
                    transcribe_mod.parse_vtt(vtt.read_text())
                )
                (work / "transcript.vtt").write_bytes(vtt.read_bytes())

        if not transcript and not args.no_whisper:
            backend = whisper.pick_backend(
                groq_key=env.get("GROQ_API_KEY"),
                openai_key=env.get("OPENAI_API_KEY"),
                forced=args.whisper,
            )
            if backend:
                audio = work / "audio.m4a"
                transcribe_mod.extract_audio_for_whisper(video, audio)
                try:
                    transcript = transcribe_mod.transcribe_via_whisper(
                        audio,
                        backend=backend,
                        groq_key=env.get("GROQ_API_KEY"),
                        openai_key=env.get("OPENAI_API_KEY"),
                        language=args.language,
                    )
                except whisper.WhisperError as e:
                    print(f"Whisper failed ({backend}): {e}", file=sys.stderr)

        transcript = transcribe_mod.insert_speaker_breaks(transcript)
        transcript_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))

    if focus:
        transcript_for_window = transcribe_mod.slice_to_window(
            transcript, start_s=focus[0], end_s=focus[1]
        )
    else:
        transcript_for_window = transcript

    # ---- Detect scenes ----
    scenes_path = work / "scenes.json"
    if cached and scenes_path.exists() and not focus:
        raw_scenes = [
            scenes_mod.Scene(t=s["t"], score=s["score"], kind=s["kind"])
            for s in json.loads(scenes_path.read_text())
        ]
    else:
        raw_scenes = scenes_mod.detect_scenes(video, threshold=args.scene_threshold)

    if focus:
        s0, s1 = focus
        s1 = min(s1, meta["duration_s"])
        raw_scenes = [s for s in raw_scenes if s0 <= s.t <= s1]
        if not raw_scenes or raw_scenes[0].t > s0:
            raw_scenes.insert(0, scenes_mod.Scene(t=s0, score=1.0, kind="detected"))
        max_gap = min(args.max_gap, 15.0)
        duration_for_floor = s1
    else:
        max_gap = args.max_gap
        duration_for_floor = meta["duration_s"]

    floored = scenes_mod.apply_coverage_floor(
        raw_scenes, duration_s=duration_for_floor, max_gap_s=max_gap
    )
    capped = scenes_mod.apply_budget_cap(floored, max_frames=args.max_frames)

    if not focus:
        scenes_path.write_text(json.dumps(
            [{"t": s.t, "score": s.score, "kind": s.kind} for s in capped],
            indent=2,
        ))

    # ---- Extract frames (PARALLEL) ----
    frames_dir = work / "frames"
    if frames_dir.exists():
        for f in frames_dir.iterdir():
            f.unlink()

    raw_frame_records = frames_mod.extract_frames(
        video, capped, out_dir=frames_dir, width_px=args.resolution
    )
    frame_records = [
        {**fr, "path": f"frames/{fr['path']}"} for fr in raw_frame_records
    ]

    return {
        "transcript": transcript,
        "transcript_for_window": transcript_for_window,
        "scenes": capped,
        "frame_records": frame_records,
        "work": work,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="watch-pro",
        description="claude-watch-pro: Enhanced video-to-notes with parallel extraction and auto-chunking",
    )
    p.add_argument("source", help="URL or local path")
    p.add_argument("--start", help="focus start (SS, MM:SS, or HH:MM:SS)")
    p.add_argument("--end", help="focus end (SS, MM:SS, or HH:MM:SS)")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--resolution", type=int, default=512, help="frame width in px")
    p.add_argument("--scene-threshold", type=float, default=0.30)
    p.add_argument("--max-gap", type=float, default=45.0, help="coverage floor seconds")
    p.add_argument("--whisper", choices=["groq", "openai"], help="force Whisper backend")
    p.add_argument("--no-whisper", action="store_true", help="disable Whisper fallback")
    p.add_argument("--out-dir", help="library root (default: ~/claude-watch/library)")

    # NEW: Language support
    p.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (ISO 639-1: en, hi, es, fr, de, etc.). Default: en",
    )

    # NEW: Auto-chunking controls
    p.add_argument(
        "--auto-chunk",
        action="store_true",
        help="Auto-split videos >30min into chunks (enabled by default for long videos)",
    )
    p.add_argument(
        "--no-auto-chunk",
        action="store_true",
        help="Disable auto-chunking even for long videos",
    )
    p.add_argument(
        "--chunk-duration",
        type=int,
        default=25,
        help="Chunk duration in minutes for auto-chunking (default: 25)",
    )

    args = p.parse_args(argv)

    if args.out_dir:
        lib.LIBRARY_ROOT = Path(args.out_dir).expanduser().resolve()
    lib.LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: resolve ----
    focus = _focus_range(args)
    meta = resolve_mod.resolve_source(args.source, focus_range=focus)
    meta["watched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = lib.slug_for(meta)
    work = lib.LIBRARY_ROOT / slug
    work.mkdir(parents=True, exist_ok=True)

    # Cache check
    cached = lib.cache_lookup(slug, meta["source_hash"])

    # ---- Stage 2: download ----
    src_dir = work / "source"
    if cached and list(src_dir.glob("video.*")):
        video = next(iter(src_dir.glob("video.*")))
    else:
        if meta["is_url"]:
            video = download_mod.download_video(meta["source"], src_dir, basename="video")
        else:
            video = download_mod.copy_local(Path(meta["source"]), src_dir, basename="video")

    # Read env for Whisper keys
    env = setup_mod._read_env()

    # ---- Determine if auto-chunking is needed ----
    duration_s = meta["duration_s"]
    should_chunk = (
        not args.no_auto_chunk
        and not focus  # Don't auto-chunk if user specified focus range
        and (args.auto_chunk or duration_s > AUTO_CHUNK_THRESHOLD_S)
    )

    if should_chunk and duration_s > AUTO_CHUNK_THRESHOLD_S:
        # Auto-chunk mode for long videos
        chunk_duration_s = args.chunk_duration * 60
        chunks = _calculate_chunks(duration_s, chunk_duration_s)

        print(f"=== claude-watch-pro: AUTO-CHUNKING ===", file=sys.stderr)
        print(f"Video duration: {_format_ts(duration_s)}", file=sys.stderr)
        print(f"Splitting into {len(chunks)} chunks of ~{args.chunk_duration} min each", file=sys.stderr)
        print(file=sys.stderr)

        all_frame_records = []
        all_transcript = []

        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            chunk_focus = (chunk_start, chunk_end)
            chunk_label = f"Chunk {i}/{len(chunks)} [{_format_ts(chunk_start)}–{_format_ts(chunk_end)}]"
            print(f"Processing {chunk_label}...", file=sys.stderr)

            # Create chunk-specific work directory
            chunk_work = work / f"chunk_{i:02d}"
            chunk_work.mkdir(exist_ok=True)

            # Process this chunk
            result = process_single_video(
                args,
                focus=chunk_focus,
                meta=meta,
                work=chunk_work,
                video=video,
                env=env,
                chunk_info=chunk_label,
            )

            # Collect results
            all_frame_records.extend(result["frame_records"])
            all_transcript.extend(result["transcript_for_window"])

        # Dedupe transcript (overlapping chunks may have duplicate cues)
        all_transcript = transcribe_mod.dedupe_cues(all_transcript)
        all_transcript = transcribe_mod.insert_speaker_breaks(all_transcript)

        # Write merged results
        transcript_path = work / "transcript.json"
        transcript_path.write_text(json.dumps(all_transcript, indent=2, ensure_ascii=False))

        frame_records = all_frame_records
        transcript = all_transcript
        transcript_for_window = all_transcript
        capped = []  # Scenes are per-chunk

    else:
        # Normal single-video processing
        result = process_single_video(
            args,
            focus=focus,
            meta=meta,
            work=work,
            video=video,
            env=env,
        )
        frame_records = result["frame_records"]
        transcript = result["transcript"]
        transcript_for_window = result["transcript_for_window"]
        capped = result["scenes"]

    # ---- Write final manifest ----
    transcript_window_path = work / "transcript.window.json"
    if focus:
        transcript_window_path.write_text(
            json.dumps(transcript_for_window, indent=2, ensure_ascii=False)
        )
        transcript_consumer_path = "transcript.window.json"
    else:
        transcript_consumer_path = "transcript.json"

    (work / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    lib.write_manifest(
        path=work / "manifest.json",
        meta=meta,
        scenes=[{"t": s.t, "score": s.score, "kind": s.kind} for s in capped] if capped else [],
        frames=frame_records,
        transcript_path=transcript_consumer_path,
        focus_range=focus,
    )

    # ---- Stdout: the contract Claude consumes ----
    duration_str = f"{int(meta['duration_s']) // 60:02d}:{int(meta['duration_s']) % 60:02d}"
    focus_str = "full" if focus is None else f"{args.start or '0:00'}–{args.end or 'end'}"
    transcript_kind = (
        "captions" if (work / "transcript.vtt").exists()
        else "whisper" if transcript
        else "none"
    )

    print("=== claude-watch-pro manifest ===")
    print(f"title: {meta['title']!r}")
    print(f"source: {meta['source']}")
    print(f"duration: {duration_str}")
    print(f"focus: {focus_str}")
    print(f"language: {args.language}")
    print(f"transcript_source: {transcript_kind}")
    if capped:
        print(f"scenes_detected: {sum(1 for s in capped if s.kind == 'detected')}")
    print(f"frames_extracted: {len(frame_records)}")
    if should_chunk and duration_s > AUTO_CHUNK_THRESHOLD_S:
        print(f"auto_chunked: yes ({len(chunks)} chunks)")
    print(f"library_dir: {work}")
    print()
    print("=== frames ===")
    for fr in frame_records:
        mm = int(fr["t"]) // 60
        ss = int(fr["t"]) % 60
        print(f"{fr['index']:04d}  t={mm:02d}:{ss:02d}  {fr['path']}  ({fr['kind']})")
    print()
    print("=== transcript ===")
    print(f"{work / transcript_consumer_path}  (load this — too long to inline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
