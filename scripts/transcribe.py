"""Caption extraction (VTT) + dedupe + speaker-break heuristic + Whisper orchestration."""
from __future__ import annotations

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from scripts import whisper

_TS_RX = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
)

# Max audio file size for Whisper API (bytes) - leave buffer under 25MB
MAX_WHISPER_BYTES = 20 * 1024 * 1024  # 20MB to be safe
# Target chunk duration in seconds (at 64kbps mono = ~0.5MB/min)
CHUNK_DURATION_S = 1200  # 20 min chunks


def _ts_to_s(h: str | None, m: str, s: str, ms: str | None) -> float:
    return (int(h) if h else 0) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def parse_vtt(text: str) -> list[dict]:
    """Parse a WebVTT file into cues. Strips formatting tags like <c.colorXXXX>."""
    cues: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _TS_RX.search(lines[i])
        if not m:
            i += 1
            continue
        t_start = _ts_to_s(m.group(1), m.group(2), m.group(3), m.group(4))
        t_end = _ts_to_s(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
            i += 1
        joined = " ".join(t for t in text_lines if t).strip()
        if joined:
            cues.append({"t_start": t_start, "t_end": t_end, "text": joined})
    return cues


_ROLLING_OVERLAP_MIN = 10  # chars of suffix/prefix overlap to count as a rolling caption


def dedupe_cues(cues: list[dict]) -> list[dict]:
    """Collapse adjacent rolling/duplicate cues from VTT (especially YouTube auto-caps).

    Handles four patterns:
    1. Identical text → extend prev's t_end.
    2. `c.text` starts with `prev.text` (rolling extension) → replace prev's text with c's longer text.
    3. `prev.text` ends with `c.text` (c is a tail already shown) → extend prev's t_end, drop c.
    4. Suffix of prev matches prefix of c (>= 10 chars) → emit only the new tail of c.
    Otherwise keep both as separate cues.
    """
    out: list[dict] = []
    for c in cues:
        if not out:
            out.append(dict(c))
            continue
        prev = out[-1]
        # 1. Identical
        if prev["text"] == c["text"]:
            prev["t_end"] = max(prev["t_end"], c["t_end"])
            continue
        # 2. Rolling extension: c is the longer continuation of prev
        if len(c["text"]) > len(prev["text"]) and c["text"].startswith(prev["text"]):
            prev["t_end"] = c["t_end"]
            prev["text"] = c["text"]
            continue
        # 3. c is already contained at the tail of prev
        if prev["text"].endswith(c["text"]):
            prev["t_end"] = max(prev["t_end"], c["t_end"])
            continue
        # 4. Suffix of prev = prefix of c → emit only the new tail
        max_check = min(len(prev["text"]), len(c["text"]))
        overlap = 0
        for k in range(max_check, _ROLLING_OVERLAP_MIN - 1, -1):
            if prev["text"][-k:] == c["text"][:k]:
                overlap = k
                break
        if overlap > 0:
            tail = c["text"][overlap:].lstrip()
            if tail:
                out.append({"t_start": c["t_start"], "t_end": c["t_end"], "text": tail})
            else:
                prev["t_end"] = max(prev["t_end"], c["t_end"])
            continue
        # 5. Unrelated → keep both
        out.append(dict(c))
    return out


def insert_speaker_breaks(cues: list[dict], threshold_s: float = 2.0) -> list[dict]:
    """Mark cues that follow a pause > threshold as a speaker_break.

    This is a heuristic — it doesn't identify WHO is speaking, just that
    something changed. Useful for interviews/panels where downstream notes
    benefit from a section boundary.
    """
    out: list[dict] = []
    for i, c in enumerate(cues):
        marked = dict(c)
        if i > 0:
            gap = c["t_start"] - cues[i - 1]["t_end"]
            marked["speaker_break"] = gap > threshold_s
        else:
            marked["speaker_break"] = False
        out.append(marked)
    return out


def slice_to_window(
    cues: list[dict], start_s: Optional[float], end_s: Optional[float]
) -> list[dict]:
    """Keep only cues that overlap the [start, end] window. None bounds = open."""
    if start_s is None and end_s is None:
        return cues
    s = start_s if start_s is not None else float("-inf")
    e = end_s if end_s is not None else float("inf")
    return [c for c in cues if c["t_end"] >= s and c["t_start"] <= e]


def fetch_native_captions(
    video_url: str,
    work_dir: Path,
    language: str = "en",
) -> Optional[Path]:
    """Try to pull native + auto-generated subs via yt-dlp.

    Args:
        video_url: URL to fetch captions from
        work_dir: Directory to save caption files
        language: ISO 639-1 language code (e.g., 'en', 'hi', 'es', 'fr')
                  Use 'all' to download all available languages.

    Returns: Path to .vtt file or None if not found.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build language pattern - try requested language first, fall back to English
    if language == "all":
        sub_langs = "all"
    else:
        # Try requested language and its variants, fall back to English
        sub_langs = f"{language}.*,en.*" if language != "en" else "en.*"

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--sub-format", "vtt",
        "-o", str(work_dir / "%(id)s.%(ext)s"),
        video_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # Note: yt-dlp may return non-zero even when successful (e.g., warnings)
    # So we check for the file regardless of return code

    # Prefer exact language match, then any variant, then English fallback
    vtts = sorted(work_dir.glob("*.vtt"))
    if not vtts:
        return None

    # Try to find best match for requested language
    for vtt in vtts:
        if f".{language}." in vtt.name or vtt.name.endswith(f".{language}.vtt"):
            return vtt

    # Fall back to first available
    return vtts[0]


def get_audio_duration(audio: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_audio_for_whisper(
    video: Path,
    out_audio: Path,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
) -> None:
    """Mono 16kHz audio for Whisper. ~0.5MB/min.

    Args:
        video: Input video/audio file
        out_audio: Output audio path
        start_s: Start time in seconds (optional, for chunking)
        duration_s: Duration in seconds (optional, for chunking)
    """
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", str(video),
    ]
    if start_s is not None:
        cmd.extend(["-ss", f"{start_s:.3f}"])
    if duration_s is not None:
        cmd.extend(["-t", f"{duration_s:.3f}"])
    cmd.extend([
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(out_audio),
    ])
    subprocess.run(cmd, check=True)


def _transcribe_chunk(
    chunk_path: Path,
    offset_s: float,
    backend: str,
    groq_key: Optional[str],
    openai_key: Optional[str],
) -> list[dict]:
    """Transcribe a single audio chunk and adjust timestamps."""
    if backend == "groq":
        cues = whisper.transcribe_groq(chunk_path, api_key=groq_key)
    else:
        cues = whisper.transcribe_openai(chunk_path, api_key=openai_key)

    # Adjust timestamps by offset
    for cue in cues:
        cue["t_start"] += offset_s
        cue["t_end"] += offset_s

    return cues


def transcribe_via_whisper(
    audio: Path,
    *,
    backend: str,
    groq_key: Optional[str],
    openai_key: Optional[str],
    language: Optional[str] = None,
) -> list[dict]:
    """Transcribe audio via Whisper API, with automatic chunking for long files.

    If audio exceeds MAX_WHISPER_BYTES, it's split into chunks that are
    transcribed in parallel and merged with adjusted timestamps.
    """
    if backend == "groq":
        if not groq_key:
            raise whisper.WhisperError("Groq backend selected but GROQ_API_KEY is unset")
    elif backend == "openai":
        if not openai_key:
            raise whisper.WhisperError("OpenAI backend selected but OPENAI_API_KEY is unset")
    else:
        raise whisper.WhisperError(f"Unknown backend: {backend}")

    # Check if chunking is needed
    audio_size = audio.stat().st_size
    if audio_size <= MAX_WHISPER_BYTES:
        # Small enough to transcribe directly
        if backend == "groq":
            return whisper.transcribe_groq(audio, api_key=groq_key, language=language)
        return whisper.transcribe_openai(audio, api_key=openai_key, language=language)

    # Need to chunk the audio
    duration = get_audio_duration(audio)
    chunk_dir = audio.parent / "whisper_chunks"
    chunk_dir.mkdir(exist_ok=True)

    # Calculate chunk boundaries
    chunks: list[tuple[Path, float]] = []  # (chunk_path, offset_seconds)
    offset = 0.0
    chunk_idx = 0

    while offset < duration:
        chunk_path = chunk_dir / f"chunk_{chunk_idx:03d}.m4a"
        chunk_duration = min(CHUNK_DURATION_S, duration - offset)

        # Extract chunk
        extract_audio_for_whisper(
            audio.parent.parent / "source" / list((audio.parent.parent / "source").glob("video.*"))[0].name
            if not audio.name.startswith("chunk_") else audio,
            chunk_path,
            start_s=offset,
            duration_s=chunk_duration,
        )
        chunks.append((chunk_path, offset))
        offset += chunk_duration
        chunk_idx += 1

    # Transcribe chunks in parallel
    all_cues: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        futures = {
            pool.submit(
                _transcribe_chunk,
                chunk_path,
                chunk_offset,
                backend,
                groq_key,
                openai_key,
            ): chunk_offset
            for chunk_path, chunk_offset in chunks
        }

        results: list[tuple[float, list[dict]]] = []
        for future in as_completed(futures):
            offset = futures[future]
            try:
                cues = future.result()
                results.append((offset, cues))
            except Exception as e:
                # Continue with other chunks on failure
                import sys
                print(f"Chunk at {offset}s failed: {e}", file=sys.stderr)

        # Sort by offset and merge
        results.sort(key=lambda x: x[0])
        for _, cues in results:
            all_cues.extend(cues)

    # Clean up chunks
    for chunk_path, _ in chunks:
        chunk_path.unlink(missing_ok=True)
    chunk_dir.rmdir()

    # Sort by timestamp and dedupe any overlaps
    all_cues.sort(key=lambda c: c["t_start"])
    return dedupe_cues(all_cues)
