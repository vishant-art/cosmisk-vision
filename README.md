# Cosmisk Vision

**Turn any video into structured creative intelligence.** Drop a competitor ad, UGC, or viral Reel — get back hook breakdowns, beat-by-beat structure, and actionable insights.

Built for D2C brands running Meta Ads at scale.

```
/cosmisk-vision https://fb.watch/competitor-ad
```

## What It Does

1. **Downloads** any video (URL or local file)
2. **Detects scene changes** — not uniform sampling, actual visual changes
3. **Transcribes** — captions first (free), Whisper API fallback
4. **Extracts frames in parallel** — 8x faster than sequential
5. **Returns structured output:**
   - Hook breakdown with timestamps
   - On-screen text verbatim
   - Beat-by-beat structure
   - CTA placement and angle
   - Creative synthesis

## Why This Exists

Every D2C brand spending ₹10L+/month on Meta has the same problem:

You're running 50+ creatives. Some work. Most don't. But you don't know **why**.

So you scrub through competitor ads trying to reverse-engineer hooks. You watch your own top performers trying to figure out what clicked. You brief your creative team with "something like that Mamaearth ad."

**Cosmisk Vision turns that loop into structured output you can actually use.**

## Install

```bash
git clone https://github.com/vishant-art/cosmisk-vision ~/.claude/skills/cosmisk-vision
```

**Dependencies:**
```bash
brew install yt-dlp ffmpeg  # macOS
```

## Usage

```bash
# Analyze a competitor Meta ad
python3 ~/.claude/skills/cosmisk-vision/scripts/watch.py "https://fb.watch/xyz"

# Analyze your own UGC
python3 ~/.claude/skills/cosmisk-vision/scripts/watch.py ~/Downloads/ugc-draft.mp4

# Hindi content
python3 ~/.claude/skills/cosmisk-vision/scripts/watch.py <url> --language hi

# Focus on first 2 minutes
python3 ~/.claude/skills/cosmisk-vision/scripts/watch.py <url> --start 0:00 --end 2:00

# High-res frames for text-heavy ads
python3 ~/.claude/skills/cosmisk-vision/scripts/watch.py <url> --resolution 1024
```

## What Makes It Different

| Feature | Standard Tools | Cosmisk Vision |
|---------|---------------|----------------|
| Frame extraction | Uniform (wasteful) | **Scene-aware** |
| Speed | Sequential | **8x faster (parallel)** |
| Video length | ~30 min limit | **Unlimited (auto-chunks)** |
| Languages | English only | **Any language** |
| Output | Chat response | **Persistent library** |

## D2C Use Cases

**1. Competitor Ad Analysis**
Drop a competitor's top-spending Meta ad. Get hook, angle, CTA, beat structure. Brief your creative team with actual insights, not vibes.

**2. UGC Review**
Analyze creator submissions before publishing. Catch weak hooks before you spend.

**3. Creative Postmortem**
Why did that ad work? Extract the structure from winners. Replicate intentionally.

**4. Trend Research**
Analyze viral Reels/TikToks in your niche. Extract patterns across multiple videos.

## All Flags

```
--start / --end       Focus on a section (MM:SS or HH:MM:SS)
--max-frames          Frame budget (default: 80)
--resolution          Frame width in px (default: 512, use 1024 for text)
--language, -l        Language code (en, hi, es, fr, de, ja, zh, etc.)
--auto-chunk          Force chunking even for short videos
--chunk-duration      Chunk size in minutes (default: 25)
--whisper             Force Whisper backend (groq|openai)
--no-whisper          Disable Whisper fallback
```

## Output Structure

```
~/claude-watch/library/<slug>/
├── frames/           # Extracted frames (JPG)
├── transcript.json   # Timestamped transcript
├── manifest.json     # Full metadata
└── notes.md          # Claude's analysis
```

## API Keys (Optional)

Captions cover most videos for free. Whisper only needed when no captions exist.

Create `~/.config/claude-watch/.env`:
```
GROQ_API_KEY=gsk_...      # Preferred (fast, cheap)
OPENAI_API_KEY=sk-...     # Fallback
```

## Part of the Cosmisk Toolkit

Cosmisk is an AI intelligence layer for D2C brands running Meta Ads at scale.

- **Cosmisk Vision** — Creative analysis (this skill)
- **OOS Detection** — Catch ads running on out-of-stock products
- **Competitor Intel** — Meta Ad Library analysis
- **Discount Leakage** — Find leaked coupon codes

More at [smashed.agency](https://smashed.agency)

## License

MIT. Built on `yt-dlp`, `ffmpeg`, and Claude's multimodal capabilities.

---

**Built by [Vishant Jain](https://linkedin.com/in/vishant-jain-facebook-ads-specialist-roi-driven-ads)** — Meta Ads for D2C brands since 2020.
