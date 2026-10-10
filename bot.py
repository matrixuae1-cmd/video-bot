#!/usr/bin/env python3
"""
Free Social Media Marketing Video Bot
=====================================
URL / description  ->  scrape  ->  LLM script  ->  TTS + music + images  ->  MP4 (1080x1920)

Usage:
    export GROQ_API_KEY=...        # or GEMINI_API_KEY=...  (both have free tiers)
    python bot.py "https://example.com/product" --out output
    python bot.py "Eco-friendly bamboo water bottle, keeps drinks cold 24h, $25" --out output

Everything except the LLM call works with no API key. Free endpoints change over time,
so every external step has a fallback (see the notes at the bottom of each section).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
W, H, FPS = 1080, 1920, 24  # 9:16 vertical for TikTok / Reels / Shorts
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AndrewNeural")  # e.g. ar-AE-HamdanNeural
TTS_RATE = os.getenv("TTS_RATE", "+8%")
TARGET_SECONDS = int(os.getenv("TARGET_SECONDS", "28"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")


# --------------------------------------------------------------------------- #
# 1) SCRAPING
# --------------------------------------------------------------------------- #
@dataclass
class PageData:
    url: str = ""
    title: str = ""
    description: str = ""
    price: str = ""
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)

    def to_prompt_text(self, limit: int = 6000) -> str:
        parts = [
            f"URL: {self.url}",
            f"TITLE: {self.title}",
            f"DESCRIPTION: {self.description}",
            f"PRICE: {self.price}" if self.price else "",
            "HEADINGS: " + " | ".join(self.headings),
            "FEATURES/BULLETS:\n- " + "\n- ".join(self.bullets),
            "BODY TEXT:\n" + "\n".join(self.paragraphs),
        ]
        return "\n".join(p for p in parts if p.strip())[:limit]


def fetch_html(url: str) -> str:
    """requests first (fast, cheap). Playwright only if USE_PLAYWRIGHT=1 and page looks empty."""
    html = ""
    try:
        r = requests.get(
            url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=25
        )
        r.raise_for_status()
        html = r.text
    except requests.RequestException as e:
        log.warning("requests failed: %s", e)

    looks_empty = len(html) < 3000 or "enable javascript" in html.lower()[:6000]
    if looks_empty and os.getenv("USE_PLAYWRIGHT") == "1":
        log.info("Falling back to Playwright (JS-rendered page)")
        html = _fetch_playwright(url) or html
    if not html:
        raise RuntimeError(f"Could not fetch {url}")
    return html


def _fetch_playwright(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright  # pip install playwright && playwright install chromium
    except ImportError:
        log.warning("Playwright not installed")
        return ""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=45000)
        page.wait_for_timeout(1500)
        html = page.content()
        browser.close()
    return html


def _walk_jsonld(node):
    if isinstance(node, list):
        for n in node:
            yield from _walk_jsonld(n)
    elif isinstance(node, dict):
        yield node
        for v in node.values():
            if isinstance(v, (list, dict)):
                yield from _walk_jsonld(v)


def parse_page(url: str, html: str) -> PageData:
    soup = BeautifulSoup(html, "html.parser")
    data = PageData(url=url)

    def meta(*names: str) -> str:
        for n in names:
            tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""

    data.title = meta("og:title", "twitter:title") or (soup.title.get_text(strip=True) if soup.title else "")
    data.description = meta("og:description", "description", "twitter:description")

    # ---- images (collected BEFORE we strip header/nav so nothing is lost) ----
    imgs: list[str] = []
    for key in ("og:image", "twitter:image"):
        if meta(key):
            imgs.append(urljoin(url, meta(key)))

    # ---- structured product data (JSON-LD) ----
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            blob = json.loads(s.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _walk_jsonld(blob):
            t = node.get("@type")
            t = " ".join(t) if isinstance(t, list) else str(t or "")
            if "Product" in t:
                data.title = data.title or str(node.get("name", ""))
                data.description = data.description or str(node.get("description", ""))
                offers = node.get("offers")
                if isinstance(offers, list) and offers:
                    offers = offers[0]
                if isinstance(offers, dict) and offers.get("price"):
                    data.price = f"{offers.get('price')} {offers.get('priceCurrency', '')}".strip()
                im = node.get("image")
                for i in (im if isinstance(im, list) else [im]):
                    if isinstance(i, str):
                        imgs.append(urljoin(url, i))

    for tag in soup.find_all("img"):
        src = tag.get("data-src") or tag.get("data-original") or tag.get("src")
        srcset = tag.get("srcset") or tag.get("data-srcset")
        if srcset:  # pick the largest candidate
            try:
                cands = [c.strip().split() for c in srcset.split(",") if c.strip()]
                cands.sort(key=lambda c: float(c[1].rstrip("wx")) if len(c) > 1 else 0)
                src = cands[-1][0]
            except (ValueError, IndexError):
                pass
        if src and not src.startswith("data:"):
            imgs.append(urljoin(url, src))

    bad = ("logo", "icon", "sprite", "avatar", "favicon", "pixel", "tracking", "badge", "flag", ".svg", ".gif")
    seen = set()
    for u in imgs:
        if u in seen or any(b in u.lower() for b in bad):
            continue
        seen.add(u)
        data.image_urls.append(u)

    # ---- text ----
    for t in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "svg", "iframe"]):
        t.decompose()
    data.headings = _uniq([h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])])[:15]
    data.paragraphs = _uniq(
        [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) >= 40]
    )[:20]
    data.bullets = _uniq(
        [li.get_text(" ", strip=True) for li in soup.find_all("li") if 20 <= len(li.get_text(strip=True)) <= 200]
    )[:20]
    return data


def _uniq(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def download_images(urls: list[str], dest: Path, referer: str, max_n: int = 8, min_side: int = 500) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    hashes = set()
    for u in urls:
        if len(saved) >= max_n:
            break
        try:
            r = requests.get(u, headers={"User-Agent": UA, "Referer": referer}, timeout=20)
            r.raise_for_status()
            if len(r.content) > 12_000_000:
                continue
            img = Image.open(io.BytesIO(r.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:  # noqa: BLE001
            log.debug("skip image %s: %s", u, e)
            continue
        w, h = img.size
        if min(w, h) < min_side or max(w, h) / min(w, h) > 3:
            continue
        sig = hashlib.md5(img.resize((8, 8)).tobytes()).hexdigest()  # near-duplicate filter
        if sig in hashes:
            continue
        hashes.add(sig)
        p = dest / f"scraped_{len(saved):02d}.jpg"
        img.save(p, quality=92)
        saved.append(p)
    log.info("Downloaded %d usable images", len(saved))
    return saved


# --------------------------------------------------------------------------- #
# 2) LLM: ANALYSIS + SCRIPT (Groq or Gemini, both free tiers, plain REST)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a senior short-form video marketer (TikTok, Reels, Shorts).
Apply what reliably performs on these platforms:
- Hook in the first 2 seconds: a bold claim, a question, a pain point, or a surprising number.
- One idea per scene, fast pacing (3-5 seconds per scene), conversational spoken language.
- Benefits over features; use concrete details from the source data only. Never invent facts, prices or claims.
- Finish with one clear call to action.
- On-screen text is punchy (max 6 words). Narration is natural speech, no emojis, no hashtags.
Return ONLY valid JSON, no markdown."""

JSON_SPEC = """{
  "title": "internal video title",
  "music_mood": "upbeat | chill | dramatic | luxury",
  "scenes": [
    {
      "narration": "spoken line for this scene",
      "on_screen_text": "max 6 words headline",
      "visual_source": "scraped | generated",
      "image_prompt": "vivid description of the ideal visual (used if generated)"
    }
  ],
  "post_caption": "caption for the social post, 1-2 sentences",
  "hashtags": ["#tag1", "#tag2"]
}"""


def build_user_prompt(page: PageData, n_images: int) -> str:
    words = int(TARGET_SECONDS * 2.6)  # ~2.6 spoken words per second
    return f"""Create a {TARGET_SECONDS}-second vertical marketing video script for this product.

SOURCE DATA:
{page.to_prompt_text()}

CONSTRAINTS:
- 5 to 7 scenes. Scene 1 = hook. Last scene = call to action.
- Total narration about {words} words.
- {n_images} real product images are available. Use visual_source "scraped" for scenes that should show the real product,
  and "generated" for mood / lifestyle / abstract scenes. Prefer "scraped" when images are available.
- Output JSON matching exactly this shape:
{JSON_SPEC}"""


def call_llm(user_prompt: str) -> str:
    groq, gemini = os.getenv("GROQ_API_KEY"), os.getenv("GEMINI_API_KEY")
    if groq:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.8,
                "response_format": {"type": "json_object"},
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if gemini:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": gemini},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.8, "responseMimeType": "application/json"},
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY (both are free).")


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def generate_script(page: PageData, n_images: int) -> dict:
    prompt = build_user_prompt(page, n_images)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            script = extract_json(call_llm(prompt))
            scenes = [s for s in script.get("scenes", []) if s.get("narration")]
            if not 3 <= len(scenes) <= 10:
                raise ValueError(f"bad scene count: {len(scenes)}")
            script["scenes"] = scenes
            script.setdefault("music_mood", "upbeat")
            script.setdefault("hashtags", [])
            return script
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("LLM attempt %d failed: %s", attempt + 1, e)
    raise RuntimeError(f"Script generation failed: {last_err}")


# --------------------------------------------------------------------------- #
# 3) ASSETS: voice (edge-tts), images (scraped / Pollinations / gradient), music (synth)
# --------------------------------------------------------------------------- #
def clean(s: str) -> str:
    """Strip emojis and odd symbols that break fonts / TTS."""
    s = re.sub(r"[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F]", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


async def _tts_all(items: list[tuple[str, Path]]) -> None:
    import edge_tts  # pip install edge-tts

    for text, path in items:
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(str(path))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("TTS retry %d: %s", attempt + 1, e)
                await asyncio.sleep(1.5)
        else:
            raise RuntimeError("edge-tts failed repeatedly")


def synthesize_voice(scenes: list[dict], out_dir: Path) -> list[Path]:
    items = [(clean(s["narration"]), out_dir / f"voice_{i:02d}.mp3") for i, s in enumerate(scenes)]
    asyncio.run(_tts_all(items))
    return [p for _, p in items]


def generate_image(prompt: str, dest: Path, seed: int) -> bool:
    """Free AI image via Pollinations (no key). Best-effort: returns False on any failure."""
    full = f"{prompt}, vertical composition, cinematic lighting, high quality, no text, no watermark"
    try:
        r = requests.get(
            f"https://image.pollinations.ai/prompt/{quote(full)}",
            params={"width": W, "height": H, "nologo": "true", "seed": seed},
            timeout=120,
        )
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            Image.open(io.BytesIO(r.content)).convert("RGB").save(dest, quality=92)
            return True
    except Exception as e:  # noqa: BLE001
        log.warning("Image generation failed: %s", e)
    return False


def gradient_image(dest: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    c1, c2 = rng.integers(20, 140, 3), rng.integers(60, 220, 3)
    t = np.linspace(0, 1, H)[:, None, None]
    arr = (c1 * (1 - t) + c2 * t) * np.ones((H, W, 3))
    Image.fromarray(arr.astype("uint8")).save(dest, quality=92)


def assign_images(scenes: list[dict], scraped: list[Path], out_dir: Path) -> list[Path]:
    """Pick a visual for each scene: scraped -> AI-generated -> gradient (always succeeds)."""
    result: list[Path] = []
    cursor = 0
    for i, sc in enumerate(scenes):
        path: Path | None = None
        wants_scraped = sc.get("visual_source", "scraped") == "scraped"
        if wants_scraped and scraped:
            path = scraped[cursor % len(scraped)]
            cursor += 1
        if path is None:
            gen = out_dir / f"generated_{i:02d}.jpg"
            if generate_image(clean(sc.get("image_prompt") or sc["on_screen_text"]), gen, seed=i + 7):
                path = gen
        if path is None and scraped:
            path = scraped[cursor % len(scraped)]
            cursor += 1
        if path is None:
            path = out_dir / f"gradient_{i:02d}.jpg"
            gradient_image(path, seed=i)
        result.append(path)
    return result


# ---- procedural background music: no download, no licence problems ----
MOODS = {  # bpm, chord progression (MIDI notes), kick, hats
    "upbeat": (118, [(60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)], True, True),
    "chill": (88, [(57, 60, 64), (53, 57, 60), (60, 64, 67), (55, 59, 62)], False, True),
    "dramatic": (100, [(57, 60, 64), (53, 57, 60), (60, 64, 67), (52, 56, 59)], True, False),
    "luxury": (80, [(60, 64, 67, 71), (57, 60, 64, 67), (53, 57, 60, 64), (55, 59, 62, 65)], False, False),
}


def _hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _add(buf: np.ndarray, start: int, sig: np.ndarray) -> None:
    end = min(len(buf), start + len(sig))
    if start < end:
        buf[start:end] += sig[: end - start]


def synth_music(duration: float, mood: str, path: Path) -> Path:
    bpm, prog, kick, hats = MOODS.get(mood, MOODS["upbeat"])
    sr = 44100
    n = int(duration * sr)
    t = np.arange(n) / sr
    beat, bar = 60 / bpm, 240 / bpm
    out = np.zeros(n)
    rng = np.random.default_rng(7)

    for i in range(int(duration / bar) + 1):
        s = int(i * bar * sr)
        e = min(n, int((i + 1) * bar * sr))
        if s >= n:
            break
        tt = t[s:e] - t[s]
        env = np.clip(np.minimum(tt / 0.4, (bar - tt) / 0.4), 0, 1)
        chord = prog[i % len(prog)]
        pad = sum(np.sin(2 * np.pi * _hz(m) * tt) + 0.3 * np.sin(4 * np.pi * _hz(m) * tt) for m in chord)
        pad = pad / len(chord) * 0.5 + 0.35 * np.sin(2 * np.pi * _hz(chord[0] - 24) * tt)
        out[s:e] += pad * env * 0.35

    kt = np.arange(int(0.25 * sr)) / sr
    kick_sig = np.sin(2 * np.pi * (48 * kt + 3 * (1 - np.exp(-30 * kt)) * 30)) * np.exp(-kt * 12) * 0.6
    ht = np.arange(int(0.06 * sr)) / sr
    for k in range(int(duration / beat) + 1):
        pos = int(k * beat * sr)
        if kick:
            _add(out, pos, kick_sig)
        if hats:
            _add(out, pos + int(beat * sr / 2), rng.standard_normal(len(ht)) * np.exp(-ht * 70) * 0.10)

    fade_in, fade_out = min(int(0.5 * sr), n), min(int(1.5 * sr), n)
    out[:fade_in] *= np.linspace(0, 1, fade_in)
    out[-fade_out:] *= np.linspace(1, 0, fade_out)
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.8
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((out * 32767).astype(np.int16).tobytes())
    return path


# --------------------------------------------------------------------------- #
# 4) VIDEO: Pillow renders frames/captions, MoviePy 2.x assembles + encodes
# --------------------------------------------------------------------------- #
def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        os.getenv("FONT_PATH"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def render_text(text: str, size: int, color=(255, 255, 255), max_w: int = 940, box: bool = False) -> np.ndarray:
    """Render wrapped, centered, outlined text to an RGBA numpy array (no ImageMagick needed)."""
    font = load_font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if probe.textlength(trial, font=font) <= max_w - 60 or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    line_h = int(size * 1.25)
    pad = 28
    img = Image.new("RGBA", (max_w, line_h * len(lines) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if box:
        d.rounded_rectangle([0, 0, max_w - 1, img.height - 1], radius=36, fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        x = (max_w - d.textlength(line, font=font)) / 2
        d.text((x, pad + i * line_h), line, font=font, fill=color, stroke_width=max(3, size // 14), stroke_fill=(0, 0, 0, 255))
    return np.array(img)


def compose_background(src: Image.Image) -> Image.Image:
    """Portrait images fill the frame; landscape/square get a blurred backdrop + sharp centered product."""
    src = ImageOps.exif_transpose(src).convert("RGB")
    cover = ImageOps.fit(src, (W, H), method=Image.LANCZOS)
    if src.width / src.height < 0.8:
        frame = cover
    else:
        frame = cover.filter(ImageFilter.GaussianBlur(45))
        frame = Image.blend(frame, Image.new("RGB", (W, H), (0, 0, 0)), 0.35)
        fg_w = int(W * 0.94)
        fg = src.resize((fg_w, int(src.height * fg_w / src.width)), Image.LANCZOS)
        if fg.height > H * 0.6:
            fg = src.resize((int(src.width * H * 0.6 / src.height), int(H * 0.6)), Image.LANCZOS)
        frame.paste(fg, ((W - fg.width) // 2, int(H * 0.40 - fg.height / 2)))
    # darken the lower part for caption legibility
    alpha = np.clip((np.linspace(0, 1, H) - 0.45) / 0.55, 0, 1) ** 1.3 * 0.65
    arr = np.array(frame, dtype=np.float32) * (1 - alpha[:, None, None])
    return Image.fromarray(arr.astype("uint8"))


def chunk_words(text: str, n: int = 3) -> list[str]:
    w = text.split()
    return [" ".join(w[i : i + n]) for i in range(0, len(w), n)]


def build_video(script: dict, images: list[Path], voices: list[Path], music: Path | None, out_path: Path) -> Path:
    from moviepy import (  # MoviePy >= 2.0 API
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        concatenate_videoclips,
    )

    scenes = script["scenes"]
    voice_clips = [AudioFileClip(str(v)) for v in voices]
    durations = [max(vc.duration + 0.45, 2.5) for vc in voice_clips]
    total = sum(durations)

    scene_clips, audio_layers, t0 = [], [], 0.0
    for i, (sc, img_path, vc, dur) in enumerate(zip(scenes, images, voice_clips, durations)):
        with Image.open(img_path) as im:
            frame = np.array(compose_background(im))
        layers = [
            ImageClip(frame)
            .with_duration(dur)
            .resized(lambda t, d=dur: 1.0 + 0.06 * t / d)  # slow Ken Burns zoom
            .with_position("center")
        ]
        headline = clean(sc.get("on_screen_text", ""))
        if headline:
            layers.append(
                ImageClip(render_text(headline.upper(), 72, (255, 255, 255), box=True))
                .with_start(0.1)
                .with_duration(min(dur - 0.1, 3.2))
                .with_position(("center", int(H * 0.09)))
            )
        # word-chunk captions timed proportionally to text length
        chunks = chunk_words(clean(sc["narration"]).upper(), 3)
        weights = np.array([len(c) + 2 for c in chunks], dtype=float)
        speech = dur - 0.45
        starts = 0.15 + np.concatenate([[0], np.cumsum(weights)[:-1]]) / weights.sum() * speech
        for chunk, st, wgt in zip(chunks, starts, weights):
            cd = wgt / weights.sum() * speech
            layers.append(
                ImageClip(render_text(chunk, 88, (255, 230, 0)))
                .with_start(float(st))
                .with_duration(float(cd))
                .with_position(("center", int(H * 0.60)))  # inside TikTok/Reels safe zone
            )
        scene_clips.append(CompositeVideoClip(layers, size=(W, H)).with_duration(dur))
        audio_layers.append(vc.with_start(t0 + 0.15))
        t0 += dur

    video = concatenate_videoclips(scene_clips, method="compose")
    mix = list(audio_layers)
    if music and music.exists():
        mix.insert(0, AudioFileClip(str(music)).with_volume_scaled(0.13))
    video = video.with_audio(CompositeAudioClip(mix).with_duration(total))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(out_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=2,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "21"],
        logger=None,
    )
    for c in voice_clips:
        c.close()
    return out_path


# --------------------------------------------------------------------------- #
# 5) ORCHESTRATION
# --------------------------------------------------------------------------- #
def run_pipeline(source: str, out_dir: str = "output") -> tuple[Path, dict]:
    out = Path(out_dir)
    work = out / "assets"
    work.mkdir(parents=True, exist_ok=True)

    # 1. scrape
    if re.match(r"^https?://", source.strip()):
        log.info("Scraping %s", source)
        page = parse_page(source.strip(), fetch_html(source.strip()))
        scraped = download_images(page.image_urls, work, referer=source.strip())
    else:
        log.info("Using provided text description (no scraping)")
        page = PageData(title=source[:70], description=source, paragraphs=[source])
        scraped = []

    # 2. script
    log.info("Generating script with LLM")
    script = generate_script(page, len(scraped))
    (out / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. assets
    log.info("Synthesizing voiceover (%d scenes)", len(script["scenes"]))
    voices = synthesize_voice(script["scenes"], work)
    images = assign_images(script["scenes"], scraped, work)
    est_total = sum(max(_audio_len(v) + 0.45, 2.5) for v in voices)
    music = synth_music(est_total + 1, script.get("music_mood", "upbeat"), work / "music.wav")

    # 4. render
    log.info("Rendering video")
    video = build_video(script, images, voices, music, out / "marketing_video.mp4")
    caption = f"{script.get('post_caption', '')}\n\n{' '.join(script.get('hashtags', []))}".strip()
    (out / "caption.txt").write_text(caption, encoding="utf-8")
    log.info("Done -> %s", video)
    return video, script


def _audio_len(path: Path) -> float:
    from moviepy import AudioFileClip

    with AudioFileClip(str(path)) as c:
        return c.duration


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Free marketing-video bot")
    ap.add_argument("source", help="Product URL or a plain-text product description")
    ap.add_argument("--out", default="output", help="Output directory")
    args = ap.parse_args()
    run_pipeline(args.source, args.out)
