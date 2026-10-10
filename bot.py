#!/usr/bin/env python3
"""
Free Social Media Marketing Video Bot  -  v2
============================================
Website link  ->  bot explores the site with a real browser (screenshots of every screen)
              ->  LLM writes the script (English, then adapted to Arabic)
              ->  voice (edge-tts) + music + captions
              ->  two vertical MP4 videos: marketing_video_en.mp4 and marketing_video_ar.mp4

Usage:
    export GROQ_API_KEY=...
    export DEMO_EMAIL=... DEMO_PASSWORD=...      # only if the site is behind a login (use a DEMO account!)
    python bot.py "https://example.com" --out output
    python bot.py "https://example.com" --use-my-images      # use your own screenshots from ./images
    python bot.py "Plain product description"                # no website: text only (+ ./images if present)
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
import shutil
import time
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:  # Arabic text shaping for captions (pip install arabic-reshaper python-bidi)
    import arabic_reshaper

    try:
        from bidi.algorithm import get_display
    except ImportError:
        from bidi import get_display
except ImportError:  # captions still render, but Arabic letters would not be joined
    arabic_reshaper = None
    get_display = None

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
W, H, FPS = 1080, 1920, 24  # 9:16 vertical for TikTok / Reels / Shorts
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TTS_RATE = os.getenv("TTS_RATE", "+8%")
TARGET_SECONDS = int(os.getenv("TARGET_SECONDS", "28"))

LANGS = {
    "en": {"name": "English", "voice": os.getenv("TTS_VOICE_EN", "en-US-AndrewNeural")},
    "ar": {"name": "Arabic", "voice": os.getenv("TTS_VOICE_AR", "ar-AE-HamdanNeural")},  # or ar-AE-FatimaNeural
}
CURRENT_LANG = "en"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")


def set_language(code: str) -> None:
    global CURRENT_LANG
    CURRENT_LANG = code


def shape(text: str) -> str:
    """Make Arabic text displayable in Pillow (joined letters + right-to-left order)."""
    if CURRENT_LANG == "ar" and arabic_reshaper and get_display:
        return get_display(arabic_reshaper.reshape(text))
    return text


# --------------------------------------------------------------------------- #
# 1) WEBSITE ANALYSIS
#    a) crawl_site(): real browser (Playwright) clicks through the site, screenshots every screen
#    b) parse_page(): simple HTML scraping (fallback / marketing sites)
# --------------------------------------------------------------------------- #
class LoginRequired(RuntimeError):
    pass


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
            ("HEADINGS: " + " | ".join(self.headings)) if self.headings else "",
            ("FEATURES/BULLETS:\n- " + "\n- ".join(self.bullets)) if self.bullets else "",
            ("BODY TEXT:\n" + "\n".join(self.paragraphs)) if self.paragraphs else "",
        ]
        return "\n".join(p for p in parts if p.strip())[:limit]


@dataclass
class Shot:
    name: str
    path: Path
    text: str
    unique: str = ""


# ---- JavaScript helpers that run inside the page ----
JS_NAV = r"""
() => {
  const out = [];
  const seen = new Set();
  for (const e of document.querySelectorAll('body *')) {
    const r = e.getBoundingClientRect();
    if (r.width < 30 || r.height < 14 || r.height > 70 || r.left > 260 || r.top < 40) continue;
    const cs = getComputedStyle(e);
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    const t = (e.innerText || '').trim();
    if (t.length < 3 || t.length > 28 || t.indexOf('\n') >= 0) continue;
    const clickable = cs.cursor === 'pointer' || e.tagName === 'A' || e.tagName === 'BUTTON' ||
                      e.getAttribute('role') === 'menuitem';
    if (!clickable || seen.has(t)) continue;
    seen.add(t);
    out.push({t: t, y: r.top});
  }
  out.sort((a, b) => a.y - b.y);
  return out.map(o => o.t);
}
"""

JS_FIND = r"""
(t) => {
  for (const e of document.querySelectorAll('body *')) {
    const r = e.getBoundingClientRect();
    if (r.width < 10 || r.height < 10 || r.left > 260 || e.children.length > 2) continue;
    if ((e.innerText || '').trim() === t) {
      return {x: r.left + r.width / 2, y: r.top + r.height / 2};
    }
  }
  return null;
}
"""

JS_LINKS = r"""
() => [...new Set([...document.querySelectorAll('a[href]')].map(a => a.href))]
"""

BLOCKED = re.compile(r"log ?out|sign ?out|delete|remove|clear|reset|danger|cancel", re.I)
EXTRA_CLICKS = [x.strip() for x in os.getenv("EXTRA_CLICKS", "Subscription,Pricing,Plans").split(",") if x.strip()]


def _login(page, email: str, password: str) -> bool:
    try:
        pw = page.locator("input[type=password]:visible").first
        em = page.locator(
            "input[type=email]:visible, input[name*=mail i]:visible, input[id*=mail i]:visible, "
            "input[placeholder*=mail i]:visible, input[type=text]:visible"
        ).first
        em.fill(email, timeout=8000)
        pw.fill(password, timeout=8000)
        clicked = False
        for loc in (
            page.get_by_role("button", name=re.compile(r"sign in|log ?in|login", re.I)).first,
            page.get_by_text(re.compile(r"sign in|log ?in", re.I)).first,
        ):
            if loc.count():
                loc.click(timeout=5000)
                clicked = True
                break
        if not clicked:
            pw.press("Enter")
        page.wait_for_timeout(4000)
        return page.locator("input[type=password]:visible").count() == 0
    except Exception as e:  # noqa: BLE001
        log.warning("Login attempt failed: %s", e)
        return False


def _click_left(page, text: str) -> bool:
    try:
        pos = page.evaluate(JS_FIND, text)
        if not pos:
            return False
        page.mouse.click(pos["x"], pos["y"])
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("click %s failed: %s", text, e)
        return False


def crawl_site(url: str, shots_dir: Path, email: str = "", password: str = "", max_nav: int = 14):
    """Open the site in a real browser, (log in), click every sidebar item and screenshot each screen."""
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    shots_dir.mkdir(parents=True, exist_ok=True)
    shots: list[Shot] = []
    meta = {"title": "", "description": ""}

    def snap(page, name: str) -> bool:
        path = shots_dir / f"shot_{len(shots):02d}.jpg"
        try:
            page.screenshot(path=str(path), type="jpeg", quality=90)
            text = (page.inner_text("body") or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("screenshot failed (%s): %s", name, e)
            return False
        if shots and text == shots[-1].text:
            return False  # nothing changed on screen
        shots.append(Shot(name=name, path=path, text=text))
        log.info("Captured screen %d: %s", len(shots), name)
        return True

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, user_agent=UA, locale="en-US"
        )
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=45000)
        except PWTimeout:
            log.info("Page never became idle, continuing anyway")
        page.wait_for_timeout(2000)

        try:
            meta["title"] = page.title()
            if page.locator('meta[name="description"]').count():
                meta["description"] = page.locator('meta[name="description"]').first.get_attribute("content") or ""
        except Exception:  # noqa: BLE001
            pass

        # ---- login wall? ----
        if page.locator("input[type=password]:visible").count():
            if not (email and password):
                browser.close()
                raise LoginRequired(
                    "This website is behind a login. Add DEMO_EMAIL and DEMO_PASSWORD as GitHub secrets "
                    "(use a DEMO account with dummy data), or run with mode 'my-images'."
                )
            log.info("Login form detected, signing in with the demo account")
            if not _login(page, email, password):
                browser.close()
                raise LoginRequired("Login failed. Check DEMO_EMAIL / DEMO_PASSWORD.")
            page.wait_for_timeout(1500)

        snap(page, "Home")

        # ---- sidebar / menu items ----
        names = [n for n in page.evaluate(JS_NAV) if not BLOCKED.search(n)][:max_nav]
        log.info("Menu items found: %s", names)
        for n in names:
            if _click_left(page, n):
                page.wait_for_timeout(1300)
                snap(page, n)
        for n in EXTRA_CLICKS:  # e.g. the pricing / subscription screen inside a settings sub-menu
            if _click_left(page, n):
                page.wait_for_timeout(1300)
                snap(page, n)

        # ---- ordinary websites: scroll the page and visit a few internal links ----
        if len(shots) < 4:
            log.info("Few menu screens found, capturing page sections and internal links")
            try:
                height = page.evaluate("document.documentElement.scrollHeight")
                for k, y in enumerate(range(900, min(int(height), 6000), 900), start=1):
                    page.evaluate(f"window.scrollTo(0, {y})")
                    page.wait_for_timeout(700)
                    snap(page, f"Section {k}")
                page.evaluate("window.scrollTo(0, 0)")
                host = urlparse(url).netloc
                skip = re.compile(r"login|signin|signup|register|privacy|terms|cart|checkout|logout", re.I)
                links = [
                    l for l in page.evaluate(JS_LINKS)
                    if urlparse(l).netloc == host and "#" not in l and not skip.search(l)
                    and l.rstrip("/") != url.rstrip("/")
                ][:5]
                for l in links:
                    try:
                        page.goto(l, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(1500)
                        snap(page, urlparse(l).path.strip("/").split("/")[-1].replace("-", " ") or "Page")
                    except Exception as e:  # noqa: BLE001
                        log.debug("link %s failed: %s", l, e)
            except Exception as e:  # noqa: BLE001
                log.warning("Fallback crawl error: %s", e)

        browser.close()

    if len(shots) < 2:
        raise RuntimeError("The crawler could only capture one screen.")
    return shots, meta


def finalize_shots(shots: list[Shot], k: int = 9) -> list[Shot]:
    """Work out what is unique on each screen (drops the repeated sidebar/header text) and keep the best k."""
    n = len(shots)
    line_sets = [{l.strip() for l in s.text.splitlines() if l.strip()} for s in shots]
    freq = Counter(l for ls in line_sets for l in ls)
    thresh = max(3, int(n * 0.5))
    for s in shots:
        seen: dict[str, None] = {}
        for l in s.text.splitlines():
            l = l.strip()
            if 2 < len(l) < 120 and freq[l] < thresh:
                seen[l] = None
        s.unique = " | ".join(seen)[:380]

    if n <= k:
        return shots
    pricing = [i for i, s in enumerate(shots) if re.search(r"pric|plan|subscri", s.name, re.I)]
    keep = {0, *pricing}
    for i in sorted(range(n), key=lambda i: -len(shots[i].unique)):
        if len(keep) >= k:
            break
        keep.add(i)
    return [shots[i] for i in sorted(keep)]


def fetch_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=25)
    r.raise_for_status()
    return r.text


def _walk_jsonld(node):
    if isinstance(node, list):
        for n in node:
            yield from _walk_jsonld(n)
    elif isinstance(node, dict):
        yield node
        for v in node.values():
            if isinstance(v, (list, dict)):
                yield from _walk_jsonld(v)


def _uniq(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def parse_page(url: str, html: str) -> PageData:
    """Simple HTML scraping (used as a fallback when the browser crawl is not possible)."""
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
    imgs: list[str] = []
    for key in ("og:image", "twitter:image"):
        if meta(key):
            imgs.append(urljoin(url, meta(key)))
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
                for i in im if isinstance(im, list) else [im]:
                    if isinstance(i, str):
                        imgs.append(urljoin(url, i))
    for tag in soup.find_all("img"):
        src = tag.get("data-src") or tag.get("data-original") or tag.get("src")
        srcset = tag.get("srcset") or tag.get("data-srcset")
        if srcset:
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
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
        except Exception as e:  # noqa: BLE001
            log.debug("skip image %s: %s", u, e)
            continue
        w, h = img.size
        if min(w, h) < min_side or max(w, h) / min(w, h) > 3:
            continue
        sig = hashlib.md5(img.resize((8, 8)).tobytes()).hexdigest()
        if sig in hashes:
            continue
        hashes.add(sig)
        p = dest / f"scraped_{len(saved):02d}.jpg"
        img.save(p, quality=92)
        saved.append(p)
    log.info("Downloaded %d usable images", len(saved))
    return saved


def load_local_images(folder: Path, dest: Path, min_side: int = 300) -> tuple[list[Path], list[str]]:
    """Owner-provided screenshots from the repo's images/ folder. The file NAME describes the image."""
    if not folder.is_dir():
        return [], []
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels: list[str] = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", f.name, e)
            continue
        if min(img.size) < min_side:
            continue
        p = dest / f"local_{len(paths):02d}.jpg"
        img.save(p, quality=95)
        paths.append(p)
        label = re.sub(r"^[\d\s._-]+", "", f.stem)
        labels.append(re.sub(r"[_-]+", " ", label).strip() or f.stem)
    log.info("Loaded %d local images from %s", len(paths), folder)
    return paths, labels


# --------------------------------------------------------------------------- #
# 2) LLM: SCRIPT (Groq or Gemini, free tiers) + Arabic adaptation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a senior short-form video marketer (TikTok, Reels, Shorts).
Apply what reliably performs on these platforms:
- Hook in the first 2 seconds: a bold claim, a question, a pain point, or a surprising number.
- One idea per scene, fast pacing (3-6 seconds per scene), conversational spoken language.
- Benefits over features; use concrete details from the source data only. Never invent facts, prices or claims.
- Finish with one clear call to action.
- On-screen text is punchy (max 6 words). Narration is natural speech, no emojis, no hashtags.
Return ONLY valid JSON, no markdown."""

ARABIC_SYSTEM_PROMPT = """You are a senior Arabic copywriter for short-form video ads in the UAE and Gulf region.
You write natural, energetic spoken Arabic (clear Modern Standard Arabic with a friendly tone that Gulf audiences
enjoy), without diacritics. Return ONLY valid JSON, no markdown."""

JSON_SPEC = """{
  "title": "internal video title",
  "music_mood": "upbeat | chill | dramatic | luxury",
  "scenes": [
    {
      "narration": "spoken line for this scene",
      "on_screen_text": "max 6 words headline",
      "visual_source": "scraped | generated",
      "image_index": 1,
      "image_prompt": "vivid description of the ideal visual (used if generated)"
    }
  ],
  "post_caption": "caption for the social post, 1-2 sentences",
  "hashtags": ["#tag1", "#tag2"]
}"""


def build_user_prompt(page: PageData, n_images: int, labels: list[str] | None = None) -> str:
    n_scenes = min(len(labels), 9) if labels else 0
    seconds = max(TARGET_SECONDS, int(n_scenes * 5.5)) if n_scenes else TARGET_SECONDS
    words = int(seconds * 2.6)  # ~2.6 spoken words per second
    label_block = ""
    if labels:
        listing = "\n".join(f"  {i + 1}. {name}" for i, name in enumerate(labels))
        label_block = f"""
AVAILABLE IMAGES (screenshots of the real product; each line describes exactly what the image shows):
{listing}

IMPORTANT RULES FOR THE SCREENSHOTS:
- Make exactly {n_scenes} scenes, one per image, in this order, so image_index = scene number. visual_source must be "scraped".
- The narration of each scene must describe the concrete details visible in THAT image
  (the screen name and the specific features, options, numbers and plan names in its description),
  in simple, exciting spoken language. Never describe anything that is not in the image description.
- Do NOT read out names of people or customer companies, or exact money amounts from the screens. Talk about the features.
- Scene 1 opens with a strong hook while showing image 1. The last scene is the call to action while showing the last image.
"""
        scene_rule = f"- Exactly {n_scenes} scenes (one per image)."
    else:
        scene_rule = "- 5 to 7 scenes. Scene 1 = hook. Last scene = call to action."
    return f"""Create a {seconds}-second vertical marketing video script for this product.

SOURCE DATA:
{page.to_prompt_text()}
{label_block}
CONSTRAINTS:
{scene_rule}
- Total narration about {words} words.
- {n_images} real product images are available. Use visual_source "scraped" for scenes that should show the real product,
  and "generated" for mood / lifestyle / abstract scenes. Prefer "scraped" when images are available.
- Output JSON matching exactly this shape:
{JSON_SPEC}"""


def call_llm(user_prompt: str, system: str = SYSTEM_PROMPT) -> str:
    groq, gemini = os.getenv("GROQ_API_KEY"), os.getenv("GEMINI_API_KEY")
    if groq:
        for attempt in range(3):
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                    "temperature": 0.8,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            if r.status_code == 429:  # free-tier rate limit: wait and retry
                time.sleep(12 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError("Groq rate limit, try again in a minute")
    if gemini:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": gemini},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.8, "responseMimeType": "application/json"},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("Set GROQ_API_KEY or GEMINI_API_KEY (both are free).")


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


def _llm_json(prompt: str, system: str, validate) -> dict:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            data = extract_json(call_llm(prompt, system))
            validate(data)
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("LLM attempt %d failed: %s", attempt + 1, e)
            time.sleep(4)
    raise RuntimeError(f"LLM failed: {last_err}")


def generate_script(page: PageData, n_images: int, labels: list[str] | None = None) -> dict:
    def validate(d: dict) -> None:
        d["scenes"] = [s for s in d.get("scenes", []) if s.get("narration")]
        if not 3 <= len(d["scenes"]) <= 10:
            raise ValueError(f"bad scene count: {len(d['scenes'])}")
        d.setdefault("music_mood", "upbeat")
        d.setdefault("hashtags", [])

    return _llm_json(build_user_prompt(page, n_images, labels), SYSTEM_PROMPT, validate)


def adapt_script(script: dict, lang: str) -> dict:
    """Turn the English script into another language while keeping scenes / images identical."""
    if lang == "en":
        return script
    n = len(script["scenes"])

    def validate(d: dict) -> None:
        if len(d.get("scenes", [])) != n or not all(s.get("narration") for s in d["scenes"]):
            raise ValueError("scene count changed")
        for src, dst in zip(script["scenes"], d["scenes"]):
            dst["visual_source"] = src.get("visual_source", "scraped")
            dst["image_index"] = src.get("image_index")
            dst["image_prompt"] = src.get("image_prompt", "")
        d["music_mood"] = script.get("music_mood", "upbeat")
        d.setdefault("hashtags", [])

    prompt = f"""Translate and adapt this short-form video ad script into Arabic.
Rules:
- Keep EXACTLY the same JSON structure, the same number of scenes ({n}) and the same order.
- narration: natural spoken Arabic, same meaning and energy, similar length. No diacritics, no emojis.
- on_screen_text: Arabic, maximum 5 words.
- Keep brand names (for example NEXA WORLD), website addresses, numbers and currency amounts in Latin characters exactly as they are.
- Keep image_index, visual_source, image_prompt (English) and music_mood unchanged.
- post_caption in Arabic; hashtags: mix of Arabic and English hashtags.
Return only the JSON.

SCRIPT:
{json.dumps(script, ensure_ascii=False)}"""
    return _llm_json(prompt, ARABIC_SYSTEM_PROMPT, validate)


# --------------------------------------------------------------------------- #
# 3) ASSETS: voice (edge-tts), images, music
# --------------------------------------------------------------------------- #
def clean(s: str) -> str:
    """Strip emojis and odd symbols that break fonts / TTS."""
    s = re.sub(r"[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F]", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


async def _tts_all(items: list[tuple[str, Path]], voice: str) -> None:
    import edge_tts  # pip install edge-tts

    for text, path in items:
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, voice, rate=TTS_RATE).save(str(path))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("TTS retry %d: %s", attempt + 1, e)
                await asyncio.sleep(1.5)
        else:
            raise RuntimeError("edge-tts failed repeatedly")


def synthesize_voice(scenes: list[dict], out_dir: Path, voice: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [(clean(s["narration"]), out_dir / f"voice_{i:02d}.mp3") for i, s in enumerate(scenes)]
    asyncio.run(_tts_all(items, voice))
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
    c1, c2 = rng.integers(15, 80, 3), rng.integers(70, 200, 3)
    t = np.linspace(0, 1, H)[:, None, None]
    arr = (c1 * (1 - t) + c2 * t) * np.ones((H, W, 3))
    img = Image.fromarray(arr.astype("uint8")).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for _ in range(7):
        r = int(rng.integers(150, 480))
        x, y = int(rng.integers(-100, W + 100)), int(rng.integers(0, H))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(rng.integers(8, 26))))
    img = Image.alpha_composite(img, overlay.filter(ImageFilter.GaussianBlur(6)))
    img.convert("RGB").save(dest, quality=92)


def assign_images(scenes: list[dict], scraped: list[Path], out_dir: Path) -> list[Path]:
    """Pick a visual per scene: matching image_index -> next image -> AI-generated -> gradient."""
    result: list[Path] = []
    cursor = 0
    for i, sc in enumerate(scenes):
        path: Path | None = None
        wants_scraped = sc.get("visual_source", "scraped") == "scraped"
        if scraped and wants_scraped:
            try:
                idx = int(sc.get("image_index"))
            except (TypeError, ValueError):
                idx = 0
            if 1 <= idx <= len(scraped):
                path = scraped[idx - 1]
            else:
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
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # also covers Arabic
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def render_text(text: str, size: int, color=(255, 255, 255), max_w: int = 940, box: bool = False) -> np.ndarray:
    """Render wrapped, centered, outlined text to an RGBA numpy array (handles Arabic via shape())."""
    font = load_font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if probe.textlength(shape(trial), font=font) <= max_w - 60 or not cur:
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
        vis = shape(line)
        x = (max_w - d.textlength(vis, font=font)) / 2
        d.text((x, pad + i * line_h), vis, font=font, fill=color, stroke_width=max(3, size // 14), stroke_fill=(0, 0, 0, 255))
    return np.array(img)


def compose_background(src: Image.Image) -> Image.Image:
    """Portrait images fill the frame; screenshots/landscape get a blurred backdrop + sharp rounded card."""
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
        x, y = (W - fg.width) // 2, int(H * 0.40 - fg.height / 2)
        mask = Image.new("L", fg.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, fg.width - 1, fg.height - 1], radius=32, fill=255)
        shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            [x, y + 20, x + fg.width, y + fg.height + 20], radius=32, fill=(0, 0, 0, 180)
        )
        frame = Image.alpha_composite(frame.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(26)))
        frame.paste(fg, (x, y), mask)
        frame = frame.convert("RGB")
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
    for sc, img_path, vc, dur in zip(scenes, images, voice_clips, durations):
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


def _audio_len(path: Path) -> float:
    from moviepy import AudioFileClip

    with AudioFileClip(str(path)) as c:
        return c.duration


# --------------------------------------------------------------------------- #
# 5) ORCHESTRATION
# --------------------------------------------------------------------------- #
def run_pipeline(
    source: str,
    out_dir: str = "output",
    images_dir: str = "images",
    langs: tuple[str, ...] | list[str] = ("en", "ar"),
    use_my_images: bool = False,
    email: str = "",
    password: str = "",
) -> dict[str, tuple[Path, dict]]:
    out = Path(out_dir)
    work = out / "assets"
    work.mkdir(parents=True, exist_ok=True)
    source = source.strip()
    is_url = bool(re.match(r"^https?://", source))

    images: list[Path] = []
    labels: list[str] = []
    page = PageData()

    # ---------- 1. get visuals + understand the product ----------
    if use_my_images:
        images, labels = load_local_images(Path(images_dir), work)
        if not images:
            raise RuntimeError(f"No usable images found in '{images_dir}'.")
        page = PageData(url=source if is_url else "", title=source[:70], description=source if not is_url else "")
        if is_url:
            try:
                p = parse_page(source, fetch_html(source))
                page.title, page.description, page.price = p.title, p.description, p.price
            except Exception as e:  # noqa: BLE001
                log.warning("Could not read page text: %s", e)
    elif is_url:
        try:
            log.info("Exploring %s with a real browser", source)
            shots, meta = crawl_site(source, work / "crawl", email, password)
            shots = finalize_shots(shots)
            screens = out / "screens"
            screens.mkdir(exist_ok=True)
            for i, s in enumerate(shots):
                shutil.copy(s.path, screens / f"{i + 1:02d}-{re.sub(r'[^a-z0-9]+', '-', s.name.lower()).strip('-')}.jpg")
            images = [s.path for s in shots]
            labels = [f"{s.name} screen. Visible content: {s.unique}" if s.unique else f"{s.name} screen" for s in shots]
            page = PageData(url=source, title=meta["title"], description=meta["description"])
        except LoginRequired:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Browser exploration failed (%s). Falling back to simple scraping.", e)
            page = parse_page(source, fetch_html(source))
            images = download_images(page.image_urls, work, referer=source)
            labels = []
    else:
        page = PageData(title=source[:70], description=source, paragraphs=[source])
        images, labels = load_local_images(Path(images_dir), work)  # optional own screenshots

    # ---------- 2. script (English first) ----------
    log.info("Generating script with LLM (%d images)", len(images))
    script_en = generate_script(page, len(images), labels or None)
    scene_images = assign_images(script_en["scenes"], images, work)

    # ---------- 3. one video per language ----------
    results: dict[str, tuple[Path, dict]] = {}
    for lang in langs:
        if lang not in LANGS:
            log.warning("Unknown language %s, skipping", lang)
            continue
        log.info("=== Building %s video ===", LANGS[lang]["name"])
        set_language(lang)
        script = adapt_script(script_en, lang)
        ldir = work / lang
        voices = synthesize_voice(script["scenes"], ldir, LANGS[lang]["voice"])
        est_total = sum(max(_audio_len(v) + 0.45, 2.5) for v in voices)
        music = synth_music(est_total + 1, script.get("music_mood", "upbeat"), ldir / "music.wav")
        video = build_video(script, scene_images, voices, music, out / f"marketing_video_{lang}.mp4")
        (out / f"script_{lang}.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
        caption = f"{script.get('post_caption', '')}\n\n{' '.join(script.get('hashtags', []))}".strip()
        (out / f"caption_{lang}.txt").write_text(caption, encoding="utf-8")
        results[lang] = (video, script)
        log.info("Done -> %s", video)
    set_language("en")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Free marketing-video bot (English + Arabic)")
    ap.add_argument("source", help="Website URL or a plain-text product description")
    ap.add_argument("--out", default="output", help="Output directory")
    ap.add_argument("--images", default="images", help="Folder with your own screenshots/logo")
    ap.add_argument("--langs", default="en,ar", help="Comma separated: en,ar")
    ap.add_argument("--use-my-images", action="store_true", help="Use the screenshots from --images instead of exploring the site")
    args = ap.parse_args()
    run_pipeline(
        args.source,
        args.out,
        args.images,
        [x.strip() for x in args.langs.split(",") if x.strip()],
        args.use_my_images,
        os.getenv("DEMO_EMAIL", ""),
        os.getenv("DEMO_PASSWORD", ""),
    )
