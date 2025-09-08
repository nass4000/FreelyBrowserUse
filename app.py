import os
import json
import time
import uuid
import queue
import threading
import re
import base64
import datetime
from html import unescape as html_unescape
from typing import Dict, List, Generator, Optional

from flask import Flask, Response, jsonify, render_template, request

# External deps
import requests

# Database
from db_models import init_db, SessionLocal, Session as DBSession, Message, Plan, SearchResult, ScrapedPage, HtmlCache

# Optional: YOLO for preview annotations
YOLO_AVAILABLE = False
YOLO_MODEL = None
YOLO_LOCK = threading.Lock()
YOLO_LAST_ERROR = None
# Ensure imaging libs are always present for overlays
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    # These are in requirements; if missing, overlays will silently no-op
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    np = None  # type: ignore
try:
    # Import YOLO; model loads lazily on first use
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Selenium / undetected chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


app = Flask(__name__, template_folder="templates", static_folder="static")


# -----------------------------
# Config: environment + file override + API
# -----------------------------
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")

def load_config() -> Dict:
    base = {
        # Shared defaults
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        # Planning
        "planning_model": os.getenv("OPENROUTER_PLANNING_MODEL", "google/gemini-2.5-flash"),
        "planning_base_url": os.getenv("PLANNING_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
        "planning_api_key": os.getenv("PLANNING_API_KEY", os.getenv("OPENROUTER_API_KEY", "")),
        # Answer
        "answer_model": os.getenv("OPENROUTER_ANSWER_MODEL", "deepseek/deepseek-reasoner"),
        "answer_base_url": os.getenv("ANSWER_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
        "answer_api_key": os.getenv("ANSWER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")),
        # Fallback
        "answer_model_fallback": os.getenv("OPENROUTER_ANSWER_MODEL_FALLBACK", "google/gemini-2.5-flash"),
        "answer_fallback_base_url": os.getenv("ANSWER_FALLBACK_BASE_URL", os.getenv("ANSWER_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))),
        "answer_fallback_api_key": os.getenv("ANSWER_FALLBACK_API_KEY", os.getenv("ANSWER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))),
        # Browser
        "browser_headless": _env_bool("BROWSER_HEADLESS", True),
        "browser_pageload_timeout": int(os.getenv("BROWSER_PAGELOAD_TIMEOUT", "25")),
        "browser_start_timeout": int(os.getenv("BROWSER_START_TIMEOUT", "30")),
        "search_results_per_query": int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5")),
        "pages_to_open": int(os.getenv("PAGES_TO_OPEN", "10")),
        # Binaries
        "chromedriver_path": os.getenv("CHROMEDRIVER_PATH", ""),
        "chrome_binary_path": os.getenv("CHROME_BINARY_PATH", ""),
        # Window sizing (used to fit live preview canvas)
        "browser_window_width": int(os.getenv("BROWSER_WINDOW_WIDTH", "1280")),
        "browser_window_height": int(os.getenv("BROWSER_WINDOW_HEIGHT", "800")),
        # Dev / advanced
        "dev_mode": _env_bool("DEV_MODE", False),
        "dev_show_prompts": _env_bool("DEV_SHOW_PROMPTS", True),
        "dev_show_requests": _env_bool("DEV_SHOW_REQUESTS", False),
        "dry_run": _env_bool("DRY_RUN", False),
        "browser_autostart": _env_bool("BROWSER_AUTOSTART", True),
        # LLM params
        "planning_temperature": float(os.getenv("PLANNING_TEMPERATURE", "0.2")),
        "planning_top_p": float(os.getenv("PLANNING_TOP_P", "1.0")),
        "answer_temperature": float(os.getenv("ANSWER_TEMPERATURE", "0.2")),
        "answer_top_p": float(os.getenv("ANSWER_TOP_P", "1.0")),
        "answer_max_tokens": int(os.getenv("ANSWER_MAX_TOKENS", "1200")),
        # Preview/replay
        "preview_enable_pan": _env_bool("PREVIEW_ENABLE_PAN", True),
        "preview_frames": int(os.getenv("PREVIEW_FRAMES", "24")),
        "preview_delay": float(os.getenv("PREVIEW_DELAY", "0.15")),
        # Preview YOLO overlay
        "preview_yolo_enable": _env_bool("PREVIEW_YOLO_ENABLE", False),
        "preview_yolo_model": os.getenv("PREVIEW_YOLO_MODEL", os.getenv("YOLO_MODEL", "yolov8n.pt")),
        "preview_yolo_conf": float(os.getenv("PREVIEW_YOLO_CONF", "0.25")),
        "preview_yolo_rich_panel": _env_bool("PREVIEW_YOLO_RICH_PANEL", True),
        "preview_yolo_topk": int(os.getenv("PREVIEW_YOLO_TOPK", "5")),
    }
    # Overlay with config file if present
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                file_conf = json.load(f)
                base.update({k: v for k, v in file_conf.items() if v is not None})
    except Exception:
        pass
    return base


def save_config(conf: Dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=2)
    except Exception:
        pass


CONFIG = load_config()

# Initialize database
try:
    init_db()
    load_html_cache_from_db()
    DB_AVAILABLE = True
except Exception as e:
    print(f"Database initialization failed: {e}. Falling back to in-memory storage.")
    DB_AVAILABLE = False

# -----------------------------
# YOLO overlay helper
# -----------------------------
def _get_yolo_model():
    global YOLO_MODEL, YOLO_LAST_ERROR
    if not YOLO_AVAILABLE:
        return None
    if YOLO_MODEL is not None:
        return YOLO_MODEL
    # Lazy load and cache
    with YOLO_LOCK:
        if YOLO_MODEL is None:
            try:
                model_path = CONFIG.get("preview_yolo_model") or "yolov8n.pt"
                # If path isn't a local file, try to fetch to bin/models
                if not os.path.isabs(model_path) or not os.path.exists(model_path):
                    resolved = _ensure_local_yolo_weights(os.path.basename(model_path))
                    if resolved:
                        model_path = resolved
                        CONFIG["preview_yolo_model"] = model_path
                        save_config(CONFIG)
                YOLO_MODEL = YOLO(model_path)
                YOLO_LAST_ERROR = None
            except Exception as e:
                YOLO_MODEL = None
                YOLO_LAST_ERROR = str(e)
        return YOLO_MODEL


def _pick_font(size_px: int):
    try:
        # Prefer a truetype font for visibility
        candidates = [
            os.path.join(os.getenv("WINDIR", "C:/Windows"), "Fonts", "segoeui.ttf"),
            os.path.join(os.getenv("WINDIR", "C:/Windows"), "Fonts", "arial.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in candidates:
            try:
                if os.path.exists(p):
                    return ImageFont.truetype(p, size_px)
            except Exception:
                continue
    except Exception:
        pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _yolo_annotate_b64(b64_png: str) -> str:
    """If enabled, draw YOLO boxes with class name only (no global status)."""
    try:
        if not CONFIG.get("preview_yolo_enable", False):
            return b64_png
        # Decode image for overlays regardless of model availability
        from io import BytesIO
        png_bytes = base64.b64decode(b64_png)
        if Image is None:
            return b64_png
        im = Image.open(BytesIO(png_bytes)).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)
        # Scale thickness and font to image size for visibility when downscaled in UI
        thickness = max(3, int(min(W, H) * 0.008))  # ~10 px on 1280x800
        font_size = max(14, int(min(W, H) * 0.035))  # ~28 px on 800 height
        font = _pick_font(font_size)

        # Optional: previously drew a debug border around the whole image.
        # Intentionally disabled to keep preview clean of extra labels/borders.

        model = _get_yolo_model()
        status_text = "YOLO: unavailable"
        n_boxes = 0
        if model is not None:
            # Predict
            conf = float(CONFIG.get("preview_yolo_conf", 0.25) or 0.25)
            res_list = model.predict(im, verbose=False, conf=conf)
            if res_list:
                res = res_list[0]
                boxes = getattr(res, "boxes", None)
                names = getattr(res, "names", {}) or {}
                if boxes is not None and len(boxes) > 0:
                    # Boxes in xyxy
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else None
                        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
                        n_boxes = int(xyxy.shape[0])
                        # Build detection list with richer info
                        detections = []
                        for i in range(n_boxes):
                            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                            w = max(1.0, x2 - x1)
                            h = max(1.0, y2 - y1)
                            area = (w * h) / max(1.0, W * H)
                            cid = int(cls[i]) if cls is not None else -1
                            cname = names.get(cid, str(cid))
                            cconf = float(confs[i]) if confs is not None else 0.0
                            detections.append({
                                "cls_id": cid,
                                "name": cname,
                                "conf": cconf,
                                "box": (x1, y1, x2, y2),
                                "area": area,
                                "size": (w, h),
                                "center": (x1 + w / 2.0, y1 + h / 2.0),
                            })

                        # Class color palette
                        def color_for(cid: int):
                            h = (hash(cid) % 360) / 360.0
                            # Convert HSV to RGB-ish by simple mapping
                            import colorsys
                            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
                            return (int(r * 255), int(g * 255), int(b * 255))

                        # Draw boxes with class-specific colors
                        for i, det in enumerate(detections):
                            x1, y1, x2, y2 = det["box"]
                            cname = det["name"]
                            color = color_for(det["cls_id"]) if det["cls_id"] is not None else (255, 50, 50)
                            for t in range(thickness):
                                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
                            if font:
                                # Show class name only (no index or confidence)
                                label = f"{cname}"
                                try:
                                    tw, th_text = draw.textsize(label, font=font)
                                except Exception:
                                    tw, th_text = (len(label) * font_size // 2, font_size)
                                y_text = max(0, y1 - th_text - 6)
                                draw.rectangle([x1, y_text, x1 + tw + 10, y_text + th_text + 6], fill=(0, 0, 0))
                                draw.text((x1 + 5, y_text + 3), label, fill=(255, 255, 255), font=font)

                        # Optional rich side panel with thumbnails and summary
                        if CONFIG.get("preview_yolo_rich_panel", True):
                            panel_w = max(220, int(W * 0.28))
                            panel_w = min(panel_w, 420)
                            # Create new canvas and paste original on left
                            from PIL import Image as _Image
                            canvas = _Image.new("RGB", (W + panel_w, H), (0, 0, 0))
                            canvas.paste(im, (0, 0))
                            pdraw = ImageDraw.Draw(canvas)
                            # Panel background
                            pdraw.rectangle([W, 0, W + panel_w, H], fill=(16, 16, 16))
                            pad = 10
                            y = pad
                            title = f"Detections: {n_boxes} | conf>={conf:.2f}"
                            try:
                                tw, th = pdraw.textsize(title, font=font)
                            except Exception:
                                tw, th = (len(title) * font_size // 2, font_size)
                            pdraw.text((W + pad, y), title, fill=(200, 220, 255), font=font)
                            y += th + 6
                            # Class counts
                            counts = {}
                            for d in detections:
                                counts[d["name"]] = counts.get(d["name"], 0) + 1
                            summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "None"
                            try:
                                sw, sh = pdraw.textsize(summary, font=font)
                            except Exception:
                                sw, sh = (len(summary) * font_size // 2, font_size)
                            pdraw.text((W + pad, y), summary, fill=(180, 255, 180), font=font)
                            y += sh + 10
                            # Thumbnails of top-K by area
                            topk = max(1, int(CONFIG.get("preview_yolo_topk", 5) or 5))
                            det_sorted = sorted(detections, key=lambda d: d["area"], reverse=True)[:topk]
                            thumb_w = panel_w - pad * 2
                            for idx, d in enumerate(det_sorted, 1):
                                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                                x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
                                x2 = max(0, min(W, x2)); y2 = max(0, min(H, y2))
                                if x2 <= x1 + 1 or y2 <= y1 + 1:
                                    continue
                                crop = im.crop((x1, y1, x2, y2))
                                # Resize maintaining aspect ratio
                                ratio = thumb_w / max(1, crop.width)
                                thh = max(30, int(crop.height * ratio))
                                try:
                                    crop = crop.resize((thumb_w, thh))
                                except Exception:
                                    pass
                                canvas.paste(crop, (W + pad, y))
                                # Caption
                                caption = f"{idx}. {d['name']} {d['conf']:.2f} | {int(d['size'][0])}×{int(d['size'][1])} ({int(d['area']*100)}%)"
                                y += thh + 4
                                pdraw.text((W + pad, y), caption, fill=(230, 230, 230), font=font)
                                # Advance
                                try:
                                    _, ch = pdraw.textsize(caption, font=font)
                                except Exception:
                                    ch = font_size
                                y += ch + 8
                            # Replace im with canvas for final encode
                            im = canvas
                    except Exception:
                        pass
                else:
                    # One retry at lower confidence to increase recall on small images
                    try:
                        low_conf = max(0.05, conf * 0.5)
                        res_list = model.predict(im, verbose=False, conf=low_conf)
                        if res_list:
                            res = res_list[0]
                            boxes = getattr(res, "boxes", None)
                            names = getattr(res, "names", {}) or {}
                            if boxes is not None and len(boxes) > 0:
                                xyxy = boxes.xyxy.cpu().numpy()
                                cls = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else None
                                confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else None
                                n_boxes = int(xyxy.shape[0])
                                color = (255, 50, 50)
                                for i in range(n_boxes):
                                    x1, y1, x2, y2 = xyxy[i].tolist()
                                    label = None
                                    if cls is not None:
                                        cid = int(cls[i])
                                        name = names.get(cid, str(cid))
                                        # Show class name only (drop confidence)
                                        label = name
                                    for t in range(thickness):
                                        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
                                    if label and font:
                                        tw, th = draw.textsize(label, font=font)
                                        y_text = max(0, y1 - th - 4)
                                        draw.rectangle([x1, y_text, x1 + tw + 8, y_text + th + 6], fill=(0, 0, 0))
                                        draw.text((x1 + 4, y_text + 3), label, fill=(255, 255, 255), font=font)
                    except Exception:
                        pass
            status_text = f"YOLO: {n_boxes}"
        elif YOLO_LAST_ERROR:
            status_text = "YOLO: error"
        # Do not draw the global top-left status label ("YOLO: n"). Intentionally removed.
        # Encode back to PNG
        out = BytesIO()
        im.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return b64_png


def _ensure_local_yolo_weights(name: str) -> Optional[str]:
    """Ensure a YOLO .pt file exists locally. Download to bin/models if missing.
    Returns local path or None on failure.
    """
    try:
        if not name or not name.endswith('.pt'):
            return None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dest_dir = os.path.join(base_dir, 'bin', 'models')
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, name)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            return dest_path
        url = f"https://github.com/ultralytics/assets/releases/latest/download/{name}"
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        tmp_path = dest_path + '.part'
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception:
        return None


# -----------------------------
# Session store with DB persistence
# -----------------------------
class AgentSession:
    def __init__(self, sid: str):
        self.session_id = sid
        self.messages: List[Dict] = []  # chat history
        self.plan: Dict = {}
        self.search_results: List[Dict] = []
        self.scraped_pages: Dict[str, Dict] = {}
        self.created_at = time.time()
        self._loaded = False

    def save_to_db(self):
        if not DB_AVAILABLE:
            return
        try:
            db = SessionLocal()
            # Check if session exists
            db_session = db.query(DBSession).filter(DBSession.id == self.session_id).first()
            if not db_session:
                db_session = DBSession(id=self.session_id, created_at=datetime.datetime.fromtimestamp(self.created_at))
                db.add(db_session)

            # Save messages
            db.query(Message).filter(Message.session_id == self.session_id).delete()
            for msg in self.messages:
                db_msg = Message(session_id=self.session_id, role=msg.get('role'), content=msg.get('content'))
                db.add(db_msg)

            # Save plan
            db.query(Plan).filter(Plan.session_id == self.session_id).delete()
            if self.plan:
                db_plan = Plan(session_id=self.session_id, queries=self.plan.get('queries'), steps=self.plan.get('steps'), rationale=self.plan.get('rationale'))
                db.add(db_plan)

            # Save search results
            db.query(SearchResult).filter(SearchResult.session_id == self.session_id).delete()
            for sr in self.search_results:
                db_sr = SearchResult(session_id=self.session_id, query=sr.get('query'), results=sr.get('results'))
                db.add(db_sr)

            # Save scraped pages
            db.query(ScrapedPage).filter(ScrapedPage.session_id == self.session_id).delete()
            for url, page in self.scraped_pages.items():
                db_page = ScrapedPage(session_id=self.session_id, url=url, title=page.get('title'), text=page.get('text'))
                db.add(db_page)

            db.commit()
        except Exception as e:
            print(f"Failed to save session {self.session_id} to DB: {e}")
        finally:
            db.close()

    def load_from_db(self):
        if not DB_AVAILABLE or self._loaded:
            return
        try:
            db = SessionLocal()
            db_session = db.query(DBSession).filter(DBSession.id == self.session_id).first()
            if db_session:
                self.created_at = db_session.created_at.timestamp()

                # Load messages
                self.messages = [{'role': m.role, 'content': m.content} for m in db_session.messages]

                # Load plan
                if db_session.plans:
                    plan = db_session.plans[0]
                    self.plan = {
                        'queries': plan.queries,
                        'steps': plan.steps,
                        'rationale': plan.rationale
                    }

                # Load search results
                self.search_results = [{'query': sr.query, 'results': sr.results} for sr in db_session.search_results]

                # Load scraped pages
                self.scraped_pages = {sp.url: {'url': sp.url, 'title': sp.title, 'text': sp.text} for sp in db_session.scraped_pages}

            self._loaded = True
        except Exception as e:
            print(f"Failed to load session {self.session_id} from DB: {e}")
        finally:
            db.close()


SESSIONS: Dict[str, AgentSession] = {}

# Cached HTML snapshots with DB persistence
CACHED_HTML: Dict[str, Dict] = {}

def cache_html_snapshot(html: str, url: str = "", title: str = "") -> str:
    cid = str(uuid.uuid4())
    cache_data = {
        "html": html or "",
        "url": url or "",
        "title": title or "",
        "created_at": time.time(),
    }
    CACHED_HTML[cid] = cache_data

    # Save to DB
    if DB_AVAILABLE:
        try:
            db = SessionLocal()
            db_cache = HtmlCache(id=cid, html=html, url=url, title=title)
            db.add(db_cache)
            db.commit()
        except Exception as e:
            print(f"Failed to save HTML cache {cid} to DB: {e}")
        finally:
            db.close()

    return cid

def load_html_cache_from_db():
    if not DB_AVAILABLE:
        return
    try:
        db = SessionLocal()
        caches = db.query(HtmlCache).all()
        for cache in caches:
            CACHED_HTML[cache.id] = {
                "html": cache.html,
                "url": cache.url,
                "title": cache.title,
                "created_at": cache.created_at.timestamp(),
            }
    except Exception as e:
        print(f"Failed to load HTML cache from DB: {e}")
    finally:
        db.close()

def _sanitize_snapshot(html: str) -> str:
    try:
        # Remove script tags
        html = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", html, flags=re.IGNORECASE | re.DOTALL)
        # Remove on* attributes (simple pass)
        html = re.sub(r"\son[a-zA-Z]+=\"[^\"]*\"", "", html)
        html = re.sub(r"\son[a-zA-Z]+='[^']*'", "", html)
    except Exception:
        pass
    return html


# -----------------------------
# Utilities
# -----------------------------
def sse(event: str, data: Dict) -> str:
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def sanitize_query(q: str) -> str:
    if not q:
        return ""
    q = q.replace("\n", " ").replace("\r", " ")
    q = q.strip()
    # Limit overly long inputs
    return q[:500]


def allowed_url(url: str) -> bool:
    if not url:
        return False
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    # Basic disallow list (auth/paywall heavy)
    blocked = []
    return not any(b in url for b in blocked)


def normalize_url(url: str) -> str:
    try:
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

        p = urlparse(url)
        params = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
                  if not k.lower().startswith("utm_") and k.lower() not in ("gclid", "fbclid")]
        clean = p._replace(query=urlencode(params), fragment="")
        u = urlunparse(clean)
        if u.endswith('/'):
            u = u[:-1]
        return u
    except Exception:
        return url


# -----------------------------
# OpenRouter Client
# -----------------------------
class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", fallback: Optional[Dict] = None):
        if not api_key:
            raise RuntimeError("API key is required.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.fallback = fallback or {}

    def _resolve_model(self, base_url: str, model: str) -> str:
        b = (base_url or "").lower()
        m = model
        # DeepSeek native API expects model ids like 'deepseek-chat' or 'deepseek-reasoner'.
        if "deepseek.com" in b:
            if "/" in m:
                m = m.split("/")[-1]
        return m

    def _post_chat(self, api_key: str, base_url: str, messages: List[Dict], model: str, stream: bool, extra_params: Optional[Dict] = None) -> requests.Response:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Research Assistant"),
        }
        resolved_model = self._resolve_model(base_url, model)
        payload = {"model": resolved_model, "messages": messages, "temperature": 0.2, "stream": stream}
        if extra_params:
            payload.update({k: v for k, v in extra_params.items() if v is not None})
        resp = requests.post(url, headers=headers, json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        return resp

    def chat(self, messages: List[Dict], model: str, stream: bool = False, extra_params: Optional[Dict] = None) -> requests.Response:
        return self._post_chat(self.api_key, self.base_url, messages, model, stream, extra_params=extra_params)

    def stream_chat_text(self, messages: List[Dict], model: str, extra_params: Optional[Dict] = None) -> Generator[dict, None, None]:
        """Yield dict events: {event: 'meta'|'token'|'usage'|'error', ...}"""
        def emit_error(msg: str):
            yield {"event": "error", "message": msg}

        try:
            resp = self.chat(messages, model=model, stream=True, extra_params=extra_params)
            meta = {
                "model": model,
                "base_url": self.base_url,
                "rate_limit": {
                    k: resp.headers.get(k)
                    for k in ("X-RateLimit-Remaining", "X-RateLimit-Limit", "X-RateLimit-Reset")
                    if resp.headers.get(k) is not None
                },
            }
            yield {"event": "meta", "meta": meta}

            last_usage = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        if "usage" in obj:
                            last_usage = obj["usage"]
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        # Stream content tokens (may be HTML)
                        content = delta.get("content")
                        if content:
                            yield {"event": "token", "text": content}
                        # Some providers stream separate reasoning/thinking tokens
                        reasoning = delta.get("reasoning") or delta.get("thought") or delta.get("thinking")
                        if reasoning:
                            yield {"event": "thinking", "text": reasoning}
                    except Exception:
                        continue
            if last_usage:
                yield {"event": "usage", "usage": last_usage, "model": model}
        except requests.HTTPError as he:
            try:
                body = he.response.text[:400]
            except Exception:
                body = str(he)
            # Add a hint for common provider/model mismatches
            hint = ""
            if "Model Not Exist" in body or "invalid_request_error" in body:
                hint = " Tip: check that your base URL matches the model vendor (e.g., deepseek.com with 'deepseek-chat' or use openrouter.ai with 'deepseek/deepseek-reasoner')."
            yield from emit_error(f"LLM HTTP error: {he} | {body}{hint}")
            yield from self._fallback_chat(messages, model, extra_params=extra_params)
        except Exception as e:
            yield from emit_error(f"LLM error: {e}")
            yield from self._fallback_chat(messages, model, extra_params=extra_params)

    def _fallback_chat(self, messages: List[Dict], model: str, extra_params: Optional[Dict] = None) -> Generator[dict, None, None]:
        def nonstream_once(m: str, api_key: str, base_url: str):
            resp = self._post_chat(api_key, base_url, messages, m, stream=False, extra_params=extra_params)
            obj = resp.json()
            message_obj = obj.get("choices", [{}])[0].get("message", {})
            content = message_obj.get("content", "")
            reasoning = message_obj.get("reasoning") or message_obj.get("thinking")
            usage = obj.get("usage")
            if content:
                yield {"event": "token", "text": content}
            if reasoning:
                yield {"event": "thinking", "text": reasoning}
            if usage:
                yield {"event": "usage", "usage": usage, "model": m}

        try:
            yield from nonstream_once(model, self.api_key, self.base_url)
        except Exception:
            fb = self.fallback.get("model") or model
            fb_key = self.fallback.get("api_key") or self.api_key
            fb_base = self.fallback.get("base_url") or self.base_url
            if fb != model:
                try:
                    yield {"event": "meta", "meta": {"model_fallback": fb, "base_url": fb_base}}
                    yield from nonstream_once(fb, fb_key, fb_base)
                    return
                except Exception as e2:
                    yield {"event": "error", "message": f"Fallback model failed: {e2}"}
            yield {"event": "error", "message": "All LLM calls failed."}


# -----------------------------
# Headless Browser Wrapper (undetected_chromedriver + Selenium)
# -----------------------------
class Browser:
    def __init__(self, headless: bool = True, pageload_timeout: int = 25, driver_path: Optional[str] = None, binary_path: Optional[str] = None):
        self.driver = self._init_driver(headless, driver_path=driver_path, binary_path=binary_path)
        self.driver.set_page_load_timeout(pageload_timeout)

    def _build_options(self, headless: bool, binary_path: Optional[str] = None):
        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        try:
            w = int(CONFIG.get("browser_window_width", 1280) or 1280)
            h = int(CONFIG.get("browser_window_height", 800) or 800)
            w = max(640, min(4096, w))
            h = max(480, min(2304, h))
            options.add_argument(f"--window-size={w},{h}")
        except Exception:
            options.add_argument("--window-size=1280,800")
        options.add_argument("--incognito")
        if binary_path:
            try:
                options.binary_location = binary_path
            except Exception:
                pass
        return options

    def _init_driver(self, headless: bool, driver_path: Optional[str] = None, binary_path: Optional[str] = None):
        # Allow manual pin via env var (e.g., 139, 140)
        env_version = os.getenv("UC_CHROME_VERSION_MAIN")

        # If a specific chromedriver path is provided or bundled, try that first
        candidate_paths = []
        if driver_path:
            candidate_paths.append(driver_path)
        # Look for bundled driver in bin/
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            win_path = os.path.join(base_dir, "bin", "chromedriver.exe")
            nix_path = os.path.join(base_dir, "bin", "chromedriver")
            if os.path.exists(win_path):
                candidate_paths.append(win_path)
            if os.path.exists(nix_path):
                candidate_paths.append(nix_path)
        except Exception:
            pass

        # Try provided/bundled driver paths with fresh options each attempt
        for p in candidate_paths:
            try:
                opts = self._build_options(headless, binary_path=binary_path)
                return uc.Chrome(options=opts, driver_executable_path=p)
            except Exception:
                continue

        # Try by version_main (from env), then default — with fresh options each time
        try:
            if env_version:
                opts_env = self._build_options(headless, binary_path=binary_path)
                return uc.Chrome(options=opts_env, version_main=int(env_version))
            opts_def = self._build_options(headless, binary_path=binary_path)
            return uc.Chrome(options=opts_def)
        except Exception as e:
            # Attempt to parse installed Chrome major version from error and retry
            msg = str(e)
            m = re.search(r"Current browser version is (\d+)", msg)
            if m:
                major = int(m.group(1))
                try:
                    opts_maj = self._build_options(headless, binary_path=binary_path)
                    return uc.Chrome(options=opts_maj, version_main=major)
                except Exception:
                    pass
            # As a last attempt, if headless, try non-headless with fresh options each call
            if headless:
                try:
                    if env_version:
                        opts_nh_env = self._build_options(False, binary_path=binary_path)
                        return uc.Chrome(options=opts_nh_env, version_main=int(env_version))
                    if m:
                        opts_nh_maj = self._build_options(False, binary_path=binary_path)
                        return uc.Chrome(options=opts_nh_maj, version_main=int(m.group(1)))
                    opts_nh = self._build_options(False, binary_path=binary_path)
                    return uc.Chrome(options=opts_nh)
                except Exception:
                    pass
            # Re-raise original exception if all retries fail
            raise

    # (rrweb helpers removed)

    # ---- Snapshot helpers (CSP-friendly fallbacks) ----
    def html_snapshot(self, max_len: int = 250_000) -> str:
        try:
            html = self.driver.execute_script("return document.documentElement ? document.documentElement.outerHTML : '';")
            if html and len(html) > max_len:
                return html[:max_len]
            return html or ""
        except Exception:
            return ""

    def screenshot_b64(self) -> Optional[str]:
        try:
            png = self.driver.get_screenshot_as_png()
            return base64.b64encode(png).decode('ascii')
        except Exception:
            return None

    def stream_screenshots(self, frames: int = 6, delay: float = 0.35) -> Generator[str, None, None]:
        for _ in range(max(1, frames)):
            b64 = self.screenshot_b64()
            if b64:
                yield b64
            time.sleep(max(0.0, delay))

    def pan_screenshots(self, frames: int = 20, delay: float = 0.15) -> Generator[str, None, None]:
        """Programmatically scroll down the page while taking screenshots to create a replay-like feel."""
        try:
            total = int(self.driver.execute_script(
                "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, document.body.offsetHeight, document.documentElement.offsetHeight, document.documentElement.clientHeight);"
            ) or 0)
            vh = int(self.driver.execute_script("return window.innerHeight;") or 0)
            max_y = max(0, total - vh)
            steps = max(1, int(frames))
            for i in range(steps):
                # Ease-in-out could be nicer; linear is fine
                y = int(max_y * (i / max(1, steps - 1)))
                try:
                    self.driver.execute_script("window.scrollTo(0, arguments[0]);", y)
                except Exception:
                    pass
                time.sleep(max(0.0, delay))
                b64 = self.screenshot_b64()
                if b64:
                    yield b64
        except Exception:
            # Fallback to static frames if scrolling fails
            yield from self.stream_screenshots(frames=frames, delay=delay)
        finally:
            try:
                self.driver.execute_script("window.scrollTo(0, 0);")
            except Exception:
                pass

    def quit(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    def _wait_ready(self, timeout: int = 20):
        WebDriverWait(self.driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        # DuckDuckGo as primary for fewer bot hurdles
        url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&t=h_&ia=web"
        results: List[Dict] = []
        try:
            self.driver.get(url)
            self._wait_ready(20)
            time.sleep(0.8)
            # Try a few selectors as DDG changes often
            anchors = []
            try:
                anchors = self.driver.find_elements(By.CSS_SELECTOR, 'a[data-testid="result-title-a"]')
            except Exception:
                pass
            if not anchors:
                anchors = self.driver.find_elements(By.CSS_SELECTOR, "h2 a, article a")

            for a in anchors:
                href = a.get_attribute("href")
                title = a.text.strip() or (a.get_attribute("title") or "").strip()
                if not href or not allowed_url(href):
                    continue
                clean = normalize_url(href)
                if any(r.get("url") == clean for r in results):
                    continue
                results.append({"title": title or clean, "url": clean, "snippet": ""})
                if len(results) >= max_results:
                    break
        except TimeoutException:
            pass
        except WebDriverException:
            pass
        # HTTP fallback if webdriver yields nothing or errors
        if not results:
            try:
                results = self._http_search(query, max_results=max_results)
            except Exception:
                pass
        return results

    def open_and_extract(self, url: str, timeout: int = 25) -> Optional[Dict]:
        if not allowed_url(url):
            return None
        try:
            url = normalize_url(url)
            self.driver.get(url)
            self._wait_ready(timeout)
            time.sleep(1.2)  # allow dynamic content to settle
            title = (self.driver.title or "").strip()
            text = self.driver.execute_script("return document.body ? document.body.innerText : '';")
            if text:
                text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if text and len(text) > 12000:
                text = text[:12000]
            return {"url": url, "title": title or url, "text": text or ""}
        except Exception:
            # Try HTTP fallback
            try:
                return self._http_fetch(url, timeout=timeout)
            except Exception:
                return None

    # ---- HTTP fallbacks (no webdriver) ----
    def _http_search(self, query: str, max_results: int = 5) -> List[Dict]:
        q = requests.utils.quote(query)
        # Use DDG HTML endpoint (simpler layout)
        ddg_url = f"https://duckduckgo.com/html/?q={q}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
        r = requests.get(ddg_url, headers=headers, timeout=10)
        r.raise_for_status()
        html = r.text
        # Very light extraction of results
        out: List[Dict] = []
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.IGNORECASE|re.DOTALL):
            href = html_unescape(m.group(1))
            title = re.sub("<[^>]+>", "", m.group(2))
            if not allowed_url(href):
                continue
            clean = normalize_url(href)
            if any(r.get("url") == clean for r in out):
                continue
            out.append({"title": (title or clean).strip(), "url": clean, "snippet": ""})
            if len(out) >= max_results:
                break
        return out

    def _http_fetch(self, url: str, timeout: int = 20) -> Optional[Dict]:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.text or ""
        # Extract title
        mt = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE|re.DOTALL)
        title = html_unescape(mt.group(1).strip()) if mt else url
        # Remove scripts/styles
        clean = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", "", text, flags=re.IGNORECASE|re.DOTALL)
        clean = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", "", clean, flags=re.IGNORECASE|re.DOTALL)
        # Strip tags to plain text
        clean = re.sub(r"<[^>]+>", "\n", clean)
        lines = [html_unescape(l).strip() for l in clean.splitlines()]
        joined = "\n".join(l for l in lines if l)
        if len(joined) > 12000:
            joined = joined[:12000]
        return {"url": normalize_url(url), "title": title, "text": joined}


# -----------------------------
# Persistent Browser Manager
# -----------------------------
class BrowserManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.browser: Optional[Browser] = None
        self.started_at: Optional[float] = None
        self.extension_active: bool = False
        self._last_ok: Optional[float] = None
        self._starting: bool = False
        self._start_error: Optional[str] = None
        self._start_thread: Optional[threading.Thread] = None
        self._start_started_at: Optional[float] = None
        self._start_seq: int = 0

    def start(self, conf: Dict) -> Dict:
        with self._lock:
            if self.browser is None:
                self.browser = Browser(
                    headless=conf.get("browser_headless", True),
                    pageload_timeout=conf.get("browser_pageload_timeout", 25),
                    driver_path=conf.get("chromedriver_path") or os.getenv("CHROMEDRIVER_PATH"),
                    binary_path=conf.get("chrome_binary_path") or os.getenv("CHROME_BINARY_PATH"),
                )
                self.extension_active = False
                self.started_at = time.time()
                # Touch driver to record first ok
                try:
                    self.browser.driver.execute_script("return 1;")
                    self._last_ok = time.time()
                except Exception:
                    self._last_ok = None
            return {"running": True, "started_at": self.started_at, "last_ok": self._last_ok, "starting": False, "error": None}

    def _start_worker(self, conf: Dict):
        # Deprecated: kept for compatibility if called directly
        self._start_worker_with_seq(conf, None)

    def _start_worker_with_seq(self, conf: Dict, seq: Optional[int]):
        try:
            self.start(conf)
            with self._lock:
                # Ignore completion if superseded
                if seq is not None and seq != self._start_seq:
                    return
                self._starting = False
                self._start_error = None
                self._start_thread = None
                self._start_started_at = None
        except Exception as e:
            with self._lock:
                if seq is not None and seq != self._start_seq:
                    return
                self._starting = False
                self._start_error = str(e)
                self._start_thread = None
                self._start_started_at = None
                # ensure cleared
                self.browser = None
                self.started_at = None
                self._last_ok = None

    def start_async(self, conf: Dict) -> Dict:
        with self._lock:
            if self.browser is not None:
                return {"running": True, "started_at": self.started_at, "last_ok": self._last_ok, "starting": False, "error": None}
            if self._starting and self._start_thread and self._start_thread.is_alive():
                return {"running": False, "started_at": None, "last_ok": None, "starting": True, "error": self._start_error}
            # kick off background start
            self._starting = True
            self._start_error = None
            self._start_seq += 1
            seq = self._start_seq
            self._start_started_at = time.time()
            self._start_thread = threading.Thread(target=self._start_worker_with_seq, args=(conf, seq), daemon=True)
            self._start_thread.start()
            return {"running": False, "started_at": None, "last_ok": None, "starting": True, "error": None}

    def stop(self) -> Dict:
        with self._lock:
            if self.browser:
                try:
                    self.browser.quit()
                finally:
                    self.browser = None
                    self.started_at = None
                    self._last_ok = None
                    self._starting = False
                    self._start_error = None
            return {"running": False, "started_at": None, "last_ok": None, "starting": False, "error": None}

    def status(self) -> Dict:
        with self._lock:
            running = False
            if self.browser is not None:
                try:
                    # Lightweight ping to ensure driver/session is alive
                    self.browser.driver.execute_script("return 1;")
                    running = True
                    self._last_ok = time.time()
                except Exception:
                    # Mark as not running if driver died
                    running = False
            # Apply startup timeout
            if not running and self._starting:
                try:
                    timeout_s = int(CONFIG.get("browser_start_timeout", 30))
                except Exception:
                    timeout_s = 30
                if self._start_started_at and (time.time() - self._start_started_at) > timeout_s:
                    # mark as failed due to timeout; drop reference to stale thread
                    self._starting = False
                    self._start_error = f"Startup timed out after {timeout_s}s. Check Chrome binary path and driver compatibility."
                    self._start_thread = None
                    self._start_started_at = None
            return {"running": running, "started_at": self.started_at, "last_ok": self._last_ok, "starting": self._starting, "error": self._start_error}

    def get_or_start(self, conf: Dict) -> Browser:
        with self._lock:
            # Start if missing and allowed
            if self.browser is None and conf.get("browser_autostart", True):
                self.start(conf)

            # If we have a browser, verify it's alive; if not, recycle
            if self.browser is not None:
                try:
                    self.browser.driver.execute_script("return 1;")
                    self._last_ok = time.time()
                except Exception:
                    # Try to rebuild driver once
                    try:
                        self.browser.quit()
                    except Exception:
                        pass
                    try:
                        self.browser = Browser(
                            headless=conf.get("browser_headless", True),
                            pageload_timeout=conf.get("browser_pageload_timeout", 25),
                            driver_path=conf.get("chromedriver_path") or os.getenv("CHROMEDRIVER_PATH"),
                            binary_path=conf.get("chrome_binary_path") or os.getenv("CHROME_BINARY_PATH"),
                        )
                        self.started_at = time.time()
                        self._last_ok = self.started_at
                    except Exception as e:
                        self.browser = None
                        self.started_at = None
                        self._last_ok = None
                        raise RuntimeError(f"Failed to restart browser: {e}")

            if self.browser is None:
                raise RuntimeError("Browser is not running. Start it from the Settings/Controls panel.")
            return self.browser

    def resize(self, width: int, height: int) -> Dict:
        with self._lock:
            if self.browser is None:
                raise RuntimeError("Browser is not running.")
            try:
                width = max(640, min(4096, int(width)))
                height = max(480, min(2304, int(height)))
            except Exception:
                raise RuntimeError("Invalid width/height")
            try:
                self.browser.driver.set_window_size(width, height)
            except Exception as e:
                raise RuntimeError(f"Failed to resize window: {e}")
            return {"width": width, "height": height}


BROWSER_MANAGER = BrowserManager()


# -----------------------------
# Agent logic (Plan -> Act -> Summarize)
# -----------------------------
class ResearchAgent:
    def __init__(self, planning_llm: OpenRouterClient, answer_llm: OpenRouterClient, planning_model: str, answer_model: str,
                 planning_params: Optional[Dict] = None, answer_params: Optional[Dict] = None, dev_show_prompts: bool = True):
        self.planning_llm = planning_llm
        self.answer_llm = answer_llm
        self.planning_model = planning_model
        self.answer_model = answer_model
        self.planning_params = planning_params or {}
        self.answer_params = answer_params or {}
        self.dev_show_prompts = dev_show_prompts

    def make_plan(self, question: str) -> Dict:
        sys_prompt = (
            "You are a research planner. Given a user question, create a concise plan with: "
            "1) 2-4 specific web search queries, 2) short ordered steps to validate facts across multiple sources.\n"
            "Return strict JSON with keys: 'queries' (array of strings), 'steps' (array of strings), and 'rationale' (1-2 sentence high-level justification)."
        )
        user = f"Question: {question}\nReturn only JSON."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        # Non-streaming small call for planning
        try:
            resp = self.planning_llm.chat(messages, model=self.planning_model, stream=False, extra_params=self.planning_params)
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            plan = json.loads(self._extract_json(content))
            queries = plan.get("queries") or []
            steps = plan.get("steps") or []
            if not isinstance(queries, list) or not queries:
                queries = [question]
            usage = data.get("usage")
            debug = {"phase": "plan", "model": self.planning_model, "messages": messages}
            return {"queries": [sanitize_query(q) for q in queries][:4], "steps": steps[:6], "rationale": plan.get("rationale")}, usage, debug
        except Exception:
            # Fallback plan
            base = sanitize_query(question)
            return {
                "queries": [base, f"site:.gov {base}", f"site:.edu {base}"],
                "steps": ["Search the web", "Open top sources", "Cross-verify facts", "Summarize with citations"],
            }, None, None

    def _extract_json(self, content: str) -> str:
        # Try to extract a JSON object from text
        content = content.strip()
        if content.startswith("{") and content.endswith("}"):
            return content
        # Heuristic: find first { ... } block
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and start < end:
            return content[start : end + 1]
        return "{}"

    def summarize_stream(self, question: str, sources: List[Dict]) -> Generator[dict, None, None]:
        # Build a compact context with enumerated sources
        numbered = []
        for i, s in enumerate(sources, 1):
            txt = s.get("text", "")[:2000]
            titled = f"[{i}] {s.get('title','').strip()}\nURL: {s.get('url','')}\nExtract:\n{txt}"
            numbered.append(titled)
        context = "\n\n".join(numbered)

        sys_prompt = (
            "You are a helpful research assistant. Use only the provided sources to answer. "
            "Return VALID, minimal HTML only (no style/script). Use <p>, <ul>, <ol>, <li>, <a>, <code>, <pre>, <blockquote>, <h3>-<h5>, <br>. "
            "Begin with a single-sentence <p><em>Approach:</em> ...</p> that states your method without exposing hidden chain-of-thought. "
            "Cite using [n] where n is the source number, and include a final <h4>Sources</h4><ul> list linking to each URL with anchors."
        )
        user_msg = (
            f"Question: {question}\n\nSources (numbered):\n{context}\n\n"
            "Write a conversational answer with citations inline like [1], [2] and a short 'Sources' list at the end."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        yield {"event": "prompt", "phase": "answer", "model": self.answer_model, "messages": messages}
        yield from self.answer_llm.stream_chat_text(messages, model=self.answer_model, extra_params=self.answer_params)


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/settings")
def settings_page():
    return render_template("settings.html")

@app.get("/api/settings")
def get_settings():
    # Do not mask here since it's a local panel; mask if needed
    return jsonify(CONFIG)

@app.post("/api/settings")
def post_settings():
    data = request.get_json(force=True, silent=True) or {}
    allowed_keys = set(CONFIG.keys())
    for k, v in list(data.items()):
        if k not in allowed_keys:
            data.pop(k)
    # Merge and persist
    CONFIG.update(data)
    save_config(CONFIG)
    return jsonify({"ok": True, "config": CONFIG})

@app.get("/api/health")
def api_health():
    import platform
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    try:
        import selenium
        info["packages"]["selenium"] = getattr(selenium, "__version__", "?")
    except Exception:
        pass
    try:
        import undetected_chromedriver as uc_mod
        info["packages"]["undetected_chromedriver"] = getattr(uc_mod, "__version__", "?")
    except Exception:
        pass
    try:
        import requests as rq
        info["packages"]["requests"] = getattr(rq, "__version__", "?")
    except Exception:
        pass
    # YOLO health
    try:
        import ultralytics as ul
        info["packages"]["ultralytics"] = getattr(ul, "__version__", "?")
    except Exception as e:
        info["packages"]["ultralytics"] = f"missing ({e})"
    try:
        import PIL
        info["packages"]["Pillow"] = getattr(PIL, "__version__", "?")
    except Exception as e:
        info["packages"]["Pillow"] = f"missing ({e})"
    try:
        import numpy as _np
        info["packages"]["numpy"] = getattr(_np, "__version__", "?")
    except Exception as e:
        info["packages"]["numpy"] = f"missing ({e})"
    try:
        import torch
        info["packages"]["torch"] = getattr(torch, "__version__", "?")
        info["packages"]["torch_cuda"] = torch.cuda.is_available()
    except Exception as e:
        info["packages"]["torch"] = f"missing ({e})"
    info["yolo"] = {
        "enabled": CONFIG.get("preview_yolo_enable", False),
        "available": YOLO_AVAILABLE,
        "model_loaded": YOLO_MODEL is not None,
        "model_path": CONFIG.get("preview_yolo_model"),
        "conf": CONFIG.get("preview_yolo_conf"),
        "last_error": YOLO_LAST_ERROR,
        "model_exists": os.path.exists(CONFIG.get("preview_yolo_model", "")) if isinstance(CONFIG.get("preview_yolo_model"), str) else False,
    }
    return jsonify({"ok": True, "health": info})


@app.post("/api/yolo/ensure")
def api_yolo_ensure():
    """Attempt to ensure YOLO weights are available locally and model can load."""
    try:
        name = os.path.basename(CONFIG.get("preview_yolo_model") or "yolov8n.pt")
        local = _ensure_local_yolo_weights(name)
        model = _get_yolo_model()
        return jsonify({
            "ok": model is not None,
            "local_path": local or CONFIG.get("preview_yolo_model"),
            "loaded": model is not None,
            "last_error": YOLO_LAST_ERROR,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/cache/<cache_id>")
def get_cached_html(cache_id: str):
    item = CACHED_HTML.get(cache_id)
    if not item:
        return "Not found", 404
    raw = item.get("html", "")
    url = item.get("url", "")
    title = item.get("title", "Cached Page")
    # Inject <base> for relative links if there's a head tag
    html = _sanitize_snapshot(raw)
    try:
        if "<head" in html.lower():
            # Insert base after first head tag
            html = re.sub(
                r"(<head[^>]*>)",
                r"\1\n<base href=\"%s\">" % (url.replace("\"", "&quot;")),
                html,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            # Wrap into minimal doc
            html = (
                f"<!doctype html><html><head><meta charset='utf-8'><base href='{url}'><title>{title}</title></head>"
                f"<body>{html}</body></html>"
            )
    except Exception:
        pass
    return Response(html, headers={"Content-Type": "text/html; charset=utf-8"})

@app.get("/api/browser/status")
def browser_status():
    return jsonify(BROWSER_MANAGER.status())

@app.get("/api/browser/diagnose")
def browser_diagnose():
    info = BROWSER_MANAGER.status().copy()
    diag = {"candidate_paths": [], "chrome_binary": CONFIG.get("chrome_binary_path") or os.getenv("CHROME_BINARY_PATH")}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        win_path = os.path.join(base_dir, "bin", "chromedriver.exe")
        nix_path = os.path.join(base_dir, "bin", "chromedriver")
        for p in [CONFIG.get("chromedriver_path"), os.getenv("CHROMEDRIVER_PATH"), win_path, nix_path]:
            if p and os.path.exists(p):
                diag["candidate_paths"].append({"path": p, "exists": True})
            elif p:
                diag["candidate_paths"].append({"path": p, "exists": False})
    except Exception:
        pass
    # quick network probe
    net = {"duckduckgo": None}
    try:
        r = requests.get("https://duckduckgo.com/", timeout=5)
        net["duckduckgo"] = (r.status_code, len(r.text))
    except Exception as e:
        net["duckduckgo"] = str(e)
    info["diagnose"] = {"paths": diag, "net": net}
    return jsonify(info)

@app.post("/api/browser/start")
def browser_start():
    try:
        # Optional width/height to set window size prior to start
        try:
            data = request.get_json(force=False, silent=True) or {}
            w = int(data.get("width")) if data and data.get("width") else None
            h = int(data.get("height")) if data and data.get("height") else None
            if w and h:
                CONFIG["browser_window_width"] = max(640, min(4096, w))
                CONFIG["browser_window_height"] = max(480, min(2304, h))
                save_config(CONFIG)
        except Exception:
            pass
        # Start without blocking the request
        status = BROWSER_MANAGER.start_async(CONFIG)
        return jsonify({"ok": True, **status})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/browser/stop")
def browser_stop():
    try:
        status = BROWSER_MANAGER.stop()
        return jsonify({"ok": True, **status})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/browser/restart")
def browser_restart():
    try:
        BROWSER_MANAGER.stop()
        status = BROWSER_MANAGER.start_async(CONFIG)
        return jsonify({"ok": True, **status})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/browser/resize")
def browser_resize():
    try:
        data = request.get_json(force=True, silent=True) or {}
        w = int(data.get("width"))
        h = int(data.get("height"))
        CONFIG["browser_window_width"] = max(640, min(4096, w))
        CONFIG["browser_window_height"] = max(480, min(2304, h))
        save_config(CONFIG)
        # If running, apply immediately
        try:
            out = BROWSER_MANAGER.resize(CONFIG["browser_window_width"], CONFIG["browser_window_height"])
            return jsonify({"ok": True, **out, "applied": True})
        except Exception as e:
            # Not running, that's fine; will apply on next start
            return jsonify({"ok": True, "applied": False, "error": str(e)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/api/browser/capture")
def browser_capture():
    try:
        conf = CONFIG
        br = BROWSER_MANAGER.get_or_start(conf)
        b64 = br.screenshot_b64()
        if not b64:
            return jsonify({"ok": False, "error": "no_screenshot"}), 500
        # Optionally annotate with YOLO
        try:
            b64 = _yolo_annotate_b64(b64)
        except Exception:
            pass
        return jsonify({"ok": True, "b64": b64})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/message")
def api_message():
    """
    POST body JSON: { message: string, session_id?: string, clicked_url?: string }
    Streams SSE events: session, plan, status, search_results, read_page, chunk, done, error
    """
    data = request.get_json(force=True, silent=True) or {}
    user_message = sanitize_query(data.get("message", "").strip())
    if not user_message:
        return jsonify({"error": "message is required"}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    clicked_url = data.get("clicked_url")

    if session_id not in SESSIONS:
        SESSIONS[session_id] = AgentSession(session_id)
        if DB_AVAILABLE:
            SESSIONS[session_id].load_from_db()
    sess = SESSIONS[session_id]
    sess.messages.append({"role": "user", "content": user_message})

    def generate() -> Generator[str, None, None]:
        # Session event
        yield sse("session", {"session_id": session_id})

        # Init clients
        try:
            conf = CONFIG
            planning_llm = OpenRouterClient(api_key=conf["planning_api_key"], base_url=conf["planning_base_url"])
            answer_llm = OpenRouterClient(
                api_key=conf["answer_api_key"],
                base_url=conf["answer_base_url"],
                fallback={
                    "model": conf.get("answer_model_fallback"),
                    "api_key": conf.get("answer_fallback_api_key") or conf.get("answer_api_key"),
                    "base_url": conf.get("answer_fallback_base_url") or conf.get("answer_base_url"),
                },
            )
        except Exception as e:
            yield sse("error", {"message": f"LLM init failed: {e}"})
            return
        
        agent = ResearchAgent(
            planning_llm,
            answer_llm,
            planning_model=conf["planning_model"],
            answer_model=conf["answer_model"],
            planning_params={
                "temperature": conf.get("planning_temperature"),
                "top_p": conf.get("planning_top_p"),
            },
            answer_params={
                "temperature": conf.get("answer_temperature"),
                "top_p": conf.get("answer_top_p"),
                "max_tokens": conf.get("answer_max_tokens"),
            },
            dev_show_prompts=conf.get("dev_show_prompts", False),
        )
        browser = None
        try:
            # PLAN
            yield sse("status", {"message": "Planning research steps..."})
            plan, plan_usage, plan_debug = agent.make_plan(user_message)
            sess.plan = plan
            # Planning meta and usage events
            yield sse("llm_meta", {"phase": "plan", "model": conf["planning_model"], "base_url": conf["planning_base_url"]})
            if plan_usage:
                yield sse("usage", {"phase": "plan", "usage": plan_usage, "model": conf["planning_model"]})
            if plan.get("rationale"):
                yield sse("thinking", {"phase": "plan", "text": plan.get("rationale")})
            if plan_debug:
                yield sse("prompt", plan_debug)
            yield sse("plan", plan)
            yield sse("trace", {"phase": "plan", "message": "Plan created", "steps": plan.get("steps", [])})
            sess.save_to_db()

            # ACT
            # Ensure persistent browser is running (fall back to HTTP if unavailable)
            browser = None
            try:
                browser = BROWSER_MANAGER.get_or_start(conf)
            except Exception as e:
                yield sse("status", {"message": f"Browser unavailable, using HTTP fallback. ({e})"})
            collected_results: List[Dict] = []
            yield sse("trace", {"phase": "act", "message": "Executing search queries", "queries": plan["queries"]})
            for q in plan["queries"][:4]:
                yield sse("status", {"message": f"Searching for: {q}"})
                if browser:
                    results = browser.search(q, max_results=conf.get("search_results_per_query", 5))
                else:
                    try:
                        results = Browser._http_search(self=None, query=q, max_results=conf.get("search_results_per_query", 5))  # type: ignore
                    except Exception:
                        results = []
                yield sse("search_results", {"query": q, "results": results})
                # Fallback previews: HTML and screenshots
                try:
                    if not browser:
                        raise RuntimeError("no_browser")
                    frames = max(3, int(conf.get("preview_frames", 24)))
                    delay = max(0.05, float(conf.get("preview_delay", 0.15)))
                    if conf.get("preview_enable_pan", True):
                        for frame in browser.pan_screenshots(frames=frames, delay=delay):
                            yield sse("screenshot", {"b64": _yolo_annotate_b64(frame)})
                    else:
                        for frame in browser.stream_screenshots(frames=frames, delay=delay):
                            yield sse("screenshot", {"b64": _yolo_annotate_b64(frame)})
                except Exception:
                    pass
                collected_results.extend(results)

            # If user clicked a link, prioritize it
            pages_to_read: List[Dict] = []
            if clicked_url and allowed_url(clicked_url):
                yield sse("status", {"message": "Opening user-selected page..."})
                if browser:
                    page = browser.open_and_extract(clicked_url)
                else:
                    try:
                        page = Browser._http_fetch(self=None, url=clicked_url, timeout=conf.get("browser_pageload_timeout", 25))  # type: ignore
                    except Exception:
                        page = None
                if page:
                    sess.scraped_pages[page["url"]] = page
                    yield sse("read_page", {"url": page["url"], "title": page["title"]})
                    sess.save_to_db()
                    # Fallback: HTML + screenshots
                    try:
                        if not browser:
                            raise RuntimeError("no_browser")
                        frames = max(5, int(conf.get("preview_frames", 24)))
                        delay = max(0.05, float(conf.get("preview_delay", 0.15)))
                        if conf.get("preview_enable_pan", True):
                            for frame in browser.pan_screenshots(frames=frames, delay=delay):
                                yield sse("screenshot", {"b64": _yolo_annotate_b64(frame)})
                        else:
                            for frame in browser.stream_screenshots(frames=frames, delay=delay):
                                yield sse("screenshot", {"b64": _yolo_annotate_b64(frame)})
                    except Exception:
                        pass
                    pages_to_read.append(page)

            # Auto-open top results if we still need context
            if len(pages_to_read) < conf.get("pages_to_open", 10):
                opened = 0
                for r in collected_results:
                    if opened >= (conf.get("pages_to_open", 10) - len(pages_to_read)):
                        break
                    url = r.get("url")
                    if not url or url in sess.scraped_pages:
                        continue
                    if browser:
                        page = browser.open_and_extract(url)
                    else:
                        try:
                            page = Browser._http_fetch(self=None, url=url, timeout=conf.get("browser_pageload_timeout", 25))  # type: ignore
                        except Exception:
                            page = None
                    if page and page.get("text"):
                        sess.scraped_pages[page["url"]] = page
                        yield sse("read_page", {"url": page["url"], "title": page["title"]})
                        sess.save_to_db()
                        # Fallback: HTML + screenshots
                        try:
                            if not browser:
                                raise RuntimeError("no_browser")
                            frames = max(5, int(conf.get("preview_frames", 24)))
                            delay = max(0.05, float(conf.get("preview_delay", 0.15)))
                            if conf.get("preview_enable_pan", True):
                                for frame in browser.pan_screenshots(frames=frames, delay=delay):
                                    yield sse("screenshot", {"b64": _yolo_annotate_b64(frame)})
                            else:
                                for frame in browser.stream_screenshots(frames=frames, delay=delay):
                                    yield sse("screenshot", {"b64": _yolo_annotate_b64(frame)})
                        except Exception:
                            pass
                        pages_to_read.append(page)
                        opened += 1

            if not pages_to_read:
                # Fallback to any previously scraped pages for the session
                pages_to_read = list(sess.scraped_pages.values())[: conf.get("pages_to_open", 10)]

            if not pages_to_read:
                yield sse("error", {"message": "No relevant pages could be opened."})
                return

            # SUMMARIZE (stream)
            if conf.get("dry_run", False):
                yield sse("status", {"message": "Dry-run mode: skipping summarization."})
                yield sse("done", {"ok": True, "dry_run": True})
                return
            yield sse("status", {"message": "Summarizing findings..."})
            yield sse("trace", {"phase": "summarize", "message": "Generating answer from gathered sources"})
            stream = agent.summarize_stream(user_message, pages_to_read)
            full_text = []
            for event in stream:
                if isinstance(event, dict):
                    et = event.get("event")
                    if et == "prompt":
                        if conf.get("dev_show_prompts", False):
                            yield sse("prompt", event)
                        continue
                    if et == "meta":
                        yield sse("llm_meta", event.get("meta", {}))
                    elif et == "token":
                        tok = event.get("text", "")
                        full_text.append(tok)
                        yield sse("chunk", {"text": tok})
                    elif et == "usage":
                        usage = event.get("usage", {})
                        model_used = event.get("model")
                        yield sse("usage", {"phase": "answer", "usage": usage, "model": model_used})
                    elif et == "error":
                        yield sse("error", {"message": event.get("message", "LLM error")})
                else:
                    full_text.append(str(event))
                    yield sse("chunk", {"text": str(event)})

            final = "".join(full_text)
            sess.messages.append({"role": "assistant", "content": final})
            sess.save_to_db()
            yield sse("done", {"ok": True})
        except Exception as e:
            yield sse("error", {"message": f"Unexpected error: {e}"})
        finally:
            # Persistent browser managed by BrowserManager; do not quit here
            pass

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(generate(), headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=True)
