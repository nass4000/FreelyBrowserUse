import os
import json
import time
import uuid
import queue
import threading
import re
import base64
from typing import Dict, List, Generator, Optional

from flask import Flask, Response, jsonify, render_template, request

# External deps
import requests

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
        "search_results_per_query": int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5")),
        "pages_to_open": int(os.getenv("PAGES_TO_OPEN", "10")),
        # Binaries
        "chromedriver_path": os.getenv("CHROMEDRIVER_PATH", ""),
        "chrome_binary_path": os.getenv("CHROME_BINARY_PATH", ""),
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
        # (rrweb removed)
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


# -----------------------------
# In-memory session store
# -----------------------------
class AgentSession:
    def __init__(self, sid: str):
        self.session_id = sid
        self.messages: List[Dict] = []  # chat history
        self.plan: Dict = {}
        self.search_results: List[Dict] = []
        self.scraped_pages: Dict[str, Dict] = {}
        self.created_at = time.time()


SESSIONS: Dict[str, AgentSession] = {}

# Cached HTML snapshots
CACHED_HTML: Dict[str, Dict] = {}

def cache_html_snapshot(html: str, url: str = "", title: str = "") -> str:
    cid = str(uuid.uuid4())
    CACHED_HTML[cid] = {
        "html": html or "",
        "url": url or "",
        "title": title or "",
        "created_at": time.time(),
    }
    return cid

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
    blocked = [
        "accounts.google.",
        "facebook.com/login",
        "x.com/i/flow",
        "linkedin.com/uas/login",
        "github.com/login",
        "auth.",
    ]
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
            return None


# -----------------------------
# Persistent Browser Manager
# -----------------------------
class BrowserManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.browser: Optional[Browser] = None
        self.started_at: Optional[float] = None
        self.extension_active: bool = False

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
            return {"running": True, "started_at": self.started_at}

    def stop(self) -> Dict:
        with self._lock:
            if self.browser:
                try:
                    self.browser.quit()
                finally:
                    self.browser = None
                    self.started_at = None
            return {"running": False}

    def status(self) -> Dict:
        with self._lock:
            return {"running": self.browser is not None, "started_at": self.started_at}

    def get_or_start(self, conf: Dict) -> Browser:
        with self._lock:
            if self.browser is None and conf.get("browser_autostart", True):
                # Reuse start() logic
                self.start(conf)
            if self.browser is None:
                raise RuntimeError("Browser is not running. Start it from the Settings/Controls panel.")
            return self.browser


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
    return jsonify({"ok": True, "health": info})

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

@app.post("/api/browser/start")
def browser_start():
    try:
        status = BROWSER_MANAGER.start(CONFIG)
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

            # ACT
            # Ensure persistent browser is running
            browser = BROWSER_MANAGER.get_or_start(conf)
            collected_results: List[Dict] = []
            yield sse("trace", {"phase": "act", "message": "Executing search queries", "queries": plan["queries"]})
            for q in plan["queries"][:4]:
                yield sse("status", {"message": f"Searching for: {q}"})
                results = browser.search(q, max_results=conf.get("search_results_per_query", 5))
                yield sse("search_results", {"query": q, "results": results})
                # Fallback previews: HTML and screenshots
                try:
                    snap_html = browser.html_snapshot()
                    if snap_html:
                        cid = cache_html_snapshot(snap_html, getattr(browser.driver, 'current_url', ''), browser.driver.title)
                        yield sse("page_cached", {"url": f"/cache/{cid}", "cache_id": cid, "title": browser.driver.title})
                    for frame in browser.stream_screenshots(frames=3, delay=0.4):
                        yield sse("screenshot", {"b64": frame})
                except Exception:
                    pass
                collected_results.extend(results)

            # If user clicked a link, prioritize it
            pages_to_read: List[Dict] = []
            if clicked_url and allowed_url(clicked_url):
                yield sse("status", {"message": "Opening user-selected page..."})
                page = browser.open_and_extract(clicked_url)
                if page:
                    sess.scraped_pages[page["url"]] = page
                    yield sse("read_page", {"url": page["url"], "title": page["title"]})
                    # Fallback: HTML + screenshots
                    try:
                        snap_html = browser.html_snapshot()
                        if snap_html:
                            cid = cache_html_snapshot(snap_html, page["url"], page["title"])
                            yield sse("page_cached", {"url": f"/cache/{cid}", "cache_id": cid, "title": page["title"]})
                        for frame in browser.stream_screenshots(frames=5, delay=0.35):
                            yield sse("screenshot", {"b64": frame})
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
                    page = browser.open_and_extract(url)
                    if page and page.get("text"):
                        sess.scraped_pages[page["url"]] = page
                        yield sse("read_page", {"url": page["url"], "title": page["title"]})
                        # Fallback: HTML + screenshots
                        try:
                            snap_html = browser.html_snapshot()
                            if snap_html:
                                cid = cache_html_snapshot(snap_html, page["url"], page["title"])
                                yield sse("page_cached", {"url": f"/cache/{cid}", "cache_id": cid, "title": page["title"]})
                            for frame in browser.stream_screenshots(frames=5, delay=0.35):
                                yield sse("screenshot", {"b64": frame})
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
