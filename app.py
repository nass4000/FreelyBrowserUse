import os
import json
import time
import uuid
import queue
import threading
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
# Config / Environment
# -----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Default models; override via env vars
PLANNING_MODEL = os.getenv("OPENROUTER_PLANNING_MODEL", "google/gemini-2.5-flash")
ANSWER_MODEL = os.getenv("OPENROUTER_ANSWER_MODEL", "deepseek/deepseek-chat-v3.1")

# Browser settings
HEADLESS = os.getenv("BROWSER_HEADLESS", "1") not in ("0", "false", "False")
BROWSER_PAGELOAD_TIMEOUT = int(os.getenv("BROWSER_PAGELOAD_TIMEOUT", "25"))
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))
PAGES_TO_OPEN = int(os.getenv("PAGES_TO_OPEN", "3"))


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


# -----------------------------
# OpenRouter Client
# -----------------------------
class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = OPENROUTER_BASE_URL):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict], model: str, stream: bool = False) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional routing headers:
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Research Assistant"),
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "stream": stream,
        }
        resp = requests.post(url, headers=headers, json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        return resp

    def stream_chat_text(self, messages: List[Dict], model: str) -> Generator[str, None, None]:
        try:
            resp = self.chat(messages, model=model, stream=True)
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except Exception:
                        # Ignore malformed stream chunks
                        continue
        except Exception as e:
            # Fall back to non-streaming single completion
            yield from self._fallback_chat(messages, model)

    def _fallback_chat(self, messages: List[Dict], model: str) -> Generator[str, None, None]:
        try:
            resp = self.chat(messages, model=model, stream=False)
            obj = resp.json()
            content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Stream as a single chunk to the client
            if content:
                yield content
        except Exception as e:
            yield f"\n[LLM error: {e}]\n"


# -----------------------------
# Headless Browser Wrapper (undetected_chromedriver + Selenium)
# -----------------------------
class Browser:
    def __init__(self, headless: bool = HEADLESS):
        options = uc.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--incognito")
        self.driver = uc.Chrome(options=options)
        self.driver.set_page_load_timeout(BROWSER_PAGELOAD_TIMEOUT)

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
                results.append({"title": title or href, "url": href, "snippet": ""})
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
# Agent logic (Plan -> Act -> Summarize)
# -----------------------------
class ResearchAgent:
    def __init__(self, llm: OpenRouterClient):
        self.llm = llm

    def make_plan(self, question: str) -> Dict:
        sys_prompt = (
            "You are a research planner. Given a user question, create a concise plan with: "
            "1) 2-4 specific web search queries, 2) short ordered steps to validate facts across multiple sources.\n"
            "Return strict JSON with keys: 'queries' (array of strings) and 'steps' (array of strings)."
        )
        user = f"Question: {question}\nReturn only JSON."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        # Non-streaming small call for planning
        try:
            resp = self.llm.chat(messages, model=PLANNING_MODEL, stream=False)
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            plan = json.loads(self._extract_json(content))
            queries = plan.get("queries") or []
            steps = plan.get("steps") or []
            if not isinstance(queries, list) or not queries:
                queries = [question]
            return {"queries": [sanitize_query(q) for q in queries][:4], "steps": steps[:6]}
        except Exception:
            # Fallback plan
            base = sanitize_query(question)
            return {
                "queries": [base, f"site:.gov {base}", f"site:.edu {base}"],
                "steps": ["Search the web", "Open top sources", "Cross-verify facts", "Summarize with citations"],
            }

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

    def summarize_stream(self, question: str, sources: List[Dict]) -> Generator[str, None, None]:
        # Build a compact context with enumerated sources
        numbered = []
        for i, s in enumerate(sources, 1):
            txt = s.get("text", "")[:2000]
            titled = f"[{i}] {s.get('title','').strip()}\nURL: {s.get('url','')}\nExtract:\n{txt}"
            numbered.append(titled)
        context = "\n\n".join(numbered)

        sys_prompt = (
            "You are a helpful research assistant. Use only the provided sources to answer. "
            "Cite using [n] where n is the source number. Provide a concise, clear, and accurate answer."
        )
        user_msg = (
            f"Question: {question}\n\nSources (numbered):\n{context}\n\n"
            "Write a conversational answer with citations inline like [1], [2] and a short 'Sources' list at the end."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        return self.llm.stream_chat_text(messages, model=ANSWER_MODEL)


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


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
            llm = OpenRouterClient(OPENROUTER_API_KEY)
        except Exception as e:
            yield sse("error", {"message": f"LLM init failed: {e}"})
            return

        agent = ResearchAgent(llm)
        browser = None
        try:
            # PLAN
            yield sse("status", {"message": "Planning research steps..."})
            plan = agent.make_plan(user_message)
            sess.plan = plan
            yield sse("plan", plan)

            # ACT
            browser = Browser(headless=HEADLESS)
            collected_results: List[Dict] = []
            for q in plan["queries"][:4]:
                yield sse("status", {"message": f"Searching for: {q}"})
                results = browser.search(q, max_results=SEARCH_RESULTS_PER_QUERY)
                yield sse("search_results", {"query": q, "results": results})
                collected_results.extend(results)

            # If user clicked a link, prioritize it
            pages_to_read: List[Dict] = []
            if clicked_url and allowed_url(clicked_url):
                yield sse("status", {"message": "Opening user-selected page..."})
                page = browser.open_and_extract(clicked_url)
                if page:
                    sess.scraped_pages[page["url"]] = page
                    yield sse("read_page", {"url": page["url"], "title": page["title"]})
                    pages_to_read.append(page)

            # Auto-open top results if we still need context
            if len(pages_to_read) < PAGES_TO_OPEN:
                opened = 0
                for r in collected_results:
                    if opened >= (PAGES_TO_OPEN - len(pages_to_read)):
                        break
                    url = r.get("url")
                    if not url or url in sess.scraped_pages:
                        continue
                    page = browser.open_and_extract(url)
                    if page and page.get("text"):
                        sess.scraped_pages[page["url"]] = page
                        yield sse("read_page", {"url": page["url"], "title": page["title"]})
                        pages_to_read.append(page)
                        opened += 1

            if not pages_to_read:
                # Fallback to any previously scraped pages for the session
                pages_to_read = list(sess.scraped_pages.values())[: PAGES_TO_OPEN]

            if not pages_to_read:
                yield sse("error", {"message": "No relevant pages could be opened."})
                return

            # SUMMARIZE (stream)
            yield sse("status", {"message": "Summarizing findings..."})
            stream = agent.summarize_stream(user_message, pages_to_read)
            full_text = []
            for token in stream:
                full_text.append(token)
                yield sse("chunk", {"text": token})

            final = "".join(full_text)
            sess.messages.append({"role": "assistant", "content": final})
            yield sse("done", {"ok": True})
        except Exception as e:
            yield sse("error", {"message": f"Unexpected error: {e}"})
        finally:
            if browser:
                browser.quit()

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(generate(), headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

