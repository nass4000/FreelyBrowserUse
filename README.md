Deep Research Assistant (Flask + Selenium + OpenRouter)
======================================================

Overview
--------
Web-based ChatGPT-like assistant that performs Plan → Act → Summarize deep web research using:
- Flask backend with Server-Sent Events (SSE) for streaming
- undetected-chromedriver (Selenium) in headless mode for browsing
- OpenRouter API for multi-model LLMs (planning vs. answering)

Key Files
---------
- `app.py` — Flask app, agent pipeline, Selenium + OpenRouter integration
- `templates/index.html` — minimal chat UI
- `static/app.js` — streaming client, shows intermediate actions
- `static/styles.css` — lightweight styling
- `requirements.txt` — Python dependencies

Setup
-----
1) Prerequisites
- Python 3.10+
- Google Chrome or Chromium installed
- Network access for OpenRouter API

2) Install deps
```
pip install -r requirements.txt
```

3) Environment
Set your OpenRouter key and optional model overrides:
```
$env:OPENROUTER_API_KEY="sk-or-..."           # PowerShell
set OPENROUTER_API_KEY=sk-or-...               # cmd
export OPENROUTER_API_KEY=sk-or-...            # bash/zsh

# Optional overrides
export OPENROUTER_PLANNING_MODEL="google/gemini-2.5-flash"
export OPENROUTER_ANSWER_MODEL="deepseek/deepseek-chat-v3.1"
export BROWSER_HEADLESS=1
```

4) Run
```
python app.py
```
Open http://localhost:5000 in your browser.

How It Works
------------
1) Plan — Calls a "planning model" to produce JSON with search queries + steps.
2) Act — Headless Selenium searches DuckDuckGo, parses results, and opens top pages.
3) Summarize — Streams LLM answer using the collected page extracts with inline [n] citations.

User Influence
--------------
When search results appear, click a link to steer browsing. The frontend posts the clicked URL back; the backend prioritizes opening and extracting that page before summarizing.

Security & Robustness
---------------------
- Sanitizes user queries (length, newlines)
- Restricts navigation to http(s) and blocks common auth pages
- Handles timeouts/errors in browsing and LLM calls; falls back to non-streaming

Notes
-----
- Google can aggressively block automation; DuckDuckGo is used by default. Adjust selectors if engines change.
- Headless browsing can be toggled via `BROWSER_HEADLESS`.
- This demo uses an in-memory session store; not suitable for multi-process or production without a shared store.

