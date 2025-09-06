# ALPHA v1 Development Log

This document summarizes the architectural decisions and major iterations made while building the Deep Research Assistant (Flask + Selenium + LLM).

## Overview
- Flask backend with Server‑Sent Events (SSE) for streaming.
- Headless browsing via undetected‑chromedriver (Selenium) with a persistent browser instance.
- OpenRouter/DeepSeek compatible LLM client with per‑phase routing (planning vs. answering), usage reporting, and minimal‑HTML answers.
- Web UI with a chat panel and a live browser monitor (screenshot feed), plus a settings panel to tune behavior.

## Architecture
- Backend
  - `OpenRouterClient` wraps OpenRouter‑style chat completions and supports vendor‑specific normalization (DeepSeek: `deepseek-chat`, `deepseek-reasoner`). Includes streaming, non‑streaming fallback, and optional fallback model/base URL/API key.
  - `ResearchAgent` runs Plan → Act → Summarize:
    - Plan: produce search queries + steps (+ rationale). Emits `llm_meta`, `usage`, `prompt`, and `thinking` events.
    - Act: execute searches and open pages, extract plaintext, and stream preview screenshots to the UI.
    - Summarize: stream a VALID, minimal HTML answer with citations; usage/meta are emitted when available.
  - `Browser` encapsulates Selenium, with:
    - Robust chromedriver initialization (supports explicit driver path, bundled `bin/chromedriver(.exe)`, or `UC_CHROME_VERSION_MAIN`).
    - Utilities for page readiness, normalized URL search results, HTML snapshots, and `screenshot_b64`.
    - Replay‑like panning: `pan_screenshots(frames, delay)` scrolls and captures a burst of frames to simulate a replay.
  - `BrowserManager` maintains a persistent browser instance and exposes `/api/browser/start|stop|status|capture`.
  - Config system (`config.json` + env) with runtime GET/POST `/api/settings`.

- Frontend
  - Chat UI streams plan/status/search results/read‑page/answer chunks/usage/meta.
  - Live browser monitor shows screenshots; a poller fetches a frame every ~1.5s when idle, and faster frames stream during actions.
  - Settings page edits config without restarting the server.

## Key Endpoints
- UI
  - `GET /` — Chat + Monitor
  - `GET /settings` — Settings panel
- Config
  - `GET /api/settings` — Current config
  - `POST /api/settings` — Update & persist config
- Browser
  - `GET /api/browser/status` — Running state
  - `POST /api/browser/start` — Start persistent browser
  - `POST /api/browser/stop` — Stop browser
  - `GET /api/browser/capture` — One‑off screenshot (base64)
- Chat
  - `POST /api/message` — Streams SSE events: `session`, `status`, `plan`, `search_results`, `read_page`, `thinking`, `llm_meta`, `usage`, `chunk`, `done`, `error`, `trace`, `prompt` (dev only)
- Snapshots (currently retained but unused in UI)
  - `GET /cache/<cache_id>` — Returns sanitized cached HTML with `<base>` injection (safe viewer)

## Iterations & Decisions
1. Baseline app
   - Flask SSE + undetected‑chromedriver
   - OpenRouter client + planning/answer models + minimal HTML answers
   - Display: plan, actions, streamed tokens, usage/meta
2. Robustness & Observability
   - Added fallback LLM calls (non‑stream and fallback model)
   - DeepSeek normalization for native API (strip vendor prefix)
   - Timeline/trace events; usage & meta surfacing
3. Browser stability
   - Driver auto‑retry/pinning and bundled `bin/chromedriver` support
   - Settings to control headless, timeouts, and counts
4. Recording attempts
   - rrweb CDN inject → blocked by CSP
   - rrweb extension (unpacked) → platform‑dependent and brittle → fully removed
   - Replaced with CSP‑safe previews: HTML snapshots (cached route) + screenshots
5. Replay‑like monitor
   - Implemented `pan_screenshots` for a short scrolling replay after each navigation
   - Added `/api/browser/capture` + client polling for an always‑on live monitor
6. UI
   - Final layout: Chat on the left, Live Browser Monitor on the right
   - Removed iframe/snapshot panes for simplicity; retained screenshot feed as the main monitor

## Configuration (selected)
- LLM routing
  - `planning_*` and `answer_*` keys for base URL, API key, and model
  - `answer_model_fallback`, `answer_fallback_*` for fallback model/provider
- Decoding
  - `planning_temperature`, `planning_top_p`
  - `answer_temperature`, `answer_top_p`, `answer_max_tokens`
- Browser & search
  - `browser_headless`, `browser_pageload_timeout`
  - `search_results_per_query`, `pages_to_open`
  - `chromedriver_path`, `chrome_binary_path`, `BROWSER_AUTOSTART`
- Preview / replay
  - `preview_enable_pan` (default: true)
  - `preview_frames` (default: 24)
  - `preview_delay` (default: 0.15s)

## Known Limitations
- Snapshot route `/cache/<id>` exists but is not used in the current UI (kept for future safe viewing/archiving).
- Panning replay provides a page‑level scroll preview; it does not capture mouse movements/clicks like rrweb.
- In‑memory stores (`SESSIONS`, `CACHED_HTML`) are per‑process and non‑durable.

## Next Steps
- Add preview controls to Settings (frames/delay/pan toggle) for quick tuning.
- Optional export: session logs (trace + prompts + usage) to JSONL.
- Optional provider self‑test in Settings to validate credentials/models.
- Optional continuous lower‑FPS capture during Act for even smoother monitor updates.

---
Generated as part of ALPHA v1 to document implementation scope and decisions.
