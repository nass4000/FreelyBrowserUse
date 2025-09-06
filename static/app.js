// Simple chat UI with streaming SSE over fetch

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('chat-form');
const inputEl = document.getElementById('chat-input');
const startBtn = document.getElementById('btn-start');
const stopBtn = document.getElementById('btn-stop');
const browserStatusEl = document.getElementById('browser-status');

let sessionId = null;
let lastUserMessage = '';
let streamingController = null;
let rrwebReplayer = null;
const screenshotEl = document.getElementById('page-screenshot');
const pageHtmlEl = document.getElementById('page-html');
const pageIframe = document.getElementById('page-iframe');

function addUserMessage(text) {
  const m = document.createElement('div');
  m.className = 'msg user';
  m.textContent = text;
  chatEl.appendChild(m);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addAssistantContainer() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant';
  const content = document.createElement('div');
  content.className = 'assistant-content';
  const actions = document.createElement('div');
  actions.className = 'assistant-actions';
  const timeline = document.createElement('div');
  timeline.className = 'timeline-block';
  const th = document.createElement('div');
  th.className = 'label';
  th.textContent = 'Timeline';
  timeline.appendChild(th);
  wrap.appendChild(actions);
  wrap.appendChild(timeline);
  wrap.appendChild(content);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { wrap, actions, content, timeline };
}

function addStatus(actionsEl, text) {
  const s = document.createElement('div');
  s.className = 'status';
  s.textContent = text;
  actionsEl.appendChild(s);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addTimeline(timelineEl, text) {
  const it = document.createElement('div');
  it.className = 'timeline-item';
  it.textContent = text;
  timelineEl.appendChild(it);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addPrompt(timelineEl, payload) {
  const wrap = document.createElement('div');
  wrap.className = 'timeline-item prompt-block';
  const title = document.createElement('div');
  title.className = 'label';
  title.textContent = `Prompt (${payload.phase || 'n/a'}) — ${payload.model || ''}`;
  wrap.appendChild(title);
  const pre = document.createElement('pre');
  pre.className = 'prompt-pre';
  try {
    const msgs = payload.messages || [];
    const lines = msgs.map(m => `- ${m.role}: ${m.content}`).join('\n');
    pre.textContent = lines;
  } catch (e) {
    pre.textContent = JSON.stringify(payload, null, 2);
  }
  wrap.appendChild(pre);
  timelineEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function renderPlan(actionsEl, plan) {
  const block = document.createElement('div');
  block.className = 'plan-block';
  const h = document.createElement('div');
  h.className = 'label';
  h.textContent = 'Plan';
  block.appendChild(h);

  if (plan.steps && plan.steps.length) {
    const ul = document.createElement('ul');
    plan.steps.forEach(st => {
      const li = document.createElement('li');
      li.textContent = st;
      ul.appendChild(li);
    });
    block.appendChild(ul);
  }
  if (plan.queries && plan.queries.length) {
    const qh = document.createElement('div');
    qh.className = 'label';
    qh.textContent = 'Search queries';
    block.appendChild(qh);
    const ql = document.createElement('ul');
    plan.queries.forEach(q => {
      const li = document.createElement('li');
      li.textContent = q;
      ql.appendChild(li);
    });
    block.appendChild(ql);
  }
  actionsEl.appendChild(block);
}

function renderSearchResults(actionsEl, query, results) {
  const block = document.createElement('div');
  block.className = 'results-block';
  const h = document.createElement('div');
  h.className = 'label';
  h.textContent = `Results for: ${query}`;
  block.appendChild(h);
  const list = document.createElement('div');
  list.className = 'results-list';

  results.forEach(r => {
    const item = document.createElement('div');
    item.className = 'result-item';
    const a = document.createElement('a');
    a.href = r.url;
    a.textContent = r.title || r.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.addEventListener('click', (e) => {
      e.preventDefault();
      // Allow user to influence browsing by selecting a link
      if (streamingController) {
        streamingController.abort();
      }
      sendMessage(lastUserMessage, r.url);
    });
    item.appendChild(a);
    if (r.snippet) {
      const sn = document.createElement('div');
      sn.className = 'snippet';
      sn.textContent = r.snippet;
      item.appendChild(sn);
    }
    list.appendChild(item);
  });

  block.appendChild(list);
  actionsEl.appendChild(block);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function parseSSEChunk(buffer, onEvent) {
  // Buffer may contain multiple events separated by \n\n
  const parts = buffer.split('\n\n');
  // Keep the last partial chunk in the buffer by returning it
  const lastPartial = buffer.endsWith('\n\n') ? '' : parts.pop();
  for (const part of parts) {
    const lines = part.split('\n');
    let eventType = 'message';
    let data = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice('event: '.length);
      } else if (line.startsWith('data: ')) {
        data += line.slice('data: '.length);
      }
    }
    try {
      const parsed = JSON.parse(data);
      onEvent(eventType, parsed);
    } catch (e) {
      // ignore malformed
    }
  }
  return lastPartial;
}

async function sendMessage(message, clickedUrl = null) {
  if (!message) return;
  lastUserMessage = message;
  addUserMessage(message);
  inputEl.value = '';

  const { actions, content, timeline } = addAssistantContainer();
  addStatus(actions, 'Starting...');
  let htmlBuffer = '';

  const ctrl = new AbortController();
  streamingController = ctrl;

  try {
    const resp = await fetch('/api/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, clicked_url: clickedUrl }),
      signal: ctrl.signal,
    });

    if (!resp.ok || !resp.body) {
      addStatus(actions, `Request failed: ${resp.status}`);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      buf = parseSSEChunk(buf, (event, data) => {
        switch (event) {
          case 'session':
            sessionId = data.session_id;
            break;
          case 'status':
            addStatus(actions, data.message || '...');
            break;
          case 'plan':
            renderPlan(actions, data);
            addTimeline(timeline, 'Plan ready');
            break;
          case 'search_results':
            renderSearchResults(actions, data.query, data.results || []);
            addTimeline(timeline, `Searched: ${data.query}`);
            break;
          case 'read_page':
            addStatus(actions, `Reading: ${data.title || data.url}`);
            addTimeline(timeline, `Opened: ${data.title || data.url}`);
            break;
          case 'chunk':
            htmlBuffer += data.text || '';
            content.innerHTML = sanitizeHTML(htmlBuffer);
            chatEl.scrollTop = chatEl.scrollHeight;
            break;
          case 'rrweb':
            try {
              const evs = data.events || [];
              if (!rrwebReplayer && window.rrweb) {
                const container = document.getElementById('rrweb-player');
                rrwebReplayer = new rrweb.Replayer([], { root: container, liveMode: true });
                rrwebReplayer.play();
              }
              if (rrwebReplayer) {
                evs.forEach(e => rrwebReplayer.addEvent(e));
              }
            } catch (e) {}
            break;
          case 'screenshot':
            try {
              if (data.b64) {
                screenshotEl.src = `data:image/png;base64,${data.b64}`;
              }
            } catch (e) {}
            break;
          case 'page_html':
            try {
              if (data.html && pageHtmlEl) {
                pageHtmlEl.innerHTML = sanitizeHTML(data.html);
              }
            } catch (e) {}
            break;
          case 'page_cached':
            try {
              if (data.url && pageIframe) {
                // bust cache
                pageIframe.src = data.url + '?t=' + Date.now();
              }
            } catch (e) {}
            break;
          case 'llm_meta':
            if (data.phase === 'plan') {
              addTimeline(timeline, `Planning model: ${data.model || ''} (${data.base_url || ''})`);
            } else {
              addStatus(actions, `Answer model: ${data.model || ''}`);
              addTimeline(timeline, `Answer model: ${data.model || ''} (${data.base_url || ''})`);
            }
            break;
          case 'usage':
            try {
              const u = data.usage || {};
              const phase = data.phase || 'answer';
              const txt = `${phase} usage — total: ${u.total_tokens ?? '?'}, prompt: ${u.prompt_tokens ?? '?'}, completion: ${u.completion_tokens ?? '?'}`;
              addStatus(actions, txt);
              addTimeline(timeline, txt);
            } catch {}
            break;
          case 'trace':
            if (data.message) addTimeline(timeline, data.message);
            break;
          case 'prompt':
            addPrompt(timeline, data);
            break;
          case 'thinking':
            addTimeline(timeline, `Thinking: ${data.text}`);
            break;
          case 'error':
            addStatus(actions, `Error: ${data.message}`);
            addTimeline(timeline, `Error: ${data.message}`);
            break;
          case 'done':
            addStatus(actions, 'Done');
            break;
        }
      });
    }
  } catch (e) {
    addStatus(actions, `Network error: ${e.message}`);
  } finally {
    streamingController = null;
  }
}

formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  const v = inputEl.value.trim();
  if (!v) return;
  sendMessage(v);
});

function sanitizeHTML(input) {
  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(input, 'text/html');
    const allowed = new Set(['P','BR','UL','OL','LI','A','CODE','PRE','BLOCKQUOTE','H3','H4','H5','EM','STRONG','B','I']);
    const walker = document.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT, null);
    const toRemove = [];
    while (walker.nextNode()) {
      const el = walker.currentNode;
      const tag = el.tagName;
      // Remove dangerous tags entirely
      if (['SCRIPT','STYLE','IFRAME','OBJECT','EMBED','LINK','META'].includes(tag)) {
        toRemove.push(el);
        continue;
      }
      // Unwrap unknown tags: replace with its children
      if (!allowed.has(tag)) {
        const parent = el.parentNode;
        while (el.firstChild) parent.insertBefore(el.firstChild, el);
        toRemove.push(el);
        continue;
      }
      // Scrub attributes
      for (const attr of Array.from(el.attributes)) {
        const name = attr.name.toLowerCase();
        const value = attr.value || '';
        if (name.startsWith('on')) { el.removeAttribute(attr.name); continue; }
        if (name === 'style') { el.removeAttribute('style'); continue; }
        if (tag === 'A' && name === 'href') {
          // Allow only http/https
          if (!/^https?:\/\//i.test(value)) { el.removeAttribute('href'); }
          else {
            el.setAttribute('target','_blank');
            el.setAttribute('rel','noopener noreferrer');
          }
          continue;
        }
        // Drop any other attributes
        if (!(tag === 'A' && name === 'href')) {
          el.removeAttribute(attr.name);
        }
      }
    }
    toRemove.forEach(node => node.remove());
    return doc.body.innerHTML;
  } catch (e) {
    // Fallback: escape
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
  }
}

async function refreshBrowserStatus() {
  try {
    const res = await fetch('/api/browser/status');
    const j = await res.json();
    if (j.running) {
      browserStatusEl.textContent = 'Browser: running';
      browserStatusEl.style.background = '#16301f';
      browserStatusEl.style.borderColor = '#335f45';
    } else {
      browserStatusEl.textContent = 'Browser: stopped';
      browserStatusEl.style.background = '#2a1c1c';
      browserStatusEl.style.borderColor = '#5e2c2c';
    }
  } catch (e) {
    browserStatusEl.textContent = 'Browser: unknown';
  }
}

async function startBrowser() {
  await fetch('/api/browser/start', { method: 'POST' });
  await refreshBrowserStatus();
}

async function stopBrowser() {
  await fetch('/api/browser/stop', { method: 'POST' });
  await refreshBrowserStatus();
}

startBtn?.addEventListener('click', startBrowser);
stopBtn?.addEventListener('click', stopBrowser);
refreshBrowserStatus();
