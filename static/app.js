// Simple chat UI with streaming SSE over fetch

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('chat-form');
const inputEl = document.getElementById('chat-input');

let sessionId = null;
let lastUserMessage = '';
let streamingController = null;

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
  wrap.appendChild(actions);
  wrap.appendChild(content);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { wrap, actions, content };
}

function addStatus(actionsEl, text) {
  const s = document.createElement('div');
  s.className = 'status';
  s.textContent = text;
  actionsEl.appendChild(s);
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

  const { actions, content } = addAssistantContainer();
  addStatus(actions, 'Starting...');

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
            break;
          case 'search_results':
            renderSearchResults(actions, data.query, data.results || []);
            break;
          case 'read_page':
            addStatus(actions, `Reading: ${data.title || data.url}`);
            break;
          case 'chunk':
            content.textContent += data.text || '';
            chatEl.scrollTop = chatEl.scrollHeight;
            break;
          case 'error':
            addStatus(actions, `Error: ${data.message}`);
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

