// The chat UI as a small React app (React + htm are vendored locally — no build step, fully
// offline). It reuses the unit-tested render helpers from chart.js (drawChart / legendHtml) for
// the SVG charts. State is client-side only; every number still comes from the Flask /api/ask
// endpoint, which computes it deterministically — the app only presents it.

const { useState, useEffect, useRef } = React;

// Bind htm to a thin createElement wrapper so templates can use plain HTML `class`/`for`
// attributes (translated to React's className/htmlFor) — keeping the markup readable.
function h(type, props, ...children) {
  if (props) {
    if ('class' in props) { props.className = props.class; delete props.class; }
    if ('for' in props) { props.htmlFor = props.for; delete props.for; }
  }
  return React.createElement(type, props, ...children);
}
const html = htm.bind(h);

// ---- header: dataset line, Ollama status lamp, "Explain with AI" toggle + model picker ----
function Header({ dsInfo, aiAvailable, models, model, setModel, llmOn, setLlmOn }) {
  const ready = aiAvailable === true && models.length > 0;
  const statusText = aiAvailable === null ? 'checking…'
    : ready ? 'AI ready' : 'AI off — plain answers';
  return html`
    <header><div class="bar wrap">
      <div class="brand">
        <span class="badge" aria-hidden="true">🌡️</span>
        <div>
          <h1>Weather Questions</h1>
          <div class="sub">${dsInfo}</div>
        </div>
      </div>
      <div class="controls">
        <span class="status">
          <span class=${'dot' + (ready ? ' on' : '')} aria-hidden="true"></span>
          <span role="status" aria-live="polite">${statusText}</span>
        </span>
        <label class="toggle">
          <input type="checkbox" checked=${llmOn} disabled=${!ready}
                 onChange=${e => setLlmOn(e.target.checked)}/>
          <span>Explain with AI</span>
        </label>
        <label class="sr-only" for="model">AI model</label>
        <select id="model" disabled=${!ready} value=${model}
                onChange=${e => setModel(e.target.value)}>
          ${ready
            ? models.map(n => html`<option key=${n} value=${n}>${n.length > 32 ? n.slice(0, 32) + '…' : n}</option>`)
            : html`<option value="">plain template</option>`}
        </select>
      </div>
    </div></header>`;
}

// ---- trust banner: the project's core promise, stated up front ----
function TrustBar() {
  return html`
    <div class="trustbar" role="note"><div class="trustinner wrap">
      <span class="tmark" aria-hidden="true">✓</span>
      <span class="ttext"><b>Every number is calculated exactly from the records.</b> The AI only
        rewords them — it never invents a figure.</span>
      <span class="tlegend" aria-hidden="true">
        <span class="lg num">🔢 computed</span>
        <span class="lg bg">📚 context</span>
        <span class="lg note">🧠 inferred</span>
      </span>
    </div></div>`;
}

function Chips({ suggestions, onPick }) {
  return html`
    <nav id="chips" class="wrap" aria-label="Example questions">
      ${suggestions.map(s => html`
        <button key=${s} type="button" class="chip" onClick=${() => onPick(s)}>${s}</button>`)}
    </nav>`;
}

// ---- chart: reuses chart.js; a Day / Month switch swaps the resampled series ----
function Chart({ chart }) {
  const grans = chart.views ? ['day', 'month'].filter(k => chart.views[k]) : [];
  const hasToggle = grans.length > 1;
  const [gran, setGran] = useState(chart.default || grans[0] || 'day');
  const view = chart.views && chart.views[gran];
  const c = view ? { ...chart, x: view.x, y: view.y } : chart;   // override the plotted series
  const cap = '📈 ' + chart.label
    + (chart.forecast ? ' — history + 14-day forecast'
      : chart.series2 ? ' — year-on-year comparison' : '');
  return html`
    <div class="chartwrap">
      <div class="chartcap">
        <span class="captext">${cap}</span>
        ${hasToggle && html`
          <span class="granctl" role="group" aria-label="Chart resolution">
            ${grans.map(k => html`
              <button key=${k} type="button" class=${gran === k ? 'active' : ''}
                      aria-pressed=${gran === k} onClick=${() => setGran(k)}>
                ${k === 'day' ? 'Day' : 'Month'}
              </button>`)}
          </span>`}
      </div>
      <div dangerouslySetInnerHTML=${{ __html: drawChart(c) + legendHtml(c) }}></div>
    </div>`;
}

function UserMessage({ text }) {
  return html`<div class="msg user"><div class="bubble">${text}</div></div>`;
}

function Thinking() {
  return html`
    <div class="msg bot"><div class="bubble">
      <span class="typing" aria-label="Thinking"><span></span><span></span><span></span></span>
    </div></div>`;
}

function BotMessage({ data }) {
  if (data.error) {
    return html`<div class="msg bot"><div class="bubble">
      <span class="err">⚠️ ${data.error}</span></div></div>`;
  }
  const llmOn = data.used_llm;
  // AI on: the prose is the answer, exact figures listed below it. AI off: the exact figure IS
  // the answer (shown as an instrument-style readout), so it isn't repeated below.
  const answer = llmOn ? data.text : (data.computed || data.text);
  const answerCls = llmOn ? 'answer' : 'answer exact';
  const how = llmOn ? 'explained by local AI' : 'exact answer from the data';
  const facts = [];
  if (llmOn && data.computed) facts.push(['num', '🔢', 'The exact figures', data.computed]);
  if (data.context) facts.push(['bg', '📚', 'Background', data.context]);
  if (data.inferred) facts.push(['note', '🧠', 'What it suggests', data.inferred]);
  const chart = data.chart;
  const showChart = chart && chart.x && chart.x.length > 1;
  return html`
    <div class="msg bot"><div class="bubble">
      <div class=${answerCls}>${answer}</div>
      ${data.resolved && html`<div class="tag">↳ understood as: “${data.resolved}”</div>`}
      ${facts.length > 0 && html`
        <div class="details">
          ${facts.map(([cls, ico, lab, val]) => html`
            <div key=${lab} class=${'fact ' + cls}>
              <span class="ico">${ico}</span>
              <span><span class="lab">${lab}:</span> <span class="val">${val}</span></span>
            </div>`)}
        </div>`}
      ${showChart && html`<${Chart} chart=${chart}/>`}
      <div class="tag"><span class="pill">${how}</span></div>
    </div></div>`;
}

function Chat({ messages, chatRef }) {
  return html`
    <main id="chat" ref=${chatRef} role="log" aria-live="polite" aria-atomic="false"
          aria-label="Conversation">
      ${messages.map((m, i) =>
        m.type === 'user' ? html`<${UserMessage} key=${i} text=${m.text}/>`
        : m.type === 'thinking' ? html`<${Thinking} key=${i}/>`
        : html`<${BotMessage} key=${i} data=${m.data}/>`)}
    </main>`;
}

function Composer({ q, setQ, onSend, busy, inputRef }) {
  return html`
    <footer><div class="composer wrap">
      <label class="sr-only" for="q">Ask a question in plain English</label>
      <input id="q" ref=${inputRef} value=${q} autocomplete="off"
             aria-label="Ask a question in plain English"
             placeholder="Ask in plain English…  e.g. “was 2013 hotter than 2014?”"
             onInput=${e => setQ(e.target.value)}
             onKeyDown=${e => { if (e.key === 'Enter') onSend(); }}/>
      <button id="send" type="button" disabled=${busy} onClick=${onSend}>Ask</button>
    </div></footer>`;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [convo, setConvo] = useState([]);      // recent {question, computed} turns for follow-ups
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [llmOn, setLlmOn] = useState(false);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('');
  const [aiAvailable, setAiAvailable] = useState(null);  // null while the status check is pending
  const [dsInfo, setDsInfo] = useState('German weather · Jena · 2009–2016 · loading…');
  const [suggestions, setSuggestions] = useState([]);
  const inputRef = useRef(null);
  const chatRef = useRef(null);

  useEffect(() => {
    fetch('/api/suggestions').then(r => r.json()).then(d => {
      setSuggestions(d.suggestions || []);
      setDsInfo(`German weather · Jena · 2009–2016 · ${(d.sensors || []).length} sensors`);
    }).catch(() => {});
    fetch('/api/models').then(r => r.json()).then(d => {
      setAiAvailable(!!d.available);
      if (d.available && d.models && d.models.length) { setModels(d.models); setModel(d.models[0]); }
    }).catch(() => setAiAvailable(false));
  }, []);

  // Keep the newest message in view.
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  async function ask(text) {
    text = (text || '').trim();
    if (!text || busy) return;
    setQ('');
    setMessages(m => [...m, { type: 'user', text }, { type: 'thinking' }]);
    setBusy(true);
    let data;
    try {
      const r = await fetch('/api/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, use_llm: llmOn, model, history: convo }),
      });
      data = await r.json();
    } catch (e) {
      data = { error: String(e) };
    }
    setMessages(m => m.slice(0, -1).concat({ type: 'bot', data }));   // replace the thinking bubble
    if (data && !data.error) {
      setConvo(c => [...c, { question: data.question, computed: data.computed || '' }].slice(-6));
    }
    setBusy(false);
    setTimeout(() => inputRef.current && inputRef.current.focus(), 0);
  }

  return html`
    <${Header} dsInfo=${dsInfo} aiAvailable=${aiAvailable} models=${models} model=${model}
               setModel=${setModel} llmOn=${llmOn} setLlmOn=${setLlmOn}/>
    <${TrustBar}/>
    <${Chips} suggestions=${suggestions} onPick=${ask}/>
    <${Chat} messages=${messages} chatRef=${chatRef}/>
    <${Composer} q=${q} setQ=${setQ} onSend=${() => ask(q)} busy=${busy} inputRef=${inputRef}/>`;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App}/>`);
