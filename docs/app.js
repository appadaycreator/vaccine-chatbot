function $(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element: ${id}`);
  return el;
}

function nowTime() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeJsonStringify(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (e) {
    return String(obj);
  }
}

function normalizeLevel(level) {
  const v = String(level || "").toLowerCase();
  if (v === "green" || v === "yellow" || v === "red") return v;
  return "yellow";
}

function levelLabel(level) {
  const v = normalizeLevel(level);
  if (v === "green") return "緑: 問題なし";
  if (v === "yellow") return "黄: 注意";
  return "赤: 要対応";
}

function pillClass(level) {
  const v = normalizeLevel(level);
  return `pill pill--${v}`;
}

function stageLabel(stage) {
  const s = String(stage || "").toLowerCase();
  if (s === "embedding") return "embedding（質問をベクトル化）";
  if (s === "search") return "search（資料から検索）";
  if (s === "generate") return "generate（回答を生成）";
  if (s === "index_check") return "index_check（PDF差分チェック）";
  if (s === "index") return "index（再インデックス）";
  if (s === "chat") return "chat";
  if (s === "reload") return "reload";
  return stage ? String(stage) : "";
}

function parseApiFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const api = params.get("api");
  return api ? decodeURIComponent(api) : null;
}

function normalizeApiBase(input) {
  const v = (input || "").trim().replace(/\/+$/, "");
  if (!v) return "";
  return v;
}

function isSameOriginUi() {
  // FastAPI が docs/ を /ui 配下で配信する導線（同一オリジンで /chat 等を叩ける）
  const p = (typeof window !== "undefined" && window.location && window.location.pathname) || "";
  return p === "/ui" || p.startsWith("/ui/");
}

function saveApiBase(v) {
  localStorage.setItem("apiBase", v);
}

function loadApiBase() {
  const fromQuery = parseApiFromQuery();
  if (fromQuery) return normalizeApiBase(fromQuery);
  const fromStorage = localStorage.getItem("apiBase") || "";
  const normalized = normalizeApiBase(fromStorage);
  if (normalized) return normalized;

  // GitHub Pages（HTTPS）上で http://localhost:8000 を初期値にすると Mixed Content で失敗し、
  // コンソールが汚れるため、HTTPSのときは空にしてユーザー入力に委ねる。
  const host = (typeof window !== "undefined" && window.location && window.location.hostname) || "";
  const protocol = (typeof window !== "undefined" && window.location && window.location.protocol) || "";
  const isLocalHost = host === "localhost" || host === "127.0.0.1";
  if (protocol === "https:" && !isLocalHost) return "";
  return "http://localhost:8000";
}

const MODEL_KEY = "model.v1";

function saveModel(v) {
  try {
    localStorage.setItem(MODEL_KEY, String(v || "").trim());
  } catch {
    /* noop */
  }
}

function loadModel() {
  try {
    return String(localStorage.getItem(MODEL_KEY) || "").trim();
  } catch {
    return "";
  }
}

const DIAG_UNSUPPORTED_KEY = "diagUnsupportedApiBases.v1";

function loadDiagUnsupportedSet() {
  try {
    const raw = localStorage.getItem(DIAG_UNSUPPORTED_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return new Set();
    const out = new Set();
    for (const x of arr) {
      const v = normalizeApiBase(String(x || ""));
      if (v) out.add(v);
    }
    return out;
  } catch {
    return new Set();
  }
}

function saveDiagUnsupportedSet(set) {
  try {
    const arr = Array.from(set || []).map((x) => normalizeApiBase(String(x || ""))).filter(Boolean);
    // いたずらに膨らまないよう上限
    localStorage.setItem(DIAG_UNSUPPORTED_KEY, JSON.stringify(arr.slice(0, 50)));
  } catch {
    /* noop */
  }
}

function markDiagUnsupported(apiBase) {
  const v = normalizeApiBase(apiBase);
  if (!v) return;
  const set = loadDiagUnsupportedSet();
  set.add(v);
  saveDiagUnsupportedSet(set);
}

function unmarkDiagUnsupported(apiBase) {
  const v = normalizeApiBase(apiBase);
  if (!v) return;
  const set = loadDiagUnsupportedSet();
  if (!set.has(v)) return;
  set.delete(v);
  saveDiagUnsupportedSet(set);
}

function isDiagUnsupported(apiBase) {
  const v = normalizeApiBase(apiBase);
  if (!v) return false;
  const set = loadDiagUnsupportedSet();
  return set.has(v);
}

function renderMarkdown(md) {
  // marked があれば利用（なければプレーンテキスト）
  const markedLib = typeof window !== "undefined" ? window.marked : null;
  const src = String(md || "");
  if (!markedLib || typeof markedLib.parse !== "function") {
    return escapeHtml(src).replaceAll("\n", "<br>");
  }

  // 可能な範囲で安全側に寄せる（HTMLブロックはエスケープする）
  const renderer = new markedLib.Renderer();
  renderer.html = (html) => escapeHtml(html);
  renderer.link = (href, title, text) => {
    const safeHref = href ? String(href) : "";
    const safeText = text ? String(text) : safeHref;
    const t = title ? ` title="${escapeHtml(title)}"` : "";
    // target=_blank + rel を固定（クリックしても元画面が壊れにくい）
    return `<a href="${escapeHtml(safeHref)}"${t} target="_blank" rel="noopener noreferrer">${safeText}</a>`;
  };

  try {
    if (typeof markedLib.setOptions === "function") {
      markedLib.setOptions({ mangle: false, headerIds: false, renderer });
    }
    return markedLib.parse(src);
  } catch {
    return escapeHtml(src).replaceAll("\n", "<br>");
  }
}

function truncateText(s, maxLen = 80) {
  const t = String(s || "").trim();
  if (!t) return "";
  if (t.length <= maxLen) return t;
  return t.slice(0, Math.max(0, maxLen - 1)) + "…";
}

function uniqPreserve(arr) {
  const out = [];
  const seen = new Set();
  for (const x of Array.isArray(arr) ? arr : []) {
    const v = String(x || "").trim();
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function resolveWantedToInstalled(modelNames, wanted) {
  const names = Array.isArray(modelNames) ? modelNames : [];
  const w = String(wanted || "").trim();
  if (!w) return "";
  // 例: wanted=gemma2 でも installed に gemma2:latest しか無い場合がある
  const exact = names.find((n) => String(n) === w);
  if (exact) return String(exact);
  const prefix = names.find((n) => String(n).startsWith(w + ":"));
  if (prefix) return String(prefix);

  // 例: wanted=gemma2:2b だが installed は gemma2 / gemma2:latest のみ、のようなケースを救う
  // - UIは過去の選択（localStorage）を保持するため、タグ違いで「赤」が固定化しやすい
  if (w.includes(":")) {
    const base = w.split(":")[0];
    if (base) {
      const baseExact = names.find((n) => String(n) === base);
      if (baseExact) return String(baseExact);
      const basePrefix = names.find((n) => String(n).startsWith(base + ":"));
      if (basePrefix) return String(basePrefix);
    }
  }
  return "";
}

function isLikelyEmbeddingModel(name, embedModel) {
  const n = String(name || "");
  const lower = n.toLowerCase();
  const em = String(embedModel || "").trim();
  if (em && (n === em || n.startsWith(em + ":"))) return true;
  // “embed” を含むモデル名は埋め込み専用のことが多い（完全一致ではないが誤選択を避けたい）
  return lower.includes("embed") || lower.includes("embedding");
}

// 会話ログ（コピー用）
// - DOM（innerHTML）から復元すると Markdown/HTML の差分が出やすいため、追加時点の“生”を保持する
const chatLog = [];

async function copyToClipboard(text) {
  const t = String(text || "");
  if (!t) return false;
  try {
    if (navigator && navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(t);
      return true;
    }
  } catch {
    /* noop */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = t;
    ta.setAttribute("readonly", "readonly");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    ta.style.top = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return !!ok;
  } catch {
    return false;
  }
}

function normalizePageLabel(x) {
  if (x && typeof x.page_label === "string" && x.page_label.trim()) return x.page_label.trim();
  if (x && typeof x.page === "number" && Number.isFinite(x.page)) return `[P${x.page}]`;
  return "[P?]";
}

function sourceLocationLabel(x) {
  if (x && typeof x.location === "string" && x.location.trim()) return x.location.trim();
  const src = x && x.source ? String(x.source) : "資料";
  return `${src} ${normalizePageLabel(x)}`.trim();
}

function normalizeExcerpt(x) {
  const ex = x && typeof x.excerpt === "string" ? x.excerpt : "";
  return String(ex).replaceAll("\r\n", "\n").replaceAll("\r", "\n").trim();
}

function renderSourcesCard(sources) {
  const arr = Array.isArray(sources) ? sources : [];
  if (!arr.length) return null;

  const wrap = document.createElement("div");
  wrap.className = "sourcesCard";

  const head = document.createElement("div");
  head.className = "sourcesCard__head";
  const title = document.createElement("div");
  title.className = "sourcesCard__title";
  title.textContent = "回答の根拠（引用）";
  const tools = document.createElement("div");
  tools.className = "sourcesCard__tools";

  const copyAllBtn = document.createElement("button");
  copyAllBtn.type = "button";
  copyAllBtn.className = "btn btn--secondary sourcesCard__btn";
  copyAllBtn.textContent = "根拠をまとめてコピー";
  copyAllBtn.addEventListener("click", async () => {
    const lines = arr.map((x) => {
      const loc = sourceLocationLabel(x);
      const ex = normalizeExcerpt(x);
      return ex ? `${loc}\n${ex}` : `${loc}`;
    });
    await copyToClipboard(lines.join("\n\n"));
  });
  tools.appendChild(copyAllBtn);

  head.appendChild(title);
  head.appendChild(tools);
  wrap.appendChild(head);

  const list = document.createElement("div");
  list.className = "sourcesCard__list";

  for (const x of arr) {
    const loc = sourceLocationLabel(x);
    const ex = normalizeExcerpt(x);
    const preview = truncateText((ex.split("\n").find((l) => l.trim()) || ex || "").trim(), 90);

    const details = document.createElement("details");
    details.className = "sourceItem";
    const summary = document.createElement("summary");
    summary.className = "sourceItem__summary";

    const left = document.createElement("div");
    left.className = "sourceItem__summaryLeft";
    const locEl = document.createElement("div");
    locEl.className = "sourceItem__loc";
    locEl.textContent = loc;
    const pv = document.createElement("div");
    pv.className = "sourceItem__preview";
    pv.textContent = preview || "（抜粋が空です）";
    left.appendChild(locEl);
    left.appendChild(pv);

    const right = document.createElement("div");
    right.className = "sourceItem__summaryRight";
    const hint = document.createElement("div");
    hint.className = "muted small";
    hint.textContent = "クリックで展開";
    right.appendChild(hint);

    summary.appendChild(left);
    summary.appendChild(right);
    details.appendChild(summary);

    const body = document.createElement("div");
    body.className = "sourceItem__body";

    const actions = document.createElement("div");
    actions.className = "sourceItem__actions";

    const copyLocBtn = document.createElement("button");
    copyLocBtn.type = "button";
    copyLocBtn.className = "btn btn--secondary sourcesCard__btn";
    copyLocBtn.textContent = "資料名/ページをコピー";
    copyLocBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      await copyToClipboard(loc);
    });

    const copyQuoteBtn = document.createElement("button");
    copyQuoteBtn.type = "button";
    copyQuoteBtn.className = "btn btn--secondary sourcesCard__btn";
    copyQuoteBtn.textContent = "引用をコピー";
    copyQuoteBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      await copyToClipboard(ex ? `${loc}\n${ex}` : loc);
    });

    actions.appendChild(copyLocBtn);
    actions.appendChild(copyQuoteBtn);

    const pre = document.createElement("pre");
    pre.className = "sourceItem__excerpt";
    pre.textContent = ex || "（この根拠には抜粋がありません）";

    body.appendChild(actions);
    body.appendChild(pre);
    details.appendChild(body);

    list.appendChild(details);
  }

  wrap.appendChild(list);
  return wrap;
}

function addMessage(role, content, meta = {}) {
  const root = $("messages");
  const div = document.createElement("div");
  const cls =
    role === "user"
      ? "msg--user"
      : role === "assistant"
        ? "msg--assistant"
        : role === "error"
          ? "msg--error"
          : "msg--system";
  div.className = `msg ${cls}`;

  const metaDiv = document.createElement("div");
  metaDiv.className = "msg__meta";
  const who =
    role === "user" ? "あなた" : role === "assistant" ? "アシスタント" : role === "error" ? "エラー" : "システム";
  const left = document.createElement("span");
  const time = nowTime();
  left.textContent = `${who} · ${time}`;
  const right = document.createElement("span");
  right.textContent = meta.model ? `モデル: ${meta.model}` : "";
  metaDiv.appendChild(left);
  metaDiv.appendChild(right);

  const contentDiv = document.createElement("div");
  contentDiv.className = "msg__content";
  if (role === "assistant") {
    contentDiv.classList.add("msg__content--markdown");
    contentDiv.innerHTML = renderMarkdown(content);
  } else {
    contentDiv.innerHTML = escapeHtml(content);
  }

  div.appendChild(metaDiv);
  div.appendChild(contentDiv);

  if (meta && meta.actions && Array.isArray(meta.actions) && meta.actions.length) {
    const act = document.createElement("div");
    act.className = "msg__actions";
    for (const a of meta.actions) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `btn ${a.kind === "secondary" ? "btn--secondary" : ""}`.trim();
      btn.textContent = a.label || "実行";
      btn.disabled = !!a.disabled;
      btn.addEventListener("click", () => {
        try {
          a.onClick && a.onClick();
        } catch {
          /* noop */
        }
      });
      act.appendChild(btn);
    }
    div.appendChild(act);
  }

  if (meta && meta.details) {
    const details = document.createElement("details");
    details.className = "details";
    const summary = document.createElement("summary");
    summary.textContent = meta.detailsSummary || "詳細ログ（展開）";
    const pre = document.createElement("pre");
    pre.textContent = String(meta.details);
    details.appendChild(summary);
    details.appendChild(pre);
    div.appendChild(details);
  }

  if (meta && meta.sources && Array.isArray(meta.sources)) {
    const card = renderSourcesCard(meta.sources);
    if (card) div.appendChild(card);
  }

  if (meta.timings) {
    const t = meta.timings || {};
    const parts = [];
    if (typeof t.embedding_ms === "number") parts.push(`埋め込み: ${t.embedding_ms}ms`);
    if (typeof t.search_ms === "number") parts.push(`検索: ${t.search_ms}ms`);
    if (typeof t.generate_ms === "number") parts.push(`生成: ${t.generate_ms}ms`);
    if (typeof t.total_ms === "number") parts.push(`合計: ${t.total_ms}ms`);
    if (t.cached_embedding === true) parts.push("埋め込みキャッシュ: 利用");
    if (t.cached_embedding === false) parts.push("埋め込みキャッシュ: 未利用");
    if (parts.length) {
      const tm = document.createElement("div");
      tm.className = "muted small";
      tm.textContent = `処理時間: ${parts.join(" / ")}`;
      div.appendChild(tm);
    }
  }

  root.appendChild(div);
  root.scrollTop = root.scrollHeight;

  // copy用の会話ログを保持
  try {
    chatLog.push({
      role: String(role || ""),
      who,
      time,
      model: meta && meta.model ? String(meta.model) : "",
      content: String(content || ""),
      sources: meta && Array.isArray(meta.sources) ? meta.sources : null,
    });
  } catch {
    /* noop */
  }
}

async function postJson(url, payload, { signal } = {}) {
  let res, bodyText;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
    bodyText = await res.text();
  } catch (e) {
    const err = new Error("API に接続できません（ネットワークエラー）");
    err.status = 0;
    err.cause = e;
    err.detail = { message: err.message, hints: [apiConnectionHint(530)] };
    throw err;
  }
  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    body = { raw: bodyText };
  }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : bodyText;
    const err = new Error(`通信エラー（状態コード: ${res.status}）`);
    err.status = res.status;
    err.detail = typeof detail === "object" ? { ...detail } : { message: String(detail) };
    if (typeof err.detail !== "object") err.detail = { message: String(detail) };
    err.detail.hints = Array.isArray(err.detail.hints) ? err.detail.hints : [];
    if (res.status === 530) err.detail.hints.unshift(apiConnectionHint(530));
    err.body = body;
    throw err;
  }
  return body;
}

async function postSearch({ apiBase, prompt, k, signal }) {
  const url = `${apiBase}/search`;
  return await postJson(
    url,
    {
      prompt,
      k,
      // Cloudflare tunnel 経由だと、応答が遅いと 524 になり CORS に見えるため、
      // クライアント側で上限を短めにして、APIが先に JSON（504等）を返せるようにする。
      timeout_s: 90,
      embedding_timeout_s: 20,
      search_timeout_s: 20,
    },
    { signal }
  );
}

async function postGenerate({ apiBase, prompt, model, context, signal }) {
  const url = `${apiBase}/generate`;
  return await postJson(
    url,
    {
      prompt,
      model,
      context,
      max_tokens: 120,
      timeout_s: 90,
      generate_timeout_s: 45,
    },
    { signal }
  );
}

async function postChat({ apiBase, prompt, model, k, signal }) {
  const url = `${apiBase}/chat`;
  return await postJson(
    url,
    {
      prompt,
      model,
      k,
      max_tokens: 120,
      timeout_s: 90,
      embedding_timeout_s: 20,
      search_timeout_s: 20,
      generate_timeout_s: 45,
    },
    { signal }
  );
}

function apiConnectionHint(statusOrErr) {
  const status = typeof statusOrErr === "number" ? statusOrErr : statusOrErr?.status;
  if (status === 530 || status === 0) {
    return "530 や接続エラーのときは、API にリクエストが届いていません。cloudflared トンネルと API サーバーが起動しているか確認してください。CORS と表示されても原因はトンネル停止のことが多いです。";
  }
  return "APIのURLとネットワーク接続を確認してください。";
}

async function getSources(apiBase) {
  let res, bodyText;
  try {
    res = await fetch(`${apiBase}/sources`);
    bodyText = await res.text();
  } catch (e) {
    const err = new Error(`API に接続できません（ネットワークエラー）`);
    err.status = 0;
    err.cause = e;
    err.detail = { message: err.message, hints: [apiConnectionHint(530)] };
    throw err;
  }
  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    body = { raw: bodyText };
  }
  if (!res.ok) {
    const err = new Error(`通信エラー（状態コード: ${res.status}）`);
    err.status = res.status;
    err.detail = body && body.detail ? body.detail : bodyText;
    err.body = body;
    err.raw = bodyText;
    if (res.status === 530) {
      err.detail = typeof err.detail === "object" ? err.detail : {};
      if (typeof err.detail === "object") {
        err.detail.hints = err.detail.hints || [];
        err.detail.hints.unshift(apiConnectionHint(530));
      }
    }
    throw err;
  }
  return body;
}

async function getDiagnostics(apiBase, model) {
  const qs = model ? `?model=${encodeURIComponent(model)}` : "";
  let res, bodyText;
  try {
    res = await fetch(`${apiBase}/diagnostics${qs}`);
    bodyText = await res.text();
  } catch (e) {
    const err = new Error(`API に接続できません（ネットワークエラー）`);
    err.status = 0;
    err.cause = e;
    err.detail = { message: err.message, hints: [apiConnectionHint(530)] };
    throw err;
  }
  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    body = { raw: bodyText };
  }
  if (!res.ok) {
    const err = new Error(`通信エラー（状態コード: ${res.status}）`);
    err.status = res.status;
    err.body = body;
    err.raw = bodyText;
    err.detail = err.detail || (body && body.detail) || bodyText;
    if (res.status === 530) {
      const d = typeof err.detail === "object" ? { ...err.detail } : { message: String(err.detail) };
      d.hints = Array.isArray(d.hints) ? d.hints : [];
      d.hints.unshift(apiConnectionHint(530));
      err.detail = d;
    }
    throw err;
  }
  return body;
}

async function reloadIndex(apiBase) {
  const res = await fetch(`${apiBase}/reload`, { method: "POST" });
  const bodyText = await res.text();
  let body;
  try {
    body = JSON.parse(bodyText);
  } catch {
    body = { raw: bodyText };
  }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : bodyText;
    const err = new Error(`通信エラー（状態コード: ${res.status}）`);
    err.status = res.status;
    err.detail = detail;
    err.body = body;
    err.raw = bodyText;
    throw err;
  }
  return body;
}

function setStatus(text, isError = false) {
  // 既存の仕様互換（軽い通知）。重大エラーは addMessage("error", ...) を使う
  addMessage(isError ? "error" : "system", String(text || ""), {});
}

function setProgress(text, isError = false) {
  const el = $("progress");
  el.textContent = text || "";
  el.classList.toggle("error", !!isError);
}

function formatElapsedMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return "";
  const sec = n / 1000;
  if (sec < 10) return `${sec.toFixed(1)}秒`;
  return `${Math.round(sec)}秒`;
}

function loadLastChatTimings() {
  try {
    const raw = localStorage.getItem("lastChatTimings.v1");
    if (!raw) return null;
    const obj = JSON.parse(raw);
    return obj && typeof obj === "object" ? obj : null;
  } catch {
    return null;
  }
}

function saveLastChatTimings(timings) {
  try {
    const t = timings && typeof timings === "object" ? timings : null;
    if (!t) return;
    const out = {
      embedding_ms: typeof t.embedding_ms === "number" ? t.embedding_ms : null,
      search_ms: typeof t.search_ms === "number" ? t.search_ms : null,
      generate_ms: typeof t.generate_ms === "number" ? t.generate_ms : null,
      total_ms: typeof t.total_ms === "number" ? t.total_ms : null,
    };
    localStorage.setItem("lastChatTimings.v1", JSON.stringify(out));
  } catch {
    /* noop */
  }
}

function estimateChatStageBudgetsMs({ lastTimings, timeouts } = {}) {
  const last = lastTimings && typeof lastTimings === "object" ? lastTimings : {};
  const t = timeouts && typeof timeouts === "object" ? timeouts : {};

  const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
  const n = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);

  const lastEmbed = n(last.embedding_ms);
  const lastSearch = n(last.search_ms);
  const lastGen = n(last.generate_ms);

  // “前回+少し余裕” を基本にして工程切替を推定（初回は固定値）
  const embedBudget = clamp(Math.round(lastEmbed ? lastEmbed * 1.3 : 1200), 700, 15000);
  const searchBudget = clamp(Math.round(lastSearch ? lastSearch * 1.3 : 1200), 700, 20000);
  const genBudget = clamp(Math.round(lastGen ? lastGen * 1.3 : 3500), 900, 45000);

  // UI表示の上限がタイムアウトを超えないよう抑える（表示が不自然に長引かないように）
  const etMs = typeof t.embedding_timeout_s === "number" ? Math.round(t.embedding_timeout_s * 1000) : null;
  const stMs = typeof t.search_timeout_s === "number" ? Math.round(t.search_timeout_s * 1000) : null;
  const gtMs = typeof t.generate_timeout_s === "number" ? Math.round(t.generate_timeout_s * 1000) : null;

  return {
    embedding_ms: etMs ? Math.min(embedBudget, Math.max(1200, Math.round(etMs * 0.8))) : embedBudget,
    search_ms: stMs ? Math.min(searchBudget, Math.max(1200, Math.round(stMs * 0.8))) : searchBudget,
    generate_ms: gtMs ? Math.min(genBudget, Math.max(1800, Math.round(gtMs * 0.85))) : genBudget,
  };
}

function startChatStageProgress({ lastTimings, timeouts, onUpdate } = {}) {
  const budgets = estimateChatStageBudgetsMs({ lastTimings, timeouts });
  const t0 = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();

  function stageFromElapsed(elapsedMs) {
    if (elapsedMs < budgets.embedding_ms) return "embedding";
    if (elapsedMs < budgets.embedding_ms + budgets.search_ms) return "search";
    return "generate";
  }

  function makeText(stage, elapsedMs) {
    const el = formatElapsedMs(elapsedMs);
    const base = `処理中: ${stageLabel(stage)}…`;
    const suffix = el ? `（${el}）` : "";
    const long =
      elapsedMs > budgets.embedding_ms + budgets.search_ms + budgets.generate_ms
        ? " 長い場合は k を下げる／軽量モデルを試すのが有効です。"
        : "";
    return `${base}${suffix}${long}`;
  }

  let id = null;
  const tick = () => {
    const now = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
    const elapsedMs = Math.max(0, now - t0);
    const stage = stageFromElapsed(elapsedMs);
    try {
      onUpdate && onUpdate(makeText(stage, elapsedMs), false, { stage, elapsedMs });
    } catch {
      /* noop */
    }
  };

  tick();
  id = setInterval(tick, 250);
  return () => {
    if (id) clearInterval(id);
  };
}

function formatApiError(err) {
  const detail = err && err.detail ? err.detail : null;
  if (detail && typeof detail === "object") {
    let msg = detail.message || detail.error || "";
    const cause =
      (detail.extra && typeof detail.extra.error === "string" ? detail.extra.error : detail.error) != null
        ? String(detail.extra?.error ?? detail.error).trim()
        : "";
    if (cause) msg = msg ? `${msg}\n原因: ${cause}` : `原因: ${cause}`;
    if (!msg) msg = JSON.stringify(detail);
    const stage = detail.stage ? `段階: ${stageLabel(detail.stage)}` : "";
    const code = detail.code ? `コード: ${detail.code}` : "";
    const head = [msg, stage, code].filter(Boolean).join(" / ");
    const hints =
      Array.isArray(detail.hints) && detail.hints.length
        ? "\n対処:\n" + detail.hints.map((x) => `- ${x}`).join("\n")
        : "";
    return `${head}${hints}`;
  }
  return err && err.message ? err.message : String(err);
}

function extractApiErrorHints(err) {
  const detail = err && err.detail ? err.detail : null;
  if (detail && typeof detail === "object") {
    if (Array.isArray(detail.hints) && detail.hints.length) return detail.hints.map((x) => String(x));
  }
  return [];
}

function summarizeApiError(err) {
  const parts = [];
  if (err && typeof err.status === "number") parts.push(`状態コード: ${err.status}`);
  const msg = err && err.message ? String(err.message) : "";
  if (msg && !parts.includes(msg)) parts.push(msg);
  const detail = err && err.detail ? err.detail : null;
  if (detail && typeof detail === "string" && detail.trim()) parts.push(detail.trim());
  if (detail && typeof detail === "object") {
    const m = detail.message || detail.error;
    if (m) parts.push(String(m));
    const stage = detail.stage ? `段階: ${stageLabel(detail.stage)}` : "";
    const code = detail.code ? `コード: ${detail.code}` : "";
    const extra = [stage, code].filter(Boolean).join(" / ");
    if (extra) parts.push(extra);
    // サーバーが返す実際の例外（SEARCH_ERROR 等の原因を表示。API は extra を detail にマージするため detail.error にも入る）
    const rawCause =
      detail.extra && typeof detail.extra.error === "string" ? detail.extra.error : detail.error;
    const cause = rawCause != null ? String(rawCause).trim() : "";
    if (cause) parts.push(`原因: ${cause}`);
  }
  const text = parts.filter(Boolean).join(" / ");
  return text || "不明なエラー";
}

function buildConversationLogText({ includeSources = true } = {}) {
  const items = Array.isArray(chatLog) ? chatLog : [];
  const parts = [];
  for (const m of items) {
    const who = m && m.who ? String(m.who) : "メッセージ";
    const time = m && m.time ? String(m.time) : "";
    const model = m && m.model ? String(m.model) : "";
    const head = `${who}${time ? ` · ${time}` : ""}${model ? ` · モデル: ${model}` : ""}`.trim();
    const body = String((m && m.content) || "").replaceAll("\r\n", "\n").replaceAll("\r", "\n").trim();
    let text = head;
    if (body) text += `\n${body}`;

    if (includeSources && m && Array.isArray(m.sources) && m.sources.length) {
      const locs = m.sources
        .map((x) => sourceLocationLabel(x))
        .map((s) => String(s || "").trim())
        .filter(Boolean);
      if (locs.length) {
        text += `\n\n根拠（引用）:\n- ${locs.join("\n- ")}`;
      }
    }
    parts.push(text.trim());
  }
  return parts.filter(Boolean).join("\n\n---\n\n");
}

function main() {
  const apiBaseEl = $("apiBase");
  const modelEl = $("model");
  const kEl = $("k");
  const promptEl = $("prompt");
  const sendBtn = $("send");
  const resendBtn = $("resend");
  const cancelBtn = $("cancel");
  const copyChatLogBtn = document.getElementById("copyChatLog");
  const saveBtn = $("saveApiBase");
  const reloadBtn = $("reloadIndex");
  const sourcesStatusEl = $("sourcesStatus");
  const sourcesIndexingEl = $("sourcesIndexing");
  const sourcesErrorEl = $("sourcesError");
  const sourcesNextActionsEl = $("sourcesNextActions");
  const sourcesListEl = $("sourcesList");
  const guardEl = $("guard");
  const diagOverallEl = $("diagOverall");
  const diagSummaryEl = $("diagSummary");
  const diagErrorEl = $("diagError");
  const diagListEl = $("diagList");

  const sameOriginUi = isSameOriginUi();
  if (sameOriginUi) {
    // /ui では「GitHub Pages 前提」の文言が混乱しやすいので、表示を寄せる
    try {
      document.title = "vaccine-chatbot（/ui）";
    } catch {
      /* noop */
    }
    try {
      const subtitle = document.querySelector(".header__title p.muted");
      if (subtitle) subtitle.textContent = "API配下のチャットUI（/ui）";
    } catch {
      /* noop */
    }
  }

  // ファーストビュー改善（アコーディオン要約）
  const connectAccordionEl = document.getElementById("connectAccordion");
  const apiBaseSummaryEl = document.getElementById("apiBaseSummary");
  const settingsSummaryEl = document.getElementById("settingsSummary");

  function updateApiBaseSummary() {
    if (!apiBaseSummaryEl) return;
    const v = sameOriginUi ? "" : normalizeApiBase(apiBaseEl.value);
    apiBaseSummaryEl.textContent = v ? truncateText(v, 44) : "未設定";
  }

  function updateSettingsSummary() {
    if (!settingsSummaryEl) return;
    const m = (modelEl && modelEl.value ? String(modelEl.value) : "").trim();
    const k = (kEl && kEl.value ? String(kEl.value) : "").trim();
    settingsSummaryEl.textContent = `モデル: ${m || "未選択"} / k=${k || "?"}`;
  }

  // /diagnostics が未実装のAPI（またはAPI以外）を指すと 404 が定期的に出るため、
  // 一度 404 を検出したらポーリングを止め、ユーザーが APIのURL を見直せるようにする。
  let diagPollId = null;
  let diagUnsupportedFor = null;
  let lastModelOptionsKey = "";

  // スキップリンク（「チャット入力へ移動」）: 画面スクロールだけでなく入力欄へフォーカスも当てる。
  // 特にモバイルでは “スクロールだけ” だとキーボードが出ない/入力開始できないことがある。
  function focusPrompt({ smooth = true } = {}) {
    // focus({preventScroll}) が使える環境ではスクロールを制御しやすい
    try {
      promptEl.focus({ preventScroll: true });
    } catch {
      try {
        promptEl.focus();
      } catch {
        /* noop */
      }
    }
    try {
      promptEl.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "center" });
    } catch {
      try {
        promptEl.scrollIntoView();
      } catch {
        /* noop */
      }
    }
  }

  const skipToPrompt = document.querySelector('a.skip-link[href="#prompt"]');
  if (skipToPrompt) {
    skipToPrompt.addEventListener("click", (e) => {
      // “#promptへ移動”を保ちつつ、確実に入力できる状態にする
      e.preventDefault();
      focusPrompt({ smooth: true });
      try {
        history.replaceState(null, "", "#prompt");
      } catch {
        /* noop */
      }
    });
  }
  // URLが最初から #prompt の場合も、入力欄へ寄せる（ただしキーボード表示はブラウザ仕様次第）
  if (typeof window !== "undefined" && window.location && window.location.hash === "#prompt") {
    setTimeout(() => focusPrompt({ smooth: false }), 0);
  }
  if (typeof window !== "undefined") {
    window.addEventListener("hashchange", () => {
      if (window.location && window.location.hash === "#prompt") {
        focusPrompt({ smooth: true });
      }
    });
  }

  if (copyChatLogBtn) {
    copyChatLogBtn.addEventListener("click", async () => {
      const text = buildConversationLogText({ includeSources: true });
      if (!text.trim()) {
        setStatus("コピーする会話ログがありません。", true);
        return;
      }
      const ok = await copyToClipboard(text);
      if (ok) {
        setStatus("会話ログをクリップボードにコピーしました。");
      } else {
        setStatus("会話ログをコピーできませんでした（ブラウザの権限/HTTPS等を確認してください）。", true);
      }
    });
  }

  if (sameOriginUi) {
    // /ui では同一オリジンで叩くため、API URL入力は不要（混乱を避ける）
    apiBaseEl.value = "";
    try {
      const card = apiBaseEl.closest("section");
      if (card) card.style.display = "none";
    } catch {
      /* noop */
    }
  } else {
    apiBaseEl.value = loadApiBase();
    updateApiBaseSummary();
    // 未設定のときは「接続先」を開いておく（初回導線）
    try {
      if (connectAccordionEl) connectAccordionEl.open = !normalizeApiBase(apiBaseEl.value);
    } catch {
      /* noop */
    }
    // 前回 404 だったURLは、ページ再読み込み時の“1回目の404”すら出さない（静かなUX）
    {
      const apiBase = normalizeApiBase(apiBaseEl.value);
      if (apiBase && isDiagUnsupported(apiBase)) {
        diagUnsupportedFor = apiBase;
      }
    }
  }
  // モデル選択は環境差が大きいので、localStorage（前回選択）を優先する
  {
    const saved = loadModel();
    if (saved && modelEl) {
      // options がまだ無い（/diagnostics で後から埋まる）ケースに備えて保持
      try {
        modelEl.dataset.savedModel = saved;
      } catch {
        /* noop */
      }
      try {
        modelEl.value = saved;
      } catch {
        /* noop */
      }
    }
  }
  updateSettingsSummary();

  // 入力中にも要約を更新して、いま何を設定しているかが分かるようにする
  apiBaseEl.addEventListener("input", () => updateApiBaseSummary());
  kEl.addEventListener("input", () => updateSettingsSummary());
  kEl.addEventListener("change", () => updateSettingsSummary());

  saveBtn.addEventListener("click", () => {
    if (sameOriginUi) return;
    const v = normalizeApiBase(apiBaseEl.value);
    saveApiBase(v);
    updateApiBaseSummary();
    try {
      // 設定が済んだら閉じて、チャットに集中できるようにする
      if (connectAccordionEl && v) connectAccordionEl.open = false;
    } catch {
      /* noop */
    }
    // “同じURLだがサーバーが更新された”ケースに備えて、保存時は一度だけ再判定できるよう解除
    unmarkDiagUnsupported(v);
    diagUnsupportedFor = null;
    setStatus(`保存しました（APIのURL）: ${v}`);
    refreshSources();
    refreshDiagnostics();
    // 404 でポーリング停止していた場合に備えて復帰させる
    if (!diagPollId) {
      diagPollId = setInterval(() => refreshDiagnostics(), 30000);
    }
  });

  modelEl.addEventListener("change", () => {
    saveModel(modelEl.value);
    updateSettingsSummary();
    refreshDiagnostics();
  });

  async function refreshSources() {
    const apiBase = sameOriginUi ? "" : normalizeApiBase(apiBaseEl.value);
    if (!sameOriginUi && !apiBase) return;
    try {
      const data = await getSources(apiBase);
      if (!data.ok) {
        sourcesStatusEl.textContent = `参照ソースの取得に失敗しました。`;
        const lines = [
          `要点: 参照ソースの取得に失敗しました（${data.error || "詳細不明"}）`,
          "",
          "対処:",
          "- APIのURLが正しいか確認してください",
          "- そのURLで /health と /status が開けるか確認してください",
        ];
        sourcesErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
        sourcesListEl.textContent = "";
        sourcesIndexingEl.textContent = "";
        sourcesNextActionsEl.textContent = "";
        setGuardReason("sources", true, "参照ソース情報を取得できませんでした（APIのURLを確認してください）。");
        return null;
      }
      const indexed = typeof data.indexed === "boolean" ? data.indexed : null;
      const indexing = data && data.indexing ? data.indexing : null;
      const running = !!(indexing && indexing.running === true);
      const last = data.last_indexed_at ? ` / 最終: ${String(data.last_indexed_at)}` : "";
      sourcesStatusEl.textContent =
        indexed === null
          ? `登録済みソース: ${data.items.length} 件${last}`
          : `登録済みソース: ${data.items.length} 件（${indexed ? "検索の準備: 完了" : "検索の準備: 未完了"}）${last}`;

      // インデックス実行状況
      if (running) {
        const startedAt = indexing && indexing.started_at ? `開始: ${String(indexing.started_at)}` : "";
        sourcesIndexingEl.textContent = `反映処理中… ${startedAt}`;
      } else {
        sourcesIndexingEl.textContent = "";
      }

      // エラー表示（直近の index エラー or init_error）
      const errObj = data.error || null;
      const initErr = data.init_error ? String(data.init_error) : "";
      if (errObj && typeof errObj === "object") {
        const msg = errObj.message ? String(errObj.message) : JSON.stringify(errObj);
        const hints =
          Array.isArray(errObj.hints) && errObj.hints.length ? "\n対処:\n" + errObj.hints.map((x) => `- ${x}`).join("\n") : "";
        sourcesErrorEl.innerHTML = escapeHtml(`${msg}${hints}`).replaceAll("\n", "<br>");
      } else if (initErr) {
        sourcesErrorEl.innerHTML = escapeHtml(initErr).replaceAll("\n", "<br>");
      } else {
        sourcesErrorEl.textContent = "";
      }

      // 次にやること
      const next = Array.isArray(data.next_actions) ? data.next_actions : [];
      sourcesNextActionsEl.innerHTML = next.length
        ? escapeHtml("次にやること:\n- " + next.join("\n- ")).replaceAll("\n", "<br>")
        : "";

      // 送信ガード（準備中/実行中は送信不可）
      if (running) {
        setGuardReason("sources", true, "資料の反映処理中のため、完了するまで送信できません。");
      } else if (indexed === false) {
        if (!data.items || !data.items.length) {
          setGuardReason("sources", true, "PDFが未配置のため送信できません。上の「次にやること」を実施してください。");
        } else {
          setGuardReason("sources", true, "検索の準備が未完了のため送信できません（必要に応じて「反映する」を実行）。");
        }
      } else {
        setGuardReason("sources", false, "");
      }

      // サーバー側が実行中なら、再インデックスボタンも抑止
      reloadBtn.disabled = running;

      if (!data.items.length) {
        sourcesListEl.innerHTML = `<div class="muted small">（まだ登録されていません）</div>`;
        return data;
      }
      const rows = data.items
        .map((it) => {
          const type = it && it.type ? String(it.type) : "";
          const original = it && it.original_filename ? String(it.original_filename) : "";
          const saved = it && it.filename ? String(it.filename) : "";
          const label = original || saved || (it && it.path ? String(it.path) : "");
          const showSaved = false;
          const meta = [];
          if (showSaved) meta.push(`保存名: ${saved}`);
          if (it && typeof it.size_bytes === "number") meta.push(`${Math.round(it.size_bytes / 1024)}KB`);
          if (it && it.ingest && typeof it.ingest === "object") {
            const st = it.ingest.status ? String(it.ingest.status) : "";
            if (st === "ocr_needed") meta.push("文字認識（OCR）が必要（文字を取り出せません）");
            if (st === "error") meta.push("取込失敗");
          }
          const metaHtml = meta.length
            ? `<span class="muted small source-meta">（${escapeHtml(meta.join(" / "))}）</span>`
            : "";
          return `<li class="source-item"><code>${escapeHtml(label)}</code>${metaHtml}</li>`;
        })
        .join("");
      sourcesListEl.innerHTML = `<ul class="sources-list">${rows}</ul>`;
      return data;
    } catch (e) {
      sourcesStatusEl.textContent = `参照ソースの取得に失敗しました。`;
      const hintLines =
        Array.isArray(e?.detail?.hints) && e.detail.hints.length
          ? e.detail.hints.map((h) => `- ${h}`)
          : ["- APIのURLが正しいか確認してください", "- そのURLで /health と /status が開けるか確認してください"];
      const lines = [
        `要点: 参照ソースの取得に失敗しました（${summarizeApiError(e)}）`,
        "",
        "対処:",
        ...hintLines,
      ];
      sourcesListEl.textContent = "";
      sourcesIndexingEl.textContent = "";
      sourcesErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
      sourcesNextActionsEl.textContent = "";
      setGuardReason("sources", true, "参照ソース情報を取得できませんでした（APIのURLを確認してください）。");
      return null;
    }
  }

  async function refreshDiagnostics({ skipModelSync = false } = {}) {
    const apiBase = sameOriginUi ? "" : normalizeApiBase(apiBaseEl.value);
    const selectedModel = (modelEl && modelEl.value ? String(modelEl.value) : "").trim();
    const savedModel = (modelEl && modelEl.dataset && modelEl.dataset.savedModel ? String(modelEl.dataset.savedModel) : "").trim();
    const model = selectedModel || savedModel || null;
    if (!sameOriginUi && !apiBase) return null;
    if (diagUnsupportedFor === apiBase) return null;
    try {
      const data = await getDiagnostics(apiBase, model);
      if (!data || data.ok !== true) {
        diagOverallEl.className = "pill pill--yellow";
        diagOverallEl.textContent = "未確認";
        diagSummaryEl.textContent = "";
        const lines = [
          "要点: 環境チェックの取得に失敗しました。",
          "",
          "対処:",
          "- APIのURLが正しいか確認してください",
          "- そのURLで /health と /status が開けるか確認してください",
        ];
        diagErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
        diagListEl.innerHTML = "";
        setGuardReason("diagnostics", false, "");
        return null;
      }

      // /diagnostics の返却（インストール済みモデル一覧）を元に、UIのモデル選択を自動更新する。
      // - “未インストール固定”で赤になり続けるのを避ける
      // - 埋め込みモデル（例: nomic-embed-text）は候補から除外する
      if (!skipModelSync && modelEl) {
        const allNames =
          data && data.models && Array.isArray(data.models.names) ? uniqPreserve(data.models.names.map((x) => String(x))) : [];
        const embedModel = data && data.meta && data.meta.embed_model ? String(data.meta.embed_model) : "";
        const llmNames = allNames.filter((n) => !isLikelyEmbeddingModel(n, embedModel));
        if (llmNames.length) {
          const key = llmNames.join("|");
          if (key !== lastModelOptionsKey) {
            lastModelOptionsKey = key;
            // options を総入替（インストール済みのみに限定）
            try {
              const frag = document.createDocumentFragment();
              for (const n of llmNames) {
                const opt = document.createElement("option");
                opt.value = n;
                opt.textContent = n;
                frag.appendChild(opt);
              }
              modelEl.innerHTML = "";
              modelEl.appendChild(frag);
            } catch {
              /* noop */
            }
          }

          const current = String(modelEl.value || "").trim();
          const requested = data && data.meta && data.meta.requested_model ? String(data.meta.requested_model) : "";
          const next =
            resolveWantedToInstalled(llmNames, current) ||
            resolveWantedToInstalled(llmNames, savedModel) ||
            resolveWantedToInstalled(llmNames, requested) ||
            llmNames[0];

          if (next && current !== next) {
            try {
              modelEl.value = next;
            } catch {
              /* noop */
            }
            try {
              modelEl.dataset.savedModel = next;
            } catch {
              /* noop */
            }
            saveModel(next);
          }

          // モデルが変わった場合は、赤/黄表示を即時に更新するために再診断する
          if (next && (model || "") !== next) {
            return await refreshDiagnostics({ skipModelSync: true });
          }
        }
      }

      const overall = data.overall || {};
      const level = normalizeLevel(overall.level);
      diagOverallEl.className = pillClass(level);
      diagOverallEl.textContent = levelLabel(level);
      const modelCount = data.models && typeof data.models.count === "number" ? data.models.count : null;
      const summary = overall.summary ? String(overall.summary) : "";
      diagSummaryEl.textContent = modelCount === null ? summary : `${summary} / モデル: ${modelCount}件`;
      diagErrorEl.textContent = "";

      const checks = Array.isArray(data.checks) ? data.checks : [];
      const metaOllamaHost = data && data.meta && data.meta.ollama_host ? String(data.meta.ollama_host) : "";
      const metaOllamaVersion = data && data.meta && data.meta.ollama_version ? String(data.meta.ollama_version) : "";
      diagListEl.innerHTML = checks
        .map((c) => {
          const lv = normalizeLevel(c && c.level);
          const label = c && c.label ? String(c.label) : "チェック";
          const msg = c && c.message ? String(c.message) : "";
          const hints = Array.isArray(c && c.hints) ? c.hints.map((x) => String(x)) : [];
          // 疎通不良の切り分けに重要なので、接続先を補足表示する
          if ((label === "Ollama疎通" || label === "モデル一覧") && metaOllamaHost) {
            hints.push(`接続先（OLLAMA_HOST）: ${metaOllamaHost}`);
          }
          if (label === "Ollama疎通" && metaOllamaVersion) {
            hints.push(`Ollama version: ${metaOllamaVersion}`);
          }
          const hintsText = hints.length ? "対処:\n- " + hints.join("\n- ") : "";
          return `
            <li class="diag-item">
              <div class="diag-item__top">
                <div class="diag-item__label">${escapeHtml(label)}</div>
                <span class="${pillClass(lv)}">${escapeHtml(levelLabel(lv))}</span>
              </div>
              <div class="diag-item__msg">${escapeHtml(msg)}</div>
              ${
                hintsText
                  ? `<div class="diag-item__hints">${escapeHtml(hintsText)}</div>`
                  : ""
              }
            </li>
          `;
        })
        .join("");

      const red = checks.find((c) => normalizeLevel(c && c.level) === "red");
      if (level === "red" && red) {
        const label = red && red.label ? String(red.label) : "環境チェック";
        const msg = red && red.message ? String(red.message) : "要対応";
        setGuardReason("diagnostics", true, `環境チェックで問題があります: ${label} / ${msg}`);
      } else {
        setGuardReason("diagnostics", false, "");
      }
      updateSettingsSummary();
      return data;
    } catch (e) {
      // 旧API（/diagnostics未実装）もあり得るので、取得失敗では送信をブロックしない
      const status = e && typeof e.status === "number" ? e.status : null;
      diagOverallEl.className = "pill pill--yellow";
      diagOverallEl.textContent = "未確認";
      diagSummaryEl.textContent = "";
      if (status === 404) {
        // 通常運用の導線: URLが FastAPI 以外（例: Streamlit / 静的ホスティング等）を指しているケースが多い
        // “未対応”ではなく、ユーザーが最短で復旧できるように「何を直すか」を表示する
        const lines = [
          "環境チェック（/diagnostics）が見つかりませんでした（状態コード: 404）。",
          "",
          "まず確認すること:",
          "- APIのURLが FastAPI（uvicorn）を指しているか",
          "- そのURLで /health と /status が開けるか",
          "",
          "よくある原因:",
          "- cloudflared のURLが別サービス（例: Streamlit）に向いている",
          "- ポート番号/転送先が違う",
          "",
          "対処:",
          "- APIのURLを正しいURLに貼り替えて「保存」を押してください",
        ];
        diagErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
        // /ui（同一オリジン）は URL入力が無いので、404抑止の永続化は不要
        diagUnsupportedFor = apiBase;
        if (!sameOriginUi) {
          // 次回のページ再読み込みでも404を出さないように記録
          markDiagUnsupported(apiBase);
        }
        if (diagPollId) {
          clearInterval(diagPollId);
          diagPollId = null;
        }
      } else {
        const hintLines =
          Array.isArray(e?.detail?.hints) && e.detail.hints.length
            ? e.detail.hints.map((h) => `- ${h}`)
            : [
                "- APIのURLが正しいか確認してください",
                "- そのURLで /health と /status が開けるか確認してください",
                "- 一時的な問題なら、少し待ってからもう一度お試しください",
              ];
        const lines = [
          `要点: 環境チェックの取得に失敗しました（${summarizeApiError(e)}）`,
          "",
          "対処:",
          ...hintLines,
        ];
        diagErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
      }
      diagListEl.innerHTML = "";
      setGuardReason("diagnostics", false, "");
      return null;
    }
  }

  async function runReload() {
    const apiBase = normalizeApiBase(apiBaseEl.value);
    if (!apiBase) {
      setStatus("APIのURLを入力してください。", true);
      return;
    }
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    async function waitForIndexCompletion() {
      let delay = 1500;
      // 最大 10 分はポーリング（重いPDFでも自己解決できるように待つ）
      const deadline = Date.now() + 10 * 60 * 1000;
      while (Date.now() < deadline) {
        const data = await refreshSources();
        const running = !!(data && data.indexing && data.indexing.running === true);
        if (!running) return data;
        await sleep(delay);
        delay = Math.min(Math.round(delay * 1.4), 8000);
      }
      return await refreshSources();
    }

    const prevText = reloadBtn.textContent;
    reloadBtn.disabled = true;
    reloadBtn.textContent = "反映中…";
    sourcesStatusEl.textContent = "反映中…（しばらくお待ちください）";
    try {
      const res = await reloadIndex(apiBase);
      // 202 で「開始」なので、ここでは完了扱いしない
      setStatus((res && res.message) || "反映処理を開始しました。完了までお待ちください。");
      const finalData = await waitForIndexCompletion();
      if (finalData && finalData.indexed === true) {
        setStatus("反映が完了しました。");
      } else if (finalData && finalData.error) {
        const msg =
          finalData.error && typeof finalData.error === "object" && finalData.error.message
            ? String(finalData.error.message)
            : "反映に失敗しました（エラー内容を確認してください）。";
        setStatus(msg, true);
      }
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      // 409: すでに実行中（サーバー側で多重実行防止）
      if (e && typeof e.status === "number" && e.status === 409) {
        setStatus("すでに反映処理が実行中です。完了までお待ちください。", true);
        await waitForIndexCompletion();
      } else {
        setStatus(`反映に失敗しました: ${msg}`, true);
        await refreshSources();
      }
    } finally {
      const d = await refreshSources();
      const running = !!(d && d.indexing && d.indexing.running === true);
      if (!running) {
        reloadBtn.textContent = prevText || "反映する";
      } else {
        reloadBtn.textContent = "反映中…";
      }
      // disabled は refreshSources() が（実行中なら）制御する
    }
  }

  reloadBtn.addEventListener("click", () => {
    runReload();
  });

  let currentController = null;
  let busy = false;
  let lastUserPrompt = "";
  let guard = { blocked: false, reason: "" };
  let guardReasons = { sources: null, diagnostics: null };

  function recomputeGuard() {
    const order = ["sources", "diagnostics"];
    for (const k of order) {
      const g = guardReasons[k];
      if (g && g.blocked) {
        guard = { blocked: true, reason: String(g.reason || "") };
        return;
      }
    }
    guard = { blocked: false, reason: "" };
  }

  function updateComposerState() {
    const blocked = !!busy || !!guard.blocked;
    sendBtn.disabled = blocked;
    // disabled にするとフォーカスが飛びやすいので、送信中のみ readOnly にする
    promptEl.disabled = false;
    promptEl.readOnly = !!busy;
    cancelBtn.hidden = !busy;
    cancelBtn.disabled = !busy;
    resendBtn.disabled = blocked || !(lastUserPrompt || "").trim();
    guardEl.textContent = guard.blocked ? guard.reason : "";
  }

  function setBusy(v) {
    busy = !!v;
    updateComposerState();
  }

  function setGuardReason(key, blocked, reason) {
    const k = String(key || "");
    guardReasons[k] = blocked ? { blocked: true, reason: String(reason || "") } : null;
    recomputeGuard();
    updateComposerState();
  }

  cancelBtn.addEventListener("click", () => {
    if (currentController) {
      currentController.abort();
    }
  });

  // 入力履歴（↑↓）
  const HISTORY_KEY = "promptHistory.v1";
  const HISTORY_LIMIT = 50;
  let history = [];
  let historyIndex = -1;
  let historyDraft = "";

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      if (Array.isArray(arr)) return arr.map((x) => String(x)).filter((x) => x.trim());
    } catch {
      /* noop */
    }
    return [];
  }

  function saveHistory(next) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next.slice(0, HISTORY_LIMIT)));
    } catch {
      /* noop */
    }
  }

  function pushHistory(prompt) {
    const p = String(prompt || "").trim();
    if (!p) return;
    history = [p, ...history.filter((x) => x !== p)].slice(0, HISTORY_LIMIT);
    saveHistory(history);
    historyIndex = -1;
    historyDraft = "";
  }

  history = loadHistory();

  function buildRewriteSuggestions(q) {
    const base = String(q || "").trim();
    if (!base) return [];
    // UIテンプレ（LLMは使わず、固定の言い換え促しで次アクションを提示）
    return [
      `${base}（対象ワクチン名・接種後の日数・症状/数値など条件を入れて、資料の記載を確認したい）`,
      `「${base}」について、資料の見出し/項目名（チェックリスト名・表の項目名）を教えてください`,
      `資料に「${base}」の類語（別の言い方）があれば教えてください（資料の用語で質問したい）`,
    ];
  }

  async function sendPrompt(prompt, { isResend = false } = {}) {
    if (busy) return;
    const apiBase = sameOriginUi ? "" : normalizeApiBase(apiBaseEl.value);
    const model =
      String(modelEl.value || "").trim() ||
      (() => {
        try {
          const opts = modelEl && modelEl.options ? Array.from(modelEl.options) : [];
          const first = opts.map((o) => String((o && o.value) || "").trim()).find(Boolean);
          return first || "";
        } catch {
          return "";
        }
      })();
    const k = Number(kEl.value || 3);
    const p = String(prompt || "").trim();
    if (!sameOriginUi && !apiBase) {
      addMessage("error", "APIのURLを入力してください。", {
        actions: [],
        details: null,
      });
      return;
    }
    if (!model) {
      addMessage("error", "モデルを選択してください（環境チェックでインストール済みモデルを自動反映します）。", {});
      return;
    }
    if (guard && guard.blocked) {
      addMessage("error", guard.reason || "検索の準備が未完了のため送信できません。", {});
      return;
    }
    if (!p) return;

    if (!sameOriginUi) saveApiBase(apiBase);
    lastUserPrompt = p;
    updateComposerState();

    // UI: ユーザーメッセージを先に出す
    addMessage("user", p, isResend ? { model: "", timings: null } : {});
    pushHistory(p);

    setBusy(true);
    const chatTimeouts = { embedding_timeout_s: 240, search_timeout_s: 120, generate_timeout_s: 240 };
    const stopProgress = startChatStageProgress({
      lastTimings: loadLastChatTimings(),
      timeouts: chatTimeouts,
      onUpdate: (text, isError) => setProgress(text, !!isError),
    });
    currentController = new AbortController();

    try {
      const chatRes = await postJson(
        `${apiBase}/chat`,
        {
          prompt: p,
          model,
          k,
          max_tokens: 120,
          timeout_s: 240,
          embedding_timeout_s: 240,
          search_timeout_s: 120,
          generate_timeout_s: 240,
        },
        { signal: currentController.signal }
      );
      const timings = Object.assign({}, chatRes.timings || {});
      if (typeof timings.total_ms !== "number") {
        const a = typeof timings.embedding_ms === "number" ? timings.embedding_ms : 0;
        const b = typeof timings.search_ms === "number" ? timings.search_ms : 0;
        const c = typeof timings.generate_ms === "number" ? timings.generate_ms : 0;
        timings.total_ms = a + b + c;
      }
      const sources = chatRes && Array.isArray(chatRes.sources) ? chatRes.sources : null;
      const candidates = chatRes && Array.isArray(chatRes.candidates) ? chatRes.candidates : null;
      const noSources = Array.isArray(sources) && sources.length === 0;
      const ans = (chatRes && chatRes.answer) || "";

      // 回答本文は「普通の会話文」にしつつ、元ソース（資料の抜粋）を会話内にそのまま表示する。
      // - 参照箇所は API の sources[].excerpt を使用（PDFから抽出したテキスト）
      // - クリック展開しなくても見えるよう、回答の直下に Markdown で差し込む
      const quoteMd = (() => {
        const arr = Array.isArray(sources) && sources.length ? sources : Array.isArray(candidates) ? candidates : [];
        const items = arr
          .map((s) => {
            const loc = sourceLocationLabel(s);
            const ex = normalizeExcerpt(s);
            if (!ex) return null;
            const q = ex
              .split("\n")
              .map((ln) => ln.trim())
              .filter(Boolean)
              .map((ln) => `> ${ln}`)
              .join("\n");
            // Markdownレンダラ側でHTMLはサニタイズしているが、locは念のためテキストとして扱う
            return `- **${loc}**\n\n${q}`;
          })
          .filter(Boolean);
        if (!items.length) return "";
        // 見出しを付けて分かりやすくする（固定フォーマットではなく、単なる付随情報として表示）
        const title = Array.isArray(sources) && sources.length ? "引用（資料の抜粋）" : "検索候補（該当箇所なし）";
        return `\n\n---\n\n**${title}**\n\n${items.join("\n\n")}`;
      })();

      const ansWithQuotes = `${String(ans || "").trim()}${quoteMd}`.trim();
      const looksLikeRefusal = noSources && /資料に記載がないため、この資料に基づく回答はできません/.test(String(ans));
      const actions = looksLikeRefusal
        ? buildRewriteSuggestions(p).map((text) => ({
            label: `言い換え案: ${truncateText(text, 16)}`,
            kind: "secondary",
            onClick: () => {
              promptEl.value = text;
              promptEl.focus();
            },
          }))
        : [];
      addMessage("assistant", ansWithQuotes, {
        model,
        sources,
        timings,
        actions: actions.length ? actions : null,
      });
      saveLastChatTimings(timings);
    } catch (err) {
      try {
        stopProgress && stopProgress();
      } catch {
        /* noop */
      }
      if (err && err.name === "AbortError") {
        addMessage("system", "キャンセルしました。", {});
      } else {
        const detail = err && err.detail && typeof err.detail === "object" ? err.detail : null;
        const stage = detail && detail.stage ? String(detail.stage) : "";
        const code = detail && detail.code ? String(detail.code) : "";
        const hints = extractApiErrorHints(err);
        const summary = summarizeApiError(err);
        const baseHints = [];
        if (err && err.status === 404) {
          baseHints.push(
            "APIのURLが FastAPI（uvicorn）を指しているか確認してください（例: cloudflared tunnel --url http://localhost:8000）。"
          );
        }
        if (err && err.status === 0) {
          baseHints.push("ネットワーク接続・CORS・APIのURLを確認してください。");
        }

        const stageHints = [];
        if (stage) {
          const st = String(stage).toLowerCase();
          if (st === "search") {
            stageHints.push("検索が遅い場合は k を小さくすると改善することがあります（例: 3→2→1）。");
          } else if (st === "generate") {
            stageHints.push("生成が遅い場合は軽量モデル（例: gemma2:2b など）に切り替えると改善することがあります。");
          } else if (st === "embedding") {
            stageHints.push("embedding が遅い/失敗する場合は Ollama 起動・Embeddingモデルの有無を確認してください。");
          } else if (st === "index_check" || st === "index") {
            stageHints.push("PDF追加直後は差分チェック/再インデックスで重くなることがあります。少し待って再送してください。");
          }
        }

        const allHints = [...baseHints, ...stageHints, ...hints].filter(Boolean);
        const hintText = allHints.length
          ? "対処:\n- " + allHints.join("\n- ")
          : "対処:\n- APIのURLとサーバー状態（/health）を確認してください。";

        const actions = [
          {
            label: "再送",
            kind: "secondary",
            disabled: !(lastUserPrompt || "").trim(),
            onClick: () => sendPrompt(lastUserPrompt, { isResend: true }),
          },
        ];

        // タイムアウト等の“自己解決”導線（stageベース）
        const st = stage ? String(stage).toLowerCase() : "";
        if (st === "search") {
          actions.unshift(
            {
              label: "k=2 にして再送",
              kind: "secondary",
              disabled: !(lastUserPrompt || "").trim(),
              onClick: () => {
                try {
                  kEl.value = "2";
                } catch {
                  /* noop */
                }
                sendPrompt(lastUserPrompt, { isResend: true });
              },
            },
            {
              label: "k=1 にして再送",
              kind: "secondary",
              disabled: !(lastUserPrompt || "").trim(),
              onClick: () => {
                try {
                  kEl.value = "1";
                } catch {
                  /* noop */
                }
                sendPrompt(lastUserPrompt, { isResend: true });
              },
            }
          );
        }
        if (st === "generate") {
          const firstModel = (() => {
            try {
              const opts = modelEl && modelEl.options ? Array.from(modelEl.options) : [];
              const first = opts.map((o) => String((o && o.value) || "").trim()).find(Boolean);
              return first || "";
            } catch {
              return "";
            }
          })();
          if (firstModel) {
            actions.unshift({
              label: `モデルを ${firstModel} にして再送`,
              kind: "secondary",
              disabled: !(lastUserPrompt || "").trim(),
              onClick: () => {
                try {
                  modelEl.value = firstModel;
                } catch {
                  /* noop */
                }
                saveModel(firstModel);
                sendPrompt(lastUserPrompt, { isResend: true });
              },
            });
          }
        }

        const stageInfo = stage ? `詰まった工程: ${stageLabel(stage)}${code ? ` / code=${code}` : ""}` : "";
        addMessage("error", `${summary}${stageInfo ? `\n${stageInfo}` : ""}\n\n${hintText}`, {
          actions,
          details: safeJsonStringify({
            status: err && err.status ? err.status : null,
            message: err && err.message ? err.message : null,
            detail: err && err.detail ? err.detail : null,
            body: err && err.body ? err.body : null,
            raw: err,
          }),
          detailsSummary: "ログ全文（展開）",
        });
      }
    } finally {
      setBusy(false);
      try {
        stopProgress && stopProgress();
      } catch {
        /* noop */
      }
      setProgress("");
      currentController = null;
    }
  }

  $("chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = (promptEl.value || "").trim();
    promptEl.value = "";
    await sendPrompt(prompt, { isResend: false });
  });

  resendBtn.addEventListener("click", () => {
    if (!lastUserPrompt) return;
    sendPrompt(lastUserPrompt, { isResend: true });
  });

  // Enter送信 / Shift+Enter改行 / 履歴（↑↓）
  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      // IME確定中のEnter等は無視したいが、ここでは素朴に扱う
      e.preventDefault();
      if (!busy && !guard.blocked) {
        $("chatForm").requestSubmit();
      }
      return;
    }

    if (e.key === "ArrowUp") {
      if (busy) return;
      const atTop = promptEl.selectionStart === 0 && promptEl.selectionEnd === 0;
      if (!atTop) return;
      if (historyIndex === -1) historyDraft = promptEl.value || "";
      if (!history.length) return;
      historyIndex = Math.min(historyIndex + 1, history.length - 1);
      promptEl.value = history[historyIndex] || "";
      // caretを末尾へ
      promptEl.selectionStart = promptEl.selectionEnd = promptEl.value.length;
      e.preventDefault();
      return;
    }

    if (e.key === "ArrowDown") {
      if (busy) return;
      const atBottom = promptEl.selectionStart === promptEl.value.length && promptEl.selectionEnd === promptEl.value.length;
      if (!atBottom) return;
      if (historyIndex === -1) return;
      historyIndex = historyIndex - 1;
      if (historyIndex < 0) {
        historyIndex = -1;
        promptEl.value = historyDraft || "";
      } else {
        promptEl.value = history[historyIndex] || "";
      }
      promptEl.selectionStart = promptEl.selectionEnd = promptEl.value.length;
      e.preventDefault();
      return;
    }
  });

  addMessage(
    "assistant",
    sameOriginUi
      ? "この画面は API（FastAPI）と同じURLで動作します。接続先の設定は不要なので、そのまま質問を送ってください。"
      : "APIのURLを設定してから、質問を送ってください。\n\n（cloudflared の HTTPS URL を貼れば、GitHub Pages からでも利用できます）"
  );

  refreshSources();
  refreshDiagnostics();
  // “動かない原因”がすぐ見えるように、軽くポーリング（失敗しても送信はブロックしない）
  diagPollId = setInterval(() => refreshDiagnostics(), 30000);
  updateComposerState();
}

main();

