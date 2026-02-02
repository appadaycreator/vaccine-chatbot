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
  left.textContent = `${who} · ${nowTime()}`;
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
}

async function postJson(url, payload, { signal } = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
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
      timeout_s: 180,
      embedding_timeout_s: 240,
      search_timeout_s: 120,
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
      timeout_s: 240,
      generate_timeout_s: 240,
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
      timeout_s: 240,
      embedding_timeout_s: 240,
      search_timeout_s: 120,
      generate_timeout_s: 240,
    },
    { signal }
  );
}

async function getSources(apiBase) {
  const res = await fetch(`${apiBase}/sources`);
  const bodyText = await res.text();
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
    throw err;
  }
  return body;
}

async function getDiagnostics(apiBase, model) {
  const qs = model ? `?model=${encodeURIComponent(model)}` : "";
  const res = await fetch(`${apiBase}/diagnostics${qs}`);
  const bodyText = await res.text();
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
    const msg = detail.message || detail.error || JSON.stringify(detail);
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
  }
  const text = parts.filter(Boolean).join(" / ");
  return text || "不明なエラー";
}

function main() {
  const apiBaseEl = $("apiBase");
  const modelEl = $("model");
  const kEl = $("k");
  const promptEl = $("prompt");
  const sendBtn = $("send");
  const resendBtn = $("resend");
  const cancelBtn = $("cancel");
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

  // /diagnostics が未実装のAPI（またはAPI以外）を指すと 404 が定期的に出るため、
  // 一度 404 を検出したらポーリングを止め、ユーザーが APIのURL を見直せるようにする。
  let diagPollId = null;
  let diagUnsupportedFor = null;

  apiBaseEl.value = loadApiBase();
  // M2 Mac mini (8GB) では軽量モデルをデフォルトにする
  if (modelEl && !modelEl.value) {
    modelEl.value = "gemma2:2b";
  }
  saveBtn.addEventListener("click", () => {
    const v = normalizeApiBase(apiBaseEl.value);
    saveApiBase(v);
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
    refreshDiagnostics();
  });

  async function refreshSources() {
    const apiBase = normalizeApiBase(apiBaseEl.value);
    if (!apiBase) return;
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
      const lines = [
        `要点: 参照ソースの取得に失敗しました（${summarizeApiError(e)}）`,
        "",
        "対処:",
        "- APIのURLが正しいか確認してください",
        "- そのURLで /health と /status が開けるか確認してください",
      ];
      sourcesListEl.textContent = "";
      sourcesIndexingEl.textContent = "";
      sourcesErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
      sourcesNextActionsEl.textContent = "";
      setGuardReason("sources", true, "参照ソース情報を取得できませんでした（APIのURLを確認してください）。");
      return null;
    }
  }

  async function refreshDiagnostics() {
    const apiBase = normalizeApiBase(apiBaseEl.value);
    const model = (modelEl && modelEl.value ? String(modelEl.value) : "gemma2:2b").trim() || "gemma2:2b";
    if (!apiBase) return null;
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

      const overall = data.overall || {};
      const level = normalizeLevel(overall.level);
      diagOverallEl.className = pillClass(level);
      diagOverallEl.textContent = levelLabel(level);
      const modelCount = data.models && typeof data.models.count === "number" ? data.models.count : null;
      const summary = overall.summary ? String(overall.summary) : "";
      diagSummaryEl.textContent = modelCount === null ? summary : `${summary} / モデル: ${modelCount}件`;
      diagErrorEl.textContent = "";

      const checks = Array.isArray(data.checks) ? data.checks : [];
      diagListEl.innerHTML = checks
        .map((c) => {
          const lv = normalizeLevel(c && c.level);
          const label = c && c.label ? String(c.label) : "チェック";
          const msg = c && c.message ? String(c.message) : "";
          const hints = Array.isArray(c && c.hints) ? c.hints.map((x) => String(x)) : [];
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
        diagUnsupportedFor = apiBase;
        if (diagPollId) {
          clearInterval(diagPollId);
          diagPollId = null;
        }
      } else {
        const lines = [
          `要点: 環境チェックの取得に失敗しました（${summarizeApiError(e)}）`,
          "",
          "対処:",
          "- APIのURLが正しいか確認してください",
          "- そのURLで /health と /status が開けるか確認してください",
          "- 一時的な問題なら、少し待ってからもう一度お試しください",
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
    const apiBase = normalizeApiBase(apiBaseEl.value);
    const model = (modelEl.value || "gemma2:2b").trim();
    const k = Number(kEl.value || 3);
    const p = String(prompt || "").trim();
    if (!apiBase) {
      addMessage("error", "APIのURLを入力してください。", {
        actions: [],
        details: null,
      });
      return;
    }
    if (guard && guard.blocked) {
      addMessage("error", guard.reason || "検索の準備が未完了のため送信できません。", {});
      return;
    }
    if (!p) return;

    saveApiBase(apiBase);
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
      const chatRes = await postChat({ apiBase, prompt: p, model, k, signal: currentController.signal });
      const timings = Object.assign({}, chatRes.timings || {});
      if (typeof timings.total_ms !== "number") {
        const a = typeof timings.embedding_ms === "number" ? timings.embedding_ms : 0;
        const b = typeof timings.search_ms === "number" ? timings.search_ms : 0;
        const c = typeof timings.generate_ms === "number" ? timings.generate_ms : 0;
        timings.total_ms = a + b + c;
      }
      const sources = chatRes && Array.isArray(chatRes.sources) ? chatRes.sources : null;
      const noSources = Array.isArray(sources) && sources.length === 0;
      const actions = noSources
        ? buildRewriteSuggestions(p).map((text) => ({
            label: `言い換え案: ${truncateText(text, 16)}`,
            kind: "secondary",
            onClick: () => {
              promptEl.value = text;
              promptEl.focus();
            },
          }))
        : [];
      addMessage("assistant", (chatRes && chatRes.answer) || "", {
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
            stageHints.push("生成が遅い場合は軽量モデル（例: gemma2:2b）に切り替えると改善することがあります。");
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
          actions.unshift({
            label: "モデルを gemma2:2b にして再送",
            kind: "secondary",
            disabled: !(lastUserPrompt || "").trim(),
            onClick: () => {
              try {
                modelEl.value = "gemma2:2b";
              } catch {
                /* noop */
              }
              sendPrompt(lastUserPrompt, { isResend: true });
            },
          });
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

  // 例: よくある質問（ボタン）
  const quick = $("quickQuestions");
  const quickItems = [
    "接種後7日間に記録する項目は？",
    "37.5度以上の発熱が出たらどうすればいい？",
    "接種部位の腫れ・痛みはどのくらい続く？（資料にある範囲で）",
    "相談先（医療機関/自治体/119）の判断の目安は？",
  ];
  quick.innerHTML = quickItems
    .map((t) => `<button type="button" class="btn btn--secondary" data-q="${escapeHtml(t)}">例: ${escapeHtml(t)}</button>`)
    .join("");
  quick.addEventListener("click", (e) => {
    const btn = e.target && e.target.closest ? e.target.closest("button[data-q]") : null;
    if (!btn) return;
    const q = btn.getAttribute("data-q") || "";
    // data属性はHTMLエスケープ済みなので復元せずそのまま（記号類も表示上は問題ない想定）
    // 実際に送る値としては textContent の方が安全
    const text = btn.textContent ? btn.textContent.replace(/^例:\s*/, "") : q;
    promptEl.value = text;
    promptEl.focus();
    if (!busy && !guard.blocked) {
      $("chatForm").requestSubmit();
    }
  });

  addMessage(
    "assistant",
    "APIのURLを設定してから、質問を送ってください。\n\n（cloudflared の HTTPS URL を貼れば、GitHub Pages からでも利用できます）"
  );

  refreshSources();
  refreshDiagnostics();
  // “動かない原因”がすぐ見えるように、軽くポーリング（失敗しても送信はブロックしない）
  diagPollId = setInterval(() => refreshDiagnostics(), 30000);
  updateComposerState();
}

main();

