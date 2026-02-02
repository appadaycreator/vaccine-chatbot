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
  if (v === "green") return "緑: OK";
  if (v === "yellow") return "黄: 注意";
  return "赤: 要対応";
}

function pillClass(level) {
  const v = normalizeLevel(level);
  return `pill pill--${v}`;
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
  const who = role === "user" ? "あなた" : role === "assistant" ? "AI" : role === "error" ? "エラー" : "システム";
  const left = document.createElement("span");
  left.textContent = `${who} · ${nowTime()}`;
  const right = document.createElement("span");
  right.textContent = meta.model ? `model: ${meta.model}` : "";
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

  if (meta.sources && Array.isArray(meta.sources) && meta.sources.length) {
    const s = document.createElement("div");
    s.className = "sources";
    const parts = meta.sources
      .map((x) => {
        const page = x && x.page ? `p.${x.page}` : "";
        const src = x && x.source ? x.source : "";
        const label = [src, page].filter(Boolean).join(" ");
        return label ? `<code>${escapeHtml(label)}</code>` : null;
      })
      .filter(Boolean);
    s.innerHTML = `根拠: ${parts.join(" ")} `;
    div.appendChild(s);
  }

  if (meta.timings) {
    const t = meta.timings || {};
    const parts = [];
    if (typeof t.embedding_ms === "number") parts.push(`embedding ${t.embedding_ms}ms`);
    if (typeof t.search_ms === "number") parts.push(`search ${t.search_ms}ms`);
    if (typeof t.generate_ms === "number") parts.push(`generate ${t.generate_ms}ms`);
    if (typeof t.total_ms === "number") parts.push(`total ${t.total_ms}ms`);
    if (t.cached_embedding === true) parts.push("cache: hit");
    if (t.cached_embedding === false) parts.push("cache: miss");
    if (parts.length) {
      const tm = document.createElement("div");
      tm.className = "muted small";
      tm.textContent = `timings: ${parts.join(" / ")}`;
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
    const err = new Error(`HTTP ${res.status}`);
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
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${bodyText}`);
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
    const err = new Error(`HTTP ${res.status}`);
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
    throw new Error(`HTTP ${res.status}: ${detail}`);
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

function formatApiError(err) {
  const detail = err && err.detail ? err.detail : null;
  if (detail && typeof detail === "object") {
    const msg = detail.message || detail.error || JSON.stringify(detail);
    const stage = detail.stage ? `stage=${detail.stage}` : "";
    const code = detail.code ? `code=${detail.code}` : "";
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
  if (err && typeof err.status === "number") parts.push(`HTTP ${err.status}`);
  const msg = err && err.message ? String(err.message) : "";
  if (msg && !parts.includes(msg)) parts.push(msg);
  const detail = err && err.detail ? err.detail : null;
  if (detail && typeof detail === "string" && detail.trim()) parts.push(detail.trim());
  if (detail && typeof detail === "object") {
    const m = detail.message || detail.error;
    if (m) parts.push(String(m));
    const stage = detail.stage ? `stage=${detail.stage}` : "";
    const code = detail.code ? `code=${detail.code}` : "";
    const extra = [stage, code].filter(Boolean).join(" ");
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
  // 一度 404 を検出したらポーリングを止め、ユーザーが API Base URL を見直せるようにする。
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
    setStatus(`保存しました: ${v}`);
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
        sourcesStatusEl.textContent = `ソース一覧取得に失敗: ${data.error || ""}`;
        sourcesListEl.textContent = "";
        sourcesIndexingEl.textContent = "";
        sourcesErrorEl.textContent = "";
        sourcesNextActionsEl.textContent = "";
        setGuardReason("sources", true, "参照ソース情報を取得できませんでした（API Base URL を確認してください）。");
        return null;
      }
      const indexed = typeof data.indexed === "boolean" ? data.indexed : null;
      const indexing = data && data.indexing ? data.indexing : null;
      const running = !!(indexing && indexing.running === true);
      const last = data.last_indexed_at ? ` / 最終: ${String(data.last_indexed_at)}` : "";
      sourcesStatusEl.textContent =
        indexed === null
          ? `登録済みソース: ${data.items.length} 件${last}`
          : `登録済みソース: ${data.items.length} 件（${indexed ? "インデックス済" : "未インデックス"}）${last}`;

      // インデックス実行状況
      if (running) {
        const startedAt = indexing && indexing.started_at ? `開始: ${String(indexing.started_at)}` : "";
        sourcesIndexingEl.textContent = `再インデックス実行中… ${startedAt}`;
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

      // 送信ガード（未完了/実行中は送信不可）
      if (running) {
        setGuardReason("sources", true, "インデックス実行中のため、完了するまで送信できません。");
      } else if (indexed === false) {
        if (!data.items || !data.items.length) {
          setGuardReason("sources", true, "PDFが未配置のため送信できません。上の「次にやること」を実施してください。");
        } else {
          setGuardReason("sources", true, "インデックス未完了のため送信できません（必要に応じて「再インデックス」を実行）。");
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
            if (st === "ocr_needed") meta.push("要OCR（テキスト抽出不可）");
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
      sourcesStatusEl.textContent = `ソース一覧取得に失敗: ${e && e.message ? e.message : String(e)}`;
      sourcesListEl.textContent = "";
      sourcesIndexingEl.textContent = "";
      sourcesErrorEl.textContent = "";
      sourcesNextActionsEl.textContent = "";
      setGuardReason("sources", true, "参照ソース情報を取得できませんでした（API Base URL を確認してください）。");
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
        diagErrorEl.textContent = "環境チェックの取得に失敗しました。";
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
          "環境チェック（/diagnostics）が見つかりませんでした（HTTP 404）。",
          "",
          "まず確認すること:",
          "- API Base URL が FastAPI（uvicorn）を指しているか",
          "- そのURLで /health と /status が開けるか",
          "",
          "よくある原因:",
          "- cloudflared のURLが別サービス（例: Streamlit）に向いている",
          "- ポート番号/転送先が違う",
          "",
          "対処:",
          "- API Base URL を正しいURLに貼り替えて「保存」を押してください",
        ];
        diagErrorEl.innerHTML = escapeHtml(lines.join("\n")).replaceAll("\n", "<br>");
        diagUnsupportedFor = apiBase;
        if (diagPollId) {
          clearInterval(diagPollId);
          diagPollId = null;
        }
      } else {
        diagErrorEl.textContent = `環境チェックの取得に失敗: ${summarizeApiError(e)}`;
      }
      diagListEl.innerHTML = "";
      setGuardReason("diagnostics", false, "");
      return null;
    }
  }

  async function runReload() {
    const apiBase = normalizeApiBase(apiBaseEl.value);
    if (!apiBase) {
      setStatus("API Base URL を入力してください。", true);
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
    reloadBtn.textContent = "再インデックス中…";
    sourcesStatusEl.textContent = "再インデックス中…（しばらくお待ちください）";
    try {
      const res = await reloadIndex(apiBase);
      // 202 で「開始」なので、ここでは完了扱いしない
      setStatus((res && res.message) || "再インデックスを開始しました。完了までお待ちください。");
      const finalData = await waitForIndexCompletion();
      if (finalData && finalData.indexed === true) {
        setStatus("再インデックスが完了しました。");
      } else if (finalData && finalData.error) {
        const msg =
          finalData.error && typeof finalData.error === "object" && finalData.error.message
            ? String(finalData.error.message)
            : "再インデックスに失敗しました（エラー内容を確認してください）。";
        setStatus(msg, true);
      }
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      // 409: すでに実行中（サーバー側で多重実行防止）
      if (String(msg).includes("HTTP 409")) {
        setStatus("すでに再インデックスが実行中です。完了までお待ちください。", true);
        await waitForIndexCompletion();
      } else {
        setStatus(`再インデックス失敗: ${msg}`, true);
        await refreshSources();
      }
    } finally {
      const d = await refreshSources();
      const running = !!(d && d.indexing && d.indexing.running === true);
      if (!running) {
        reloadBtn.textContent = prevText || "再読み込み";
      } else {
        reloadBtn.textContent = "再インデックス中…";
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
    promptEl.disabled = blocked;
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

  async function sendPrompt(prompt, { isResend = false } = {}) {
    if (busy) return;
    const apiBase = normalizeApiBase(apiBaseEl.value);
    const model = (modelEl.value || "gemma2:2b").trim();
    const k = Number(kEl.value || 3);
    const p = String(prompt || "").trim();
    if (!apiBase) {
      addMessage("error", "API Base URL を入力してください。", {
        actions: [],
        details: null,
      });
      return;
    }
    if (guard && guard.blocked) {
      addMessage("error", guard.reason || "インデックス未完了のため送信できません。", {});
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
    setProgress("処理中: /chat（検索＋生成）…");
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
      addMessage("assistant", (chatRes && chatRes.answer) || "", {
        model,
        sources: chatRes && chatRes.sources ? chatRes.sources : null,
        timings,
      });
    } catch (err) {
      if (err && err.name === "AbortError") {
        addMessage("system", "キャンセルしました。", {});
      } else {
        const hints = extractApiErrorHints(err);
        const summary = summarizeApiError(err);
        const baseHints = [];
        if (err && err.status === 404) {
          baseHints.push(
            "API Base URL が FastAPI（uvicorn）を指しているか確認してください（例: cloudflared tunnel --url http://localhost:8000）。"
          );
        }
        if (err && err.status === 0) {
          baseHints.push("ネットワーク接続・CORS・API Base URL を確認してください。");
        }
        const allHints = [...baseHints, ...hints].filter(Boolean);
        const hintText = allHints.length ? "対処:\n- " + allHints.join("\n- ") : "対処:\n- API Base URL とサーバー状態（/health）を確認してください。";

        addMessage("error", `${summary}\n\n${hintText}`, {
          actions: [
            {
              label: "再送",
              kind: "secondary",
              disabled: !(lastUserPrompt || "").trim(),
              onClick: () => sendPrompt(lastUserPrompt, { isResend: true }),
            },
          ],
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
    "API Base URL を設定して、質問を送ってください。\n\n（cloudflaredのURLを貼ればGitHub Pagesからでも叩けます）"
  );

  refreshSources();
  refreshDiagnostics();
  // “動かない原因”がすぐ見えるように、軽くポーリング（失敗しても送信はブロックしない）
  diagPollId = setInterval(() => refreshDiagnostics(), 30000);
  updateComposerState();
}

main();

