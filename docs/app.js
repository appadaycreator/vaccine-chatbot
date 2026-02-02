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
  return normalizeApiBase(fromStorage) || "http://localhost:8000";
}

function addMessage(role, content, meta = {}) {
  const root = $("messages");
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "msg--user" : "msg--assistant"}`;

  const metaDiv = document.createElement("div");
  metaDiv.className = "msg__meta";
  const who = role === "user" ? "あなた" : "AI";
  const left = document.createElement("span");
  left.textContent = `${who} · ${nowTime()}`;
  const right = document.createElement("span");
  right.textContent = meta.model ? `model: ${meta.model}` : "";
  metaDiv.appendChild(left);
  metaDiv.appendChild(right);

  const contentDiv = document.createElement("div");
  contentDiv.className = "msg__content";
  contentDiv.innerHTML = escapeHtml(content);

  div.appendChild(metaDiv);
  div.appendChild(contentDiv);

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
  const root = $("messages");
  const p = document.createElement("div");
  p.className = `muted small ${isError ? "error" : ""}`;
  p.textContent = text;
  root.appendChild(p);
  root.scrollTop = root.scrollHeight;
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

function main() {
  const apiBaseEl = $("apiBase");
  const modelEl = $("model");
  const kEl = $("k");
  const promptEl = $("prompt");
  const sendBtn = $("send");
  const cancelBtn = $("cancel");
  const saveBtn = $("saveApiBase");
  const reloadBtn = $("reloadIndex");
  const sourcesStatusEl = $("sourcesStatus");
  const sourcesIndexingEl = $("sourcesIndexing");
  const sourcesErrorEl = $("sourcesError");
  const sourcesNextActionsEl = $("sourcesNextActions");
  const sourcesListEl = $("sourcesList");
  const guardEl = $("guard");

  apiBaseEl.value = loadApiBase();
  // M2 Mac mini (8GB) では軽量モデルをデフォルトにする
  if (modelEl && !modelEl.value) {
    modelEl.value = "gemma2:2b";
  }
  saveBtn.addEventListener("click", () => {
    const v = normalizeApiBase(apiBaseEl.value);
    saveApiBase(v);
    setStatus(`保存しました: ${v}`);
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
        setGuard(true, "参照ソース情報を取得できませんでした（API Base URL を確認してください）。");
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
        setGuard(true, "インデックス実行中のため、完了するまで送信できません。");
      } else if (indexed === false) {
        if (!data.items || !data.items.length) {
          setGuard(true, "PDFが未配置のため送信できません。上の「次にやること」を実施してください。");
        } else {
          setGuard(true, "インデックス未完了のため送信できません（必要に応じて「再インデックス」を実行）。");
        }
      } else {
        setGuard(false, "");
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
      setGuard(true, "参照ソース情報を取得できませんでした（API Base URL を確認してください）。");
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
  let guard = { blocked: false, reason: "" };

  function updateComposerState() {
    const blocked = !!busy || !!guard.blocked;
    sendBtn.disabled = blocked;
    promptEl.disabled = blocked;
    cancelBtn.hidden = !busy;
    cancelBtn.disabled = !busy;
    guardEl.textContent = guard.blocked ? guard.reason : "";
  }

  function setBusy(v) {
    busy = !!v;
    updateComposerState();
  }

  function setGuard(blocked, reason) {
    guard = { blocked: !!blocked, reason: blocked ? String(reason || "") : "" };
    updateComposerState();
  }

  cancelBtn.addEventListener("click", () => {
    if (currentController) {
      currentController.abort();
    }
  });

  $("chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const apiBase = normalizeApiBase(apiBaseEl.value);
    const prompt = (promptEl.value || "").trim();
    const model = (modelEl.value || "gemma2:2b").trim();
    const k = Number(kEl.value || 3);
    if (!apiBase) {
      setStatus("API Base URL を入力してください。", true);
      return;
    }
    if (guard && guard.blocked) {
      setStatus(guard.reason || "インデックス未完了のため送信できません。", true);
      return;
    }
    if (!prompt) return;

    saveApiBase(apiBase);
    promptEl.value = "";
    addMessage("user", prompt);
    setStatus("送信しました。");
    setBusy(true);
    setProgress("処理中: /chat（検索＋生成）…");
    currentController = new AbortController();

    try {
      const chatRes = await postChat({ apiBase, prompt, model, k, signal: currentController.signal });
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
        setStatus("キャンセルしました。");
      } else {
        // 404 の場合は「別ポート/別サービスを tunnel している」ことが多いので、ヒントを出す
        if (err && err.status === 404) {
          setStatus(
            `エラー: ${formatApiError(err)}\nAPI Base URL が FastAPI（uvicorn）を指しているか確認してください（例: cloudflared tunnel --url http://localhost:8000）。`,
            true
          );
        } else {
          setStatus(`エラー: ${formatApiError(err)}`, true);
        }
      }
    }
    setBusy(false);
    setProgress("");
    currentController = null;
  });

  addMessage(
    "assistant",
    "API Base URL を設定して、質問を送ってください。\n（cloudflaredのURLを貼ればGitHub Pagesからでも叩けます）"
  );

  refreshSources();
}

main();

