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

  root.appendChild(div);
  root.scrollTop = root.scrollHeight;
}

async function postChat({ apiBase, prompt, model, k }) {
  const url = `${apiBase}/chat`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model, k, max_tokens: 120, timeout_s: 120 }),
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
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return body;
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

async function uploadPdf(apiBase, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${apiBase}/sources/upload`, { method: "POST", body: form });
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

function main() {
  const apiBaseEl = $("apiBase");
  const modelEl = $("model");
  const kEl = $("k");
  const promptEl = $("prompt");
  const saveBtn = $("saveApiBase");
  const pdfFileEl = $("pdfFile");
  const uploadBtn = $("uploadPdf");
  const sourcesStatusEl = $("sourcesStatus");
  const sourcesListEl = $("sourcesList");

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
        return;
      }
      sourcesStatusEl.textContent = `登録済みソース: ${data.items.length} 件`;
      if (!data.items.length) {
        sourcesListEl.innerHTML = `<div class="muted small">（まだ登録されていません）</div>`;
        return;
      }
      const rows = data.items
        .map((it) => {
          const type = it && it.type ? String(it.type) : "";
          const original = it && it.original_filename ? String(it.original_filename) : "";
          const saved = it && it.filename ? String(it.filename) : "";
          const label = original || saved || (it && it.path ? String(it.path) : "");
          const showSaved = type === "upload" && saved && original && saved !== original;
          const meta = [];
          if (showSaved) meta.push(`保存名: ${saved}`);
          if (it && typeof it.size_bytes === "number") meta.push(`${Math.round(it.size_bytes / 1024)}KB`);
          const metaHtml = meta.length
            ? `<span class="muted small source-meta">（${escapeHtml(meta.join(" / "))}）</span>`
            : "";
          return `<li class="source-item"><code>${escapeHtml(label)}</code>${metaHtml}</li>`;
        })
        .join("");
      sourcesListEl.innerHTML = `<ul class="sources-list">${rows}</ul>`;
    } catch (e) {
      sourcesStatusEl.textContent = `ソース一覧取得に失敗: ${e && e.message ? e.message : String(e)}`;
      sourcesListEl.textContent = "";
    }
  }

  uploadBtn.addEventListener("click", async () => {
    const apiBase = normalizeApiBase(apiBaseEl.value);
    if (!apiBase) {
      setStatus("API Base URL を入力してください。", true);
      return;
    }
    const f = pdfFileEl.files && pdfFileEl.files[0] ? pdfFileEl.files[0] : null;
    if (!f) {
      setStatus("アップロードするPDFを選択してください。", true);
      return;
    }
    if (f.type && f.type !== "application/pdf") {
      setStatus("PDFファイルを選択してください。", true);
      return;
    }
    setStatus(`PDFをアップロード中…（${f.name}）`);
    try {
      await uploadPdf(apiBase, f);
      setStatus(`アップロード完了: ${f.name}`);
      pdfFileEl.value = "";
      await refreshSources();
    } catch (e) {
      setStatus(`アップロード失敗: ${e && e.message ? e.message : String(e)}`, true);
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
    if (!prompt) return;

    saveApiBase(apiBase);
    promptEl.value = "";
    addMessage("user", prompt);
    setStatus("送信中…");

    try {
      const data = await postChat({ apiBase, prompt, model, k });
      addMessage("assistant", data.answer || "", { model, sources: data.sources });
    } catch (err) {
      setStatus(`エラー: ${err && err.message ? err.message : String(err)}`, true);
    }
  });

  addMessage(
    "assistant",
    "API Base URL を設定して、質問を送ってください。\n（cloudflaredのURLを貼ればGitHub Pagesからでも叩けます）"
  );

  refreshSources();
}

main();

