#!/usr/bin/env node
/**
 * 530 フローの契約チェック（Node で実行可能。Node が無い場合は tests/verify_530_flow.py を使用）。
 */
const fs = require("fs");
const path = require("path");
const appPath = path.join(__dirname, "../docs/app.js");
const appJs = fs.readFileSync(appPath, "utf8");
const checks = [];
if (!appJs.includes("show530Banner")) checks.push("FAIL: show530Banner が app.js に存在しません");
else checks.push("OK: show530Banner が定義されている");
if (!appJs.includes("banner530")) checks.push("FAIL: banner530 が app.js に存在しません");
else checks.push("OK: banner530 バナーが実装されている");
if (!appJs.includes("isLikely530OrNetworkError")) checks.push("FAIL: isLikely530OrNetworkError が app.js に存在しません");
else checks.push("OK: isLikely530OrNetworkError が定義されている");
if (!appJs.includes("err.status = 0") || !appJs.includes("API に接続できません")) checks.push("FAIL: fetch 失敗時に err.status = 0 とメッセージを設定しているか確認");
else checks.push("OK: fetch 失敗時に err.status = 0 とメッセージを設定");
const failed = checks.some((c) => c.startsWith("FAIL"));
checks.forEach((c) => console.log(c));
process.exit(failed ? 1 : 0);
