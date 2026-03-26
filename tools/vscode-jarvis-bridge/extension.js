"use strict";

const http = require("http");
const path = require("path");
const vscode = require("vscode");

let server = null;

function currentWorkspacePath() {
  const folders = vscode.workspace.workspaceFolders || [];
  if (!folders.length) {
    const activeFile = currentActiveFile();
    if (!activeFile) {
      return "";
    }
    return path.dirname(activeFile);
  }
  return folders[0].uri.fsPath || "";
}

function currentActiveFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !editor.document) {
    return "";
  }
  return editor.document.uri.fsPath || "";
}

function currentSelectionText() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !editor.document || !editor.selection) {
    return "";
  }
  if (editor.selection.isEmpty) {
    return "";
  }
  return editor.document.getText(editor.selection) || "";
}

function normalizeSlashes(p) {
  return String(p || "").replace(/\\/g, "/");
}

function applyPathMappings(hostPath, mappings) {
  const pathIn = normalizeSlashes(hostPath).trim();
  if (!pathIn || !Array.isArray(mappings) || !mappings.length) {
    return pathIn;
  }
  let best = null;
  for (const mapping of mappings) {
    if (!mapping || typeof mapping !== "object") {
      continue;
    }
    const from = normalizeSlashes(mapping.from || "").replace(/\/+$/, "");
    const to = normalizeSlashes(mapping.to || "").replace(/\/+$/, "");
    if (!from || !to) {
      continue;
    }
    if (pathIn === from || pathIn.startsWith(`${from}/`)) {
      if (!best || from.length > best.from.length) {
        best = { from, to };
      }
    }
  }
  if (!best) {
    return pathIn;
  }
  const suffix = pathIn.slice(best.from.length);
  return `${best.to}${suffix}` || best.to;
}

function resolveWorkspaceForJarvis(settings) {
  const workspace = currentWorkspacePath();
  const forced = String(settings.forceWorkspacePath || "").trim();
  if (forced) {
    return forced;
  }
  return applyPathMappings(workspace, settings.workspacePathMappings);
}

function resolveActiveFileForJarvis(settings) {
  const activeFile = currentActiveFile();
  return applyPathMappings(activeFile, settings.workspacePathMappings);
}

async function proxyRequest(req, res, settings) {
  const chunks = [];
  req.on("data", (chunk) => chunks.push(chunk));
  req.on("end", async () => {
    const rawBody = Buffer.concat(chunks);
    const targetBaseUrl = settings.targetBaseUrl;
    let targetUrl = `${targetBaseUrl}${req.url || "/"}`;
    const headers = { ...req.headers };
    delete headers.host;
    delete headers["content-length"];

    const workspace = resolveWorkspaceForJarvis(settings);
    const activeFile = resolveActiveFileForJarvis(settings);
    const selection = currentSelectionText();
    if (workspace) {
      headers["x-jarvis-workspace"] = workspace;
      headers["x-workspace-path"] = workspace;
    }
    const hostWorkspace = currentWorkspacePath();
    if (hostWorkspace) {
      headers["x-jarvis-host-workspace"] = hostWorkspace;
    }
    if (activeFile) {
      headers["x-jarvis-active-file"] = activeFile;
    }
    if (selection) {
      headers["x-jarvis-selection"] = selection.slice(0, 4096);
    }

    let upstream;
    try {
      upstream = await fetch(targetUrl, {
        method: req.method || "GET",
        headers,
        body: ["GET", "HEAD"].includes((req.method || "GET").toUpperCase()) ? undefined : rawBody
      });
    } catch (err) {
      res.writeHead(502, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: `Bridge upstream error: ${String(err)}` }));
      return;
    }

    const outHeaders = {};
    upstream.headers.forEach((value, key) => {
      if (key.toLowerCase() === "content-length") {
        return;
      }
      outHeaders[key] = value;
    });
    outHeaders["x-jarvis-bridge"] = "vscode";
    res.writeHead(upstream.status, outHeaders);
    if (!upstream.body) {
      res.end();
      return;
    }

    const reader = upstream.body.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        if (value && value.length) {
          res.write(Buffer.from(value));
        }
      }
      res.end();
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(502, { "content-type": "application/json" });
      }
      res.end(JSON.stringify({ error: `Bridge stream error: ${String(err)}` }));
    }
  });
}

function startServer(context) {
  const cfg = vscode.workspace.getConfiguration();
  const port = Number(cfg.get("jarvisBridge.listenPort", 8787));
  const targetBaseUrl = String(cfg.get("jarvisBridge.targetBaseUrl", "http://127.0.0.1:8080")).replace(/\/+$/, "");
  const forceWorkspacePath = String(cfg.get("jarvisBridge.forceWorkspacePath", "")).trim();
  const rawMappings = cfg.get("jarvisBridge.workspacePathMappings", []);
  const workspacePathMappings = Array.isArray(rawMappings)
    ? rawMappings
        .filter((m) => m && typeof m === "object")
        .map((m) => ({ from: String(m.from || ""), to: String(m.to || "") }))
        .filter((m) => m.from.trim() && m.to.trim())
    : [];
  const settings = {
    targetBaseUrl,
    forceWorkspacePath,
    workspacePathMappings,
  };

  if (server) {
    server.close();
    server = null;
  }

  server = http.createServer((req, res) => {
    proxyRequest(req, res, settings);
  });
  server.listen(port, "127.0.0.1", () => {
    const mode = forceWorkspacePath
      ? `forced=${forceWorkspacePath}`
      : workspacePathMappings.length
        ? `mapped=${workspacePathMappings.length}`
        : "raw";
    vscode.window.setStatusBarMessage(`Jarvis Bridge: 127.0.0.1:${port} -> ${targetBaseUrl} (${mode})`, 5000);
  });

  context.subscriptions.push({
    dispose: () => {
      if (server) {
        server.close();
        server = null;
      }
    }
  });
}

function activate(context) {
  startServer(context);
  context.subscriptions.push(
    vscode.commands.registerCommand("jarvisBridge.restart", () => startServer(context)),
    vscode.commands.registerCommand("jarvisBridge.stop", () => {
      if (server) {
        server.close();
        server = null;
        vscode.window.setStatusBarMessage("Jarvis Bridge stopped", 3000);
      }
    })
  );
}

function deactivate() {
  if (server) {
    server.close();
    server = null;
  }
}

module.exports = {
  activate,
  deactivate
};
