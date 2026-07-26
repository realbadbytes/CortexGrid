const state = {
  network: null,
  refreshing: false,
  busyNodes: new Set(),
  chat: {
    bridge: null,
    history: [],
    busy: false,
  },
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, maximumFractionDigits = 2) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(
    Number(value ?? 0),
  );
}

function ctx(value, maximumFractionDigits = 6) {
  return `${number(value, maximumFractionDigits)} CTX`;
}

function rewardPolicy(chain) {
  const service = chain.economics.service || {};
  const bootstrap = chain.economics.bootstrap || {};
  return (
    `${number(service.providerSharePercent, 1)}% service revenue` +
    ` + ${number(bootstrap.currentMatchMultiplier, 2)}× bootstrap`
  );
}

function parameterCount(value) {
  const parameters = Number(value ?? 0);
  if (parameters >= 1e12) return `${number(parameters / 1e12, 2)}T`;
  if (parameters >= 1e9) return `${number(parameters / 1e9, 2)}B`;
  if (parameters >= 1e6) return `${number(parameters / 1e6, 0)}M`;
  return number(parameters, 0);
}

function shortHash(value, length = 18) {
  const text = String(value ?? "");
  return text ? `${text.slice(0, length)}…` : "—";
}

function timeLabel(timestamp, includeDate = false) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "—";
  const options = {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  };
  if (includeDate) {
    options.year = "numeric";
    options.month = "2-digit";
    options.day = "2-digit";
  }
  return new Intl.DateTimeFormat("en-CA", options).format(date);
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("visible");
  window.clearTimeout(toast.timeout);
  toast.timeout = window.setTimeout(
    () => element.classList.remove("visible"),
    3200,
  );
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.message || body.error || "Request failed");
  }
  return body;
}

function bridgeFromFragment() {
  const prefix = "#cortex-chat=";
  const storageKey = "cortexgrid-chat-bridge";
  const fromFragment = window.location.hash.startsWith(prefix);
  let encoded = fromFragment
    ? window.location.hash.slice(prefix.length)
    : "";
  if (fromFragment) {
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}`,
    );
  } else {
    try {
      encoded = window.sessionStorage.getItem(storageKey) || "";
    } catch {
      encoded = "";
    }
  }
  if (!encoded) return null;
  try {
    const base64 = encoded.replaceAll("-", "+").replaceAll("_", "/");
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
    const bridge = JSON.parse(window.atob(padded));
    const endpoint = new URL(bridge.endpoint);
    if (
      !["http:", "https:"].includes(endpoint.protocol) ||
      endpoint.username ||
      endpoint.password ||
      typeof bridge.token !== "string" ||
      bridge.token.length < 32
    ) {
      throw new Error("invalid local bridge");
    }
    const result = {
      endpoint: endpoint.origin,
      token: bridge.token,
      nodeId: String(bridge.nodeId || ""),
    };
    try {
      window.sessionStorage.setItem(storageKey, encoded);
    } catch {
      // The live page still works when browser storage is disabled.
    }
    return result;
  } catch {
    try {
      window.sessionStorage.removeItem(storageKey);
    } catch {
      // Nothing else is required when browser storage is disabled.
    }
    throw new Error("The CortexGrid Chat connection link is invalid.");
  }
}

async function bridgeRequest(path, options = {}) {
  if (!state.chat.bridge) {
    throw new Error("Local compute is not connected.");
  }
  const response = await fetch(`${state.chat.bridge.endpoint}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${state.chat.bridge.token}`,
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.message || body.error || "Local bridge request failed");
  }
  return body;
}

function appendChatMessage(kind, label, text, metadata = "") {
  const message = document.createElement("div");
  message.className = `chat-message ${kind}`;

  const name = document.createElement("span");
  name.textContent = label;
  const content = document.createElement("div");
  const body = document.createElement("p");
  body.textContent = text;
  content.append(body);

  if (metadata) {
    const meta = document.createElement("p");
    meta.className = "chat-meta";
    meta.textContent = metadata;
    content.append(meta);
  }

  message.append(name, content);
  $("#chat-log").append(message);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return message;
}

function conversationPrompt(userPrompt) {
  if (!state.chat.history.length) return userPrompt;
  const exchanges = state.chat.history
    .map(
      ([user, assistant]) => `User: ${user}\nAssistant: ${assistant}`,
    )
    .join("\n");
  const prompt =
    "Continue the conversation. Answer the final user message.\n" +
    `${exchanges}\nUser: ${userPrompt}\nAssistant:`;
  return prompt.length <= 7_000 ? prompt : userPrompt;
}

function setChatReady(ready, status) {
  $("#cortex-chat").classList.toggle("connected", ready);
  $("#cortex-chat").classList.toggle("failed", !ready && Boolean(state.chat.bridge));
  setText("#chat-status", status);
  $("#chat-prompt").disabled = !ready;
  $("#chat-submit").disabled = !ready;
  $("#chat-new").disabled = !ready;
  $("#chat-prompt").placeholder = ready
    ? "Ask the distributed model"
    : "Connect local compute to begin";
}

async function connectBrowserChat() {
  try {
    state.chat.bridge = bridgeFromFragment();
    if (!state.chat.bridge) return;
    setChatReady(false, "CONNECTING LOCAL COMPUTE");
    const status = await bridgeRequest("/v1/browser/status");
    if (!status.connected) {
      throw new Error("The local compute participant is not online.");
    }
    const provider = status.provider || {};
    const stage = provider.assignment?.shard?.index;
    setChatReady(
      true,
      `${provider.nodeId || state.chat.bridge.nodeId} / STAGE ${stage ?? "PENDING"}`,
    );
    $("#bridge-setup").hidden = true;
    $("#chat-log").replaceChildren();
    appendChatMessage(
      "system",
      "System",
      `Connected to ${provider.name || provider.nodeId}. ` +
        `Current balance: ${ctx(provider.balance, 6)}.`,
    );
    $("#chat-prompt").focus();
  } catch (error) {
    setChatReady(false, "LOCAL COMPUTE CONNECTION FAILED");
    appendChatMessage("error", "Error", error.message);
  }
}

async function submitChat(event) {
  event.preventDefault();
  if (state.chat.busy || !state.chat.bridge) return;
  const input = $("#chat-prompt");
  const userPrompt = input.value.trim();
  if (userPrompt.length < 2) return;

  state.chat.busy = true;
  input.value = "";
  input.disabled = true;
  $("#chat-submit").disabled = true;
  $("#chat-new").disabled = true;
  appendChatMessage("user", "You", userPrompt);
  const pending = appendChatMessage(
    "system",
    "Network",
    "Running distributed inference…",
  );

  try {
    const result = await bridgeRequest("/v1/browser/inference", {
      method: "POST",
      body: JSON.stringify({ prompt: conversationPrompt(userPrompt) }),
    });
    pending.remove();
    const answer = String(result.output?.text || "");
    const route = (result.route || [])
      .map((stage) => `${stage.nodeName || stage.nodeId}[${stage.stage}]`)
      .join(" → ");
    const block = result.block || {};
    const access = result.access || {};
    appendChatMessage(
      "assistant",
      "Cortex",
      answer,
      `route ${route || "—"} · block #${block.height ?? "—"} · ` +
        `earned ${ctx(access.participantReward, 6)}`,
    );
    state.chat.history.push([userPrompt, answer]);
    await refresh();
  } catch (error) {
    pending.remove();
    appendChatMessage("error", "Error", error.message);
  } finally {
    state.chat.busy = false;
    input.disabled = false;
    $("#chat-submit").disabled = false;
    $("#chat-new").disabled = false;
    input.focus();
  }
}

function resetChat() {
  state.chat.history = [];
  $("#chat-log").replaceChildren();
  appendChatMessage(
    "system",
    "System",
    "New conversation. Local compute remains connected.",
  );
  $("#chat-prompt").focus();
}

function renderOverview(network) {
  const { chain, metrics } = network;
  const healthy = network.status === "healthy";

  setText(
    "#network-label",
    healthy ? "NETWORK ONLINE" : "NETWORK DEGRADED",
  );

  const connection = $("#connection");
  connection.classList.toggle("degraded", !healthy);
  connection.classList.remove("offline");
  $(".live-proof").classList.toggle("degraded", !healthy);
  $(".live-proof").classList.remove("offline");

  setText("#reward-rate", rewardPolicy(chain));

  setText(
    "#ledger-label",
    chain.valid ? "Ledger check passed" : "Ledger check failed",
  );
  setText("#check-mark", chain.valid ? "✓" : "!");
  $("#ledger-check").classList.toggle("failed", !chain.valid);

  setText("#stat-workers", `${metrics.nodesOnline} / ${metrics.nodesTotal}`);
  setText("#stat-inferences", number(metrics.inferencesFinalized, 0));
  setText("#stat-rewards", ctx(chain.token.providerRewards, 6));
}

function renderAccounting(network) {
  const { chain } = network;
  setText("#detail-height", `#${chain.height}`);
  setText("#detail-hash", shortHash(chain.latestBlockHash, 24));
  $("#detail-hash").title = chain.latestBlockHash;
  setText("#detail-supply", ctx(chain.token.totalSupply, 6));
  setText("#detail-max-supply", ctx(chain.token.maxSupply, 0));
  setText("#detail-genesis", ctx(chain.token.genesisSupply, 0));
  setText("#detail-issued", ctx(chain.token.totalIssued, 6));
  setText("#detail-burned", ctx(chain.token.totalBurned, 6));
  setText("#detail-rate", rewardPolicy(chain));
}

function renderRuntime(network) {
  const model = network.model;
  const runtime = network.modelRuntime || {};
  setText("#model-name", model.name || "Active model");
  setText(
    "#model-summary",
    `${number(model.modelMemoryGiB, 2)} GiB split into ` +
      `${number(model.shardCount, 0)} stages; ` +
      `${number(model.replicatedMemoryGiB, 2)} GiB placed across providers.`,
  );
  setText("#protocol-version", network.protocolVersion);
  setText("#model-id", model.id);
  setText("#model-version", model.version);
  setText("#model-precision", model.precision);
  setText("#model-parameters", parameterCount(model.parameters));
  setText("#model-context", number(model.contextWindow, 0));
  setText("#runtime-engine", runtime.engine || model.architecture);
  setText(
    "#runtime-state",
    runtime.active ? "ACTIVE" : runtime.available ? "READY" : "DETERMINISTIC",
  );
}

function renderPlacements(network) {
  const placements = network.model.placements || [];
  $("#model-placements").innerHTML = placements.length
    ? placements
        .map((stage) => {
          const layers = Array.isArray(stage.layerRange)
            ? `${stage.layerRange[0]}–${stage.layerRange[1]}`
            : "N/A";
          const replicas = stage.replicas.length
            ? stage.replicas.join(", ")
            : "UNASSIGNED";
          return `
            <article class="stage-cell">
              <header>
                <b>STAGE ${String(stage.index).padStart(2, "0")}</b>
                <span>${stage.replicas.length} replica${stage.replicas.length === 1 ? "" : "s"}</span>
              </header>
              <dl>
                <div><dt>SIZE</dt><dd>${number(stage.sizeGiB, 2)} GiB</dd></div>
                <div><dt>LAYERS</dt><dd>${escapeHtml(layers)}</dd></div>
                <div><dt>COMPUTE</dt><dd>${number(stage.computeUnits, 0)} units</dd></div>
                <div><dt>HOSTS</dt><dd title="${escapeHtml(replicas)}">${escapeHtml(replicas)}</dd></div>
              </dl>
            </article>
          `;
        })
        .join("")
    : "<p>No model stages are currently placed.</p>";
}

function roleLabel(role) {
  return String(role || "provider").replaceAll("-", " ").toUpperCase();
}

function renderProviders(network) {
  $("#provider-rows").innerHTML = network.nodes.length
    ? network.nodes
        .map((node) => {
          const stages = node.shards.length
            ? node.shards
                .map((stage) => String(stage.index).padStart(2, "0"))
                .join(", ")
            : "—";
          const nextStatus = node.status === "online" ? "offline" : "online";
          const control = node.remote
            ? "LEASED"
            : `<button type="button" data-node-toggle="${escapeHtml(node.id)}" ${
                state.busyNodes.has(node.id) ? "disabled" : ""
              }>${nextStatus === "online" ? "SET ONLINE" : "SET OFFLINE"}</button>`;
          return `
            <tr>
              <td><span class="state-label ${escapeHtml(node.status)}">${escapeHtml(node.status.toUpperCase())}</span></td>
              <td><b>${escapeHtml(node.name)}</b><small>${escapeHtml(node.id)}</small></td>
              <td>${escapeHtml(roleLabel(node.role))}</td>
              <td>${number(node.allocatedMemoryGiB, 2)} / ${number(node.memoryGiB, 1)} GiB</td>
              <td>${escapeHtml(stages)}</td>
              <td>${number(node.completedJobs, 0)}</td>
              <td>${ctx(node.balance, 6)}</td>
              <td>${control}</td>
            </tr>
          `;
        })
        .join("")
    : '<tr><td colspan="8">No providers registered.</td></tr>';
}

function renderBlocks(network) {
  const blocks = network.chain.recentBlocks || [];
  const labels = {
    genesis: "Network and initial CTX created",
    inference: "Inference verified; providers rewarded",
    transfer: "CTX transferred",
  };
  $("#block-rows").innerHTML = blocks.length
    ? blocks
        .slice(0, 8)
        .map(
          (block) => `
            <tr>
              <td>#${number(block.height, 0)}</td>
              <td>${escapeHtml(labels[block.kind] || block.kind)}</td>
              <td class="hash" title="${escapeHtml(block.hash)}">${escapeHtml(shortHash(block.hash))}</td>
              <td>${escapeHtml(timeLabel(block.timestamp))}</td>
            </tr>
          `,
        )
        .join("")
    : '<tr><td colspan="4">No blocks recorded.</td></tr>';
}

function renderNetwork(network) {
  state.network = network;
  renderOverview(network);
  renderAccounting(network);
  renderRuntime(network);
  renderPlacements(network);
  renderProviders(network);
  renderBlocks(network);
  setText("#last-updated", timeLabel(new Date().toISOString(), true));
}

function renderJoinMode() {
  const interactive = $("#join-mode").value === "interactive";
  $("#interactive-panel").hidden = !interactive;
  $("#worker-panel").hidden = interactive;
}

function configureCommands() {
  const coordinator = window.location.origin;
  setText(
    "#browser-command",
    `make chat-browser COORDINATOR_URL=${coordinator} ` +
      "ADVERTISE_HOST=THIS_MACHINE_IP TERM_MEMORY_GIB=24",
  );
  setText(
    "#worker-command",
    "make join NODE_ID=my-node MEMORY_GIB=24 " +
      `COORDINATOR_URL=${coordinator} ADVERTISE_HOST=THIS_MACHINE_IP`,
  );
}

async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true;
  $("#refresh-network").disabled = true;
  try {
    renderNetwork(await request("/api/network"));
  } catch (error) {
    const connection = $("#connection");
    connection.classList.remove("degraded");
    connection.classList.add("offline");
    $(".live-proof").classList.remove("degraded");
    $(".live-proof").classList.add("offline");
    setText("#network-label", "COORDINATOR OFFLINE");
    toast(error.message, true);
  } finally {
    state.refreshing = false;
    $("#refresh-network").disabled = false;
  }
}

async function toggleNode(nodeId) {
  if (!state.network || state.busyNodes.has(nodeId)) return;
  const node = state.network.nodes.find((candidate) => candidate.id === nodeId);
  if (!node || node.remote) return;
  const status = node.status === "online" ? "offline" : "online";
  state.busyNodes.add(nodeId);
  renderProviders(state.network);
  try {
    await request(`/api/nodes/${encodeURIComponent(nodeId)}/status`, {
      method: "POST",
      body: JSON.stringify({ status }),
    });
    toast(`${node.id}: ${status.toUpperCase()}`);
    await refresh();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.busyNodes.delete(nodeId);
    if (state.network) renderProviders(state.network);
  }
}

async function copyCommand(targetId) {
  const target = document.getElementById(targetId);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target.textContent.trim());
    toast("COMMAND COPIED");
  } catch {
    toast("CLIPBOARD ACCESS UNAVAILABLE", true);
  }
}

document.addEventListener("click", (event) => {
  const nodeButton = event.target.closest("[data-node-toggle]");
  if (nodeButton) toggleNode(nodeButton.dataset.nodeToggle);
  const copyButton = event.target.closest("[data-copy-target]");
  if (copyButton) copyCommand(copyButton.dataset.copyTarget);
});

$("#refresh-network").addEventListener("click", refresh);
$("#join-mode").addEventListener("change", renderJoinMode);
$("#chat-form").addEventListener("submit", submitChat);
$("#chat-new").addEventListener("click", resetChat);
$("#chat-prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
$("#technical").addEventListener("toggle", (event) => {
  setText(".expand-label", event.currentTarget.open ? "Close" : "Open");
});
$("#how").addEventListener("toggle", (event) => {
  setText(".how-label", event.currentTarget.open ? "Close" : "Open");
});

configureCommands();
renderJoinMode();
refresh();
connectBrowserChat();
window.setInterval(refresh, 8_000);
