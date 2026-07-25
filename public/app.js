const state = {
  network: null,
  refreshing: false,
  busyNodes: new Set(),
};

const $ = (selector) => document.querySelector(selector);

$("#overview").after($("#chain"));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, maximumFractionDigits = 1) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(
    Number(value ?? 0),
  );
}

function parameterCount(value) {
  const parameters = Number(value ?? 0);
  if (parameters >= 1e12) return `${number(parameters / 1e12, 2)}T`;
  if (parameters >= 1e9) return `${number(parameters / 1e9, 2)}B`;
  if (parameters >= 1e6) return `${number(parameters / 1e6, 0)}M`;
  return number(parameters, 0);
}

function shortHash(value, length = 14) {
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

function renderOverview(network) {
  const { metrics } = network;
  const { chain } = network;
  const healthy = network.status === "healthy";

  setText("#protocol-version", network.protocolVersion);
  setText("#metric-consensus", "USEFUL WORK");
  setText("#metric-chain-valid", chain.valid ? "VALID" : "INVALID");
  setText("#metric-height", `#${metrics.chainHeight}`);
  setText("#metric-supply", `${number(chain.token.totalSupply, 2)} CTX`);
  setText(
    "#metric-rewards",
    `${number(chain.token.providerRewards, 3)} CTX`,
  );
  setText("#metric-burned", `${number(chain.token.totalBurned, 4)} CTX`);
  setText("#metric-inferences", number(metrics.inferencesFinalized, 0));
  setText("#metric-nodes", `${metrics.nodesOnline}/${metrics.nodesTotal}`);

  const chainStatus = $("#metric-chain-valid");
  chainStatus.classList.toggle("good", chain.valid);
  chainStatus.classList.toggle("warn", !chain.valid);

  const connection = $("#connection");
  connection.classList.toggle("degraded", !healthy);
  connection.classList.remove("offline");
  setText(
    "#network-label",
    healthy
      ? `${metrics.nodesOnline} NODES / LIVE`
      : `${metrics.nodesOnline} NODES / DEGRADED`,
  );
}

function renderRuntime(network) {
  const model = network.model;
  const runtime = network.modelRuntime || {};
  setText("#model-name", model.name);
  setText("#model-id", model.id);
  setText("#model-version", model.version);
  setText("#model-precision", model.precision);
  setText("#model-parameters", parameterCount(model.parameters));
  setText("#model-context", number(model.contextWindow, 0));
  setText("#model-stage-count", number(model.shardCount, 0));
  setText("#runtime-engine", runtime.engine || model.architecture);
  setText(
    "#runtime-state",
    runtime.active ? "ACTIVE" : runtime.available ? "READY" : "DETERMINISTIC",
  );
}

function renderPlacements(network) {
  const model = network.model;
  setText(
    "#model-placement-summary",
    `${number(model.modelMemoryGiB, 2)} GiB MODEL / ` +
      `${number(model.replicatedMemoryGiB, 2)} GiB PLACED`,
  );
  $("#model-placements").innerHTML = model.placements
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
            <span>${stage.replicas.length}×</span>
          </header>
          <dl>
            <div><dt>SIZE</dt><dd>${number(stage.sizeGiB, 2)} GiB</dd></div>
            <div><dt>LAYERS</dt><dd>${escapeHtml(layers)}</dd></div>
            <div><dt>WORK</dt><dd>${number(stage.computeUnits, 0)} units</dd></div>
            <div><dt>HOSTS</dt><dd class="replica-list" title="${escapeHtml(replicas)}">${escapeHtml(replicas)}</dd></div>
          </dl>
        </article>
      `;
    })
    .join("");
}

function roleLabel(role) {
  return String(role || "provider").replaceAll("-", " ").toUpperCase();
}

function renderProviders(network) {
  $("#provider-rows").innerHTML = network.nodes
    .map((node) => {
      const stages = node.shards.length
        ? node.shards
            .map((stage) => String(stage.index).padStart(2, "0"))
            .join(",")
        : "—";
      const canControl = !node.remote;
      const nextStatus = node.status === "online" ? "offline" : "online";
      const control = canControl
        ? `<button type="button" data-node-toggle="${escapeHtml(node.id)}" ${
            state.busyNodes.has(node.id) ? "disabled" : ""
          }>${nextStatus === "online" ? "SET ONLINE" : "SET OFFLINE"}</button>`
        : "LEASED";
      return `
        <tr>
          <td><span class="state-label ${escapeHtml(node.status)}">${escapeHtml(node.status.toUpperCase())}</span></td>
          <td>
            <span class="node-id">${escapeHtml(node.name)} / ${escapeHtml(node.id)}</span>
            <span class="node-operator">${escapeHtml(node.operator)}</span>
          </td>
          <td><span class="role">${escapeHtml(roleLabel(node.role))}</span></td>
          <td>${escapeHtml(node.region)}</td>
          <td>${number(node.allocatedMemoryGiB, 2)} / ${number(node.memoryGiB, 1)} GiB</td>
          <td><span class="stage-list" title="${escapeHtml(stages)}">${escapeHtml(stages)}</span></td>
          <td>${number(node.networkLatencyMs, 0)} ms</td>
          <td>${number(node.completedJobs, 0)}</td>
          <td>${number(node.balance, 4)} CTX</td>
          <td>${control}</td>
        </tr>
      `;
    })
    .join("");
}

function renderAccounting(network) {
  const chain = network.chain;
  const epoch = chain.economics.currentEpoch;
  setText("#token-supply", `${number(chain.token.totalSupply, 4)} CTX`);
  setText("#token-max-supply", `${number(chain.token.maxSupply, 0)} CTX`);
  setText(
    "#token-rewards",
    `${number(chain.token.providerRewards, 4)} CTX`,
  );
  setText("#token-burned", `${number(chain.token.totalBurned, 4)} CTX`);
  setText("#token-epoch", `#${epoch.index}`);
  setText("#token-epoch-fees", `${number(epoch.fees, 4)} CTX`);
  setText("#token-epoch-issuance", `${number(epoch.issuance, 4)} CTX`);
  setText(
    "#token-epoch-budget",
    `${number(epoch.remainingBudget, 4)} CTX`,
  );
  setText("#chain-valid", chain.valid ? "VALID" : "INVALID");
  $("#chain-valid").classList.toggle("state-label", chain.valid);
}

function renderBlocks(network) {
  const blocks = network.chain.recentBlocks || [];
  $("#block-rows").innerHTML = blocks.length
    ? blocks
        .slice(0, 10)
        .map(
          (block) => `
            <tr>
              <td>#${number(block.height, 0)}</td>
              <td>${escapeHtml(block.kind.toUpperCase())}</td>
              <td class="block-hash" title="${escapeHtml(block.hash)}">${escapeHtml(shortHash(block.hash, 20))}</td>
              <td>${escapeHtml(timeLabel(block.timestamp))}</td>
            </tr>
          `,
        )
        .join("")
    : '<tr><td colspan="4" class="loading-row">NO BLOCKS</td></tr>';
}

function renderEvents(network) {
  const events = network.recentEvents || [];
  $("#event-log").innerHTML = events.length
    ? events
        .slice(0, 12)
        .map(
          (event) => `
            <li>
              <time datetime="${escapeHtml(event.timestamp)}">${escapeHtml(timeLabel(event.timestamp))}</time>
              <span class="event-kind ${escapeHtml(event.type)}">${escapeHtml(event.type)}</span>
              <strong>${escapeHtml(event.title)}</strong>
              <span class="event-detail">${escapeHtml(event.detail)}</span>
            </li>
          `,
        )
        .join("")
    : '<li class="loading-row">NO EVENTS</li>';
}

function renderNetwork(network) {
  state.network = network;
  renderOverview(network);
  renderRuntime(network);
  renderPlacements(network);
  renderProviders(network);
  renderAccounting(network);
  renderBlocks(network);
  renderEvents(network);
  setText("#last-updated", timeLabel(new Date().toISOString(), true));
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

refresh();
window.setInterval(refresh, 8_000);
