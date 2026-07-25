const state = {
  network: null,
  lastProof: null,
  busy: false,
};

const $ = (selector) => document.querySelector(selector);

function shortHash(value, length = 12) {
  return value ? `${value.slice(0, length)}…` : "—";
}

function number(value, maximumFractionDigits = 1) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(value ?? 0);
}

function timeLabel(timestamp) {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(timestamp));
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("visible");
  window.clearTimeout(toast.timeout);
  toast.timeout = window.setTimeout(() => element.classList.remove("visible"), 3500);
}

async function request(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.message || body.error || "Request failed");
  return body;
}

function renderMetrics(network) {
  const { metrics, chain } = network;
  $("#metric-memory").textContent = `${number(metrics.totalMemoryGiB, 0)} GiB`;
  $("#metric-nodes").textContent = `${metrics.nodesOnline} / ${metrics.nodesTotal}`;
  $("#metric-receipts").textContent = number(metrics.verifiedReceipts, 0);
  $("#metric-height").textContent = `#${metrics.chainHeight}`;
  $("#model-name").textContent = network.model.name;
  $("#model-stages").textContent = `${network.model.shardCount} execution stages`;
  $("#empty-stage-count").textContent = network.model.shardCount;

  const networkPill = $(".network-pill");
  networkPill.classList.toggle("degraded", network.status !== "healthy");
  $("#network-label").textContent =
    network.status === "healthy"
      ? `${metrics.nodesOnline} nodes / live`
      : `${metrics.nodesOnline} nodes / rerouting`;
  $("#topology-status").textContent =
    `${metrics.nodesOnline}/${metrics.nodesTotal} ONLINE`;

  $("#token-supply").textContent = number(chain.token.totalSupply, 2);
  $("#token-burned").textContent = `${number(chain.token.totalBurned, 4)} CTX`;
  $("#token-rewards").textContent = `${number(chain.token.providerRewards, 3)} CTX`;
  $("#token-demo-balance").textContent = `${number(chain.demoWallet.balance, 2)} CTX`;
  const valid = $("#chain-valid");
  valid.textContent = chain.valid ? "✓ CHAIN VALID" : "CHAIN INVALID";
}

function renderTopology(network) {
  const container = $("#topology");
  const nodes = network.nodes;
  const width = 420;
  const height = 318;
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = 166;
  const radiusY = 118;
  const positions = nodes.map((_, index) => {
    const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
    return {
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  });

  container.innerHTML = "";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  positions.forEach(({ x, y }) => {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", centerX);
    line.setAttribute("y1", centerY);
    line.setAttribute("x2", x);
    line.setAttribute("y2", y);
    line.setAttribute("class", "topology-line");
    svg.appendChild(line);
  });
  container.appendChild(svg);

  const center = document.createElement("div");
  center.className = "topology-center";
  const parameterLabel =
    network.model.parameters >= 1e9
      ? `${number(network.model.parameters / 1e9, 2)}B`
      : `${number(network.model.parameters / 1e6, 0)}M`;
  center.innerHTML = `<span>${parameterLabel}</span><small>${network.model.shardCount} STAGES</small>`;
  container.appendChild(center);

  nodes.forEach((node, index) => {
    const button = document.createElement("button");
    button.className = `topology-node ${node.status === "offline" ? "offline" : ""}`;
    button.style.left = `${(positions[index].x / width) * 100}%`;
    button.style.top = `${(positions[index].y / height) * 100}%`;
    button.style.setProperty("--node-color", node.accent);
    button.textContent = node.name.slice(0, 2).toUpperCase();
    button.title = `${node.name}: click to set ${node.status === "online" ? "offline" : "online"}`;
    button.dataset.nodeToggle = node.id;
    container.appendChild(button);
  });
}

function renderShards(network) {
  $("#shard-strip").innerHTML = network.model.placements
    .map(
      (shard) => `
        <div class="shard-cell">
          <b>S${String(shard.index).padStart(2, "0")}</b>
          <span>${shard.replicas.length}× REPLICA</span>
        </div>
      `,
    )
    .join("");
}

function renderNodes(network) {
  $("#node-grid").innerHTML = network.nodes
    .map(
      (node) => `
        <article class="node-card ${node.status === "offline" ? "offline" : ""}" style="--node-color:${node.accent}">
          <div class="node-card-head">
            <div class="node-name">
              <span class="node-avatar">${node.name.slice(0, 2).toUpperCase()}</span>
              <div><b>${node.name}</b><small>${node.operator} / ${node.region}</small></div>
            </div>
            <button class="node-toggle" data-node-toggle="${node.id}" aria-label="Set ${node.name} ${node.status === "online" ? "offline" : "online"}"></button>
          </div>
          <div class="node-stats">
            <div><span>Memory</span><b>${number(node.allocatedMemoryGiB, 1)} / ${node.memoryGiB} GiB</b></div>
            <div><span>Jobs</span><b>${node.completedJobs}</b></div>
            <div><span>Earned</span><b>${number(node.balance, 3)} CTX</b></div>
          </div>
          <div class="node-shards">${node.shards.map((shard) => `<span>S${String(shard.index).padStart(2, "0")}</span>`).join("")}</div>
        </article>
      `,
    )
    .join("");
}

function renderBlocks(network) {
  $("#block-list").innerHTML = network.chain.recentBlocks
    .slice(0, 6)
    .map(
      (block) => `
        <div class="block-row">
          <span class="block-height">#${block.height}</span>
          <span class="block-hash">${block.hash}</span>
          <span class="block-kind">${block.kind === "inference" ? "useful work" : block.kind}</span>
          <span class="block-time">${timeLabel(block.timestamp)}</span>
        </div>
      `,
    )
    .join("");
}

function renderNetwork(network) {
  state.network = network;
  renderMetrics(network);
  renderTopology(network);
  renderShards(network);
  renderNodes(network);
  renderBlocks(network);
}

async function refresh() {
  try {
    renderNetwork(await request("/api/network"));
  } catch (error) {
    $("#network-label").textContent = "Coordinator offline";
    toast(error.message, true);
  }
}

function renderResult(result) {
  state.lastProof = result.proof;
  $("#result-empty").classList.add("hidden");
  $("#result-content").classList.remove("hidden");
  $("#result-height").textContent = `#${result.block.height}`;
  $("#result-text").textContent = result.output.text;
  $("#result-block-hash").textContent = shortHash(result.block.hash, 18);
  $("#result-block-hash").title = result.block.hash;
  $("#result-proof-root").textContent = shortHash(result.proof.root, 18);
  $("#result-proof-root").title = result.proof.root;
  $("#result-kernel-time").textContent = `${number(result.performance.kernelMicros / 1000, 2)} ms`;
  $("#result-reward").textContent = `${number(result.accounting.providerRewards, 3)} CTX`;

  const accents = new Map(state.network.nodes.map((node) => [node.id, node.accent]));
  $("#route-track").innerHTML = result.route
    .map(
      (stage) => `
        <div class="route-stage" style="--stage-color:${accents.get(stage.nodeId) || "#7187ed"}" title="${stage.nodeName} / ${stage.region} / ${stage.kernelMicros} μs">
          <b>S${String(stage.stage).padStart(2, "0")}</b>
          <span>${stage.nodeName}</span>
        </div>
      `,
    )
    .join("");
  $("#route-track").style.gridTemplateColumns =
    `repeat(${Math.min(result.route.length, 12)}, 1fr)`;
  $("#result-panel").scrollIntoView({ behavior: "smooth", block: "center" });
}

async function runInference() {
  if (state.busy) return;
  const prompt = $("#prompt").value.trim();
  if (prompt.length < 2) {
    toast("Enter a prompt first.", true);
    return;
  }
  state.busy = true;
  const button = $("#run-inference");
  button.disabled = true;
  button.classList.add("loading");
  try {
    const result = await request("/api/inference", {
      method: "POST",
      body: JSON.stringify({ modelId: state.network.model.id, prompt }),
    });
    renderResult(result);
    toast(`Block #${result.block.height} finalized from 12 signed receipts.`);
    await refresh();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.busy = false;
    button.disabled = false;
    button.classList.remove("loading");
  }
}

async function toggleNode(nodeId) {
  const node = state.network.nodes.find((candidate) => candidate.id === nodeId);
  if (!node || state.busy) return;
  const status = node.status === "online" ? "offline" : "online";
  try {
    await request(`/api/nodes/${nodeId}/status`, {
      method: "POST",
      body: JSON.stringify({ status }),
    });
    toast(`${node.name} is ${status}. Replica routing updated.`);
    await refresh();
  } catch (error) {
    toast(error.message, true);
  }
}

async function verifyLastProof() {
  if (!state.lastProof) return;
  try {
    const result = await request("/api/proofs/verify", {
      method: "POST",
      body: JSON.stringify({ proof: state.lastProof }),
    });
    toast(
      result.valid
        ? `${result.receiptCount} signatures, activation links, and Merkle root verified.`
        : "Proof verification failed.",
      !result.valid,
    );
  } catch (error) {
    toast(error.message, true);
  }
}

document.addEventListener("click", (event) => {
  const promptButton = event.target.closest("[data-prompt]");
  if (promptButton) $("#prompt").value = promptButton.dataset.prompt;
  const nodeButton = event.target.closest("[data-node-toggle]");
  if (nodeButton) toggleNode(nodeButton.dataset.nodeToggle);
});

$("#run-inference").addEventListener("click", runInference);
$("#verify-proof").addEventListener("click", verifyLastProof);

refresh();
window.setInterval(() => {
  if (!state.busy) refresh();
}, 8_000);
