// app.js - fraud-review-app frontend
// Fetches from the same-origin FastAPI backend, renders the paginated case
// queue, the case detail panel (including a SHAP bar chart and a D3
// cluster diagram), and the budget simulator. No build step, no framework.
// Styling is Tailwind utility classes (loaded via CDN in index.html/docs.html);
// this file only ever needs to reference element IDs, never CSS class names,
// to hook up behavior.

const API_BASE = "";

let selectedTransactionId = null;
let budgetDebounceTimer = null;

// pagination state
let currentOffset = 0;
let currentLimit = 20;
let currentTotal = 0;

// --- formatting helpers ---------------------------------------------

const RISK_BADGE_BASE =
  "inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-data text-xs font-bold";

function riskTier(score) {
  if (score >= 0.75) return { label: "Critical", cls: `${RISK_BADGE_BASE} bg-risk-critical text-black` };
  if (score >= 0.5) return { label: "High", cls: `${RISK_BADGE_BASE} bg-risk-high text-black` };
  if (score >= 0.25) return { label: "Medium", cls: `${RISK_BADGE_BASE} bg-risk-medium text-white` };
  return { label: "Low", cls: `${RISK_BADGE_BASE} bg-risk-low text-black` };
}

function fmtAmount(amt) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(amt);
}

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

// Escapes untrusted text (top_reason, device strings, email domains, etc.
// all ultimately come from the pipeline's data, not literal user input
// here, but escaping on the way into innerHTML is cheap insurance).
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

// --- summary strip -----------------------------------------------------

async function loadSummary() {
  try {
    const res = await fetch(`${API_BASE}/api/summary`);
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("stat-n-cases").textContent = data.n_cases.toLocaleString();
    document.getElementById("stat-n-cluster-members").textContent = data.n_cluster_members.toLocaleString();
    document.getElementById("stat-pr-auc").textContent = data.pr_auc != null ? data.pr_auc.toFixed(4) : "\u2014";
  } catch (err) {
    console.error("Failed to load summary", err);
  }
}

// --- case queue (paginated) --------------------------------------------

async function loadCases() {
  const minRiskInput = document.getElementById("min-risk-score").value;
  const minRisk = minRiskInput === "" ? "0" : minRiskInput;
  const clusterOnly = document.getElementById("cluster-only").checked;
  const search = document.getElementById("txn-search").value.trim();
  const params = new URLSearchParams({
    min_risk_score: minRisk,
    cluster_only: String(clusterOnly),
    limit: String(currentLimit),
    offset: String(currentOffset),
  });
  if (search) {
    params.set("search", search);
  }

  try {
    const res = await fetch(`${API_BASE}/api/cases?${params}`);
    if (!res.ok) return;
    const body = await res.json();
    currentTotal = body.total;
    renderCaseTable(body.items);
    renderPaginationControls();
  } catch (err) {
    console.error("Failed to load cases", err);
  }
}

function resetToFirstPage() {
  currentOffset = 0;
  loadCases();
}

function renderCaseTable(cases) {
  const tbody = document.getElementById("case-table-body");
  const emptyState = document.getElementById("queue-empty-state");
  tbody.innerHTML = "";

  if (cases.length === 0) {
    emptyState.hidden = false;
    return;
  }
  emptyState.hidden = true;

  const cellBase = "px-3 py-2 border-b border-line whitespace-nowrap";

  for (const c of cases) {
    const tier = riskTier(c.risk_score);
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.className =
      "cursor-pointer hover:bg-canvas aria-selected:bg-[#EFEDE7] focus-visible:outline focus-visible:outline-2 focus-visible:outline-ink focus-visible:outline-offset-[-3px]";
    tr.dataset.transactionId = String(c.TransactionID);
    tr.setAttribute("aria-selected", String(c.TransactionID === selectedTransactionId));

    tr.innerHTML = `
      <td class="${cellBase} font-data">${c.TransactionID}</td>
      <td class="${cellBase} font-data">${fmtAmount(c.TransactionAmt)}</td>
      <td class="${cellBase}"><span class="${tier.cls}">${tier.label} &middot; ${c.risk_score.toFixed(2)}</span></td>
      <td class="${cellBase}">${c.cluster_id != null ? `<span class="text-xs text-muted">Ring #${c.cluster_id}</span>` : ""}</td>
      <td class="px-3 py-2 border-b border-line whitespace-normal min-w-[220px] text-muted">${escapeHtml(c.top_reason)}</td>
    `;

    tr.addEventListener("click", () => selectCase(c.TransactionID));
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectCase(c.TransactionID);
      }
    });

    tbody.appendChild(tr);
  }
}

function renderPaginationControls() {
  const summaryEl = document.getElementById("pagination-summary");
  const prevBtn = document.getElementById("prev-page-btn");
  const nextBtn = document.getElementById("next-page-btn");

  if (currentTotal === 0) {
    summaryEl.textContent = "No cases to show.";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    return;
  }

  const start = currentOffset + 1;
  const end = Math.min(currentOffset + currentLimit, currentTotal);
  summaryEl.textContent = `Showing ${start.toLocaleString()}\u2013${end.toLocaleString()} of ${currentTotal.toLocaleString()} cases`;

  prevBtn.disabled = currentOffset === 0;
  nextBtn.disabled = currentOffset + currentLimit >= currentTotal;
}

// --- case detail -----------------------------------------------------

async function selectCase(transactionId) {
  selectedTransactionId = transactionId;

  document.querySelectorAll("#case-table-body tr").forEach((tr) => {
    tr.setAttribute("aria-selected", String(Number(tr.dataset.transactionId) === transactionId));
  });

  const panel = document.getElementById("detail-panel");
  panel.innerHTML = `<p class="text-muted text-sm">Loading case ${transactionId}&hellip;</p>`;

  try {
    const res = await fetch(`${API_BASE}/api/cases/${transactionId}`);
    if (!res.ok) {
      panel.innerHTML = `<p class="text-muted text-sm">Could not load case ${transactionId}.</p>`;
      return;
    }
    const detail = await res.json();
    renderDetailPanel(detail);
  } catch (err) {
    console.error("Failed to load case detail", err);
    panel.innerHTML = `<p class="text-muted text-sm">Could not load case ${transactionId}.</p>`;
  }
}

function renderDetailPanel(detail) {
  const panel = document.getElementById("detail-panel");
  const tier = riskTier(detail.risk_score);

  let html = `
    <div class="font-data text-lg font-bold">Transaction ${detail.TransactionID}</div>
    <div class="flex gap-4 flex-wrap my-2 text-sm items-center">
      <span>${fmtAmount(detail.TransactionAmt)}</span>
      <span class="${tier.cls}">${tier.label} &middot; ${detail.risk_score.toFixed(3)}</span>
    </div>
    <p class="text-muted text-sm">${escapeHtml(detail.top_reason)}</p>
  `;

  if (detail.shap_features && detail.shap_features.length > 0) {
    const maxAbs = Math.max(...detail.shap_features.map((f) => Math.abs(f.shap_value)));
    html += `<h3 class="text-sm font-semibold mt-4 mb-2">Why this was flagged</h3><ul class="flex flex-col gap-2 list-none m-0 p-0">`;
    for (const f of detail.shap_features) {
      const isPositive = f.shap_value >= 0;
      const widthPct = maxAbs > 0 ? (Math.abs(f.shap_value) / maxAbs) * 100 : 0;
      const barColor = isPositive ? "bg-risk-critical" : "bg-risk-medium";
      html += `
        <li class="grid grid-cols-[minmax(90px,140px)_1fr_auto] items-center gap-2 text-[0.82rem]">
          <span class="font-data overflow-hidden text-ellipsis whitespace-nowrap" title="${escapeHtml(f.feature_name)}">${escapeHtml(f.feature_name)}</span>
          <span class="bg-canvas border border-line rounded h-3.5 overflow-hidden">
            <span class="${barColor} h-full block" style="width: ${widthPct.toFixed(1)}%"></span>
          </span>
          <span class="font-data text-xs text-muted text-right">${isPositive ? "+" : ""}${f.shap_value.toFixed(2)}</span>
        </li>
      `;
    }
    html += `</ul>`;
  }

  if (detail.cluster_info) {
    const ci = detail.cluster_info;
    const totalMembers = ci.member_transaction_ids.length;
    html += `
      <h3 class="text-sm font-semibold mt-4 mb-2">Linked transactions (ring #${ci.cluster_id})</h3>
      <p class="text-muted text-sm">Shared attribute: <strong class="text-ink">${escapeHtml(ci.shared_attribute)}</strong>
        &middot; Cluster fraud rate: <span class="font-data font-semibold">${fmtPct(ci.cluster_fraud_rate)}</span>
        &middot; ${totalMembers.toLocaleString()} linked transaction${totalMembers === 1 ? "" : "s"}</p>
      <ul class="pl-4 text-sm max-h-48 overflow-y-auto">
        ${ci.member_transaction_ids.map((id) => `<li class="mb-1">Transaction ${id}</li>`).join("")}
      </ul>
      <div class="mt-3">
        <svg id="cluster-diagram" role="img" class="w-full h-[220px] border border-line rounded-md bg-canvas"
             aria-label="Network diagram of transaction ${detail.TransactionID} and up to ${Math.min(totalMembers, MAX_DIAGRAM_NODES)} of its ${totalMembers} linked transactions"></svg>
        <p class="text-xs text-muted mt-2" id="diagram-caption"></p>
      </div>
    `;
  }

  panel.innerHTML = html;

  if (detail.cluster_info) {
    renderClusterDiagram(detail.TransactionID, detail.cluster_info);
  }
}

// Hard cap on how many cluster-mates the force-directed diagram will draw.
// Force layouts get illegible well before real ring sizes do - this
// dataset's rings range from 3 members up to 800+, and cramming hundreds
// of nodes into a fixed-height SVG produces an unreadable overlapping mess
// (confirmed directly against a real large ring, not a hypothetical). The
// full member list above is never capped - only the diagram is - so
// nothing is actually hidden from the non-visual equivalent, only from
// the visual that can't represent it legibly anyway.
const MAX_DIAGRAM_NODES = 20; // center + up to 19 cluster-mates

// --- D3 cluster diagram -----------------------------------------------

function renderClusterDiagram(centerId, clusterInfo) {
  const svgEl = document.getElementById("cluster-diagram");
  const captionEl = document.getElementById("diagram-caption");
  if (!svgEl || typeof d3 === "undefined") return;

  const totalMembers = clusterInfo.member_transaction_ids.length;
  const shownIds = clusterInfo.member_transaction_ids.slice(0, MAX_DIAGRAM_NODES - 1);
  const isCapped = totalMembers > shownIds.length;

  if (captionEl) {
    captionEl.textContent = isCapped
      ? `This diagram shows ${shownIds.length} of ${totalMembers} linked transactions (limited for legibility). The full list above contains all ${totalMembers}.`
      : "This diagram is a visual summary of the linked-transactions list above; the list contains the same information in full.";
  }

  const width = svgEl.clientWidth || 400;
  const height = 220;

  const nodes = [
    { id: centerId, isCenter: true },
    ...shownIds.map((id) => ({ id, isCenter: false })),
  ];
  const links = shownIds.map((id) => ({ source: centerId, target: id }));

  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${width} ${height}`);
  svg.selectAll("*").remove();

  const simulation = d3
    .forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(70))
    .force("charge", d3.forceManyBody().strength(-180))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(28));

  const link = svg
    .append("g")
    .attr("stroke", "#B9B4A8")
    .attr("stroke-width", 1.5)
    .selectAll("line")
    .data(links)
    .join("line");

  // Shared attribute is stated once in the text above the diagram, not
  // repeated as a label on every single edge - with more than a couple of
  // links, per-edge labels just stack into illegible overlapping text.

  const node = svg
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", (d) => (d.isCenter ? 16 : 11))
    .attr("fill", (d) => (d.isCenter ? "#D55E00" : "#0072B2"))
    .attr("stroke", "#1C1B19")
    .attr("stroke-width", 1);

  const label = svg
    .append("g")
    .selectAll("text.node-label")
    .data(nodes)
    .join("text")
    .attr("class", "node-label")
    .attr("font-size", 10)
    .attr("fill", "#1C1B19")
    .attr("text-anchor", "middle")
    .attr("dy", (d) => (d.isCenter ? -22 : -16))
    .text((d) => d.id);

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node
      .attr("cx", (d) => Math.max(16, Math.min(width - 16, d.x)))
      .attr("cy", (d) => Math.max(16, Math.min(height - 16, d.y)));

    label
      .attr("x", (d) => Math.max(16, Math.min(width - 16, d.x)))
      .attr("y", (d) => Math.max(16, Math.min(height - 16, d.y)));
  });
}

// --- budget simulator --------------------------------------------------

function debounceBudgetFetch() {
  clearTimeout(budgetDebounceTimer);
  budgetDebounceTimer = setTimeout(loadBudgetSimulation, 300);
}

async function loadBudgetSimulation() {
  const pct = document.getElementById("budget-slider").value;
  const resultEl = document.getElementById("budget-result");

  try {
    const res = await fetch(`${API_BASE}/api/budget-simulation?budget_pct=${pct / 100}`);
    if (!res.ok) {
      resultEl.innerHTML = `<p class="text-muted text-sm">Could not load simulation.</p>`;
      return;
    }
    const data = await res.json();

    const metricLabel = data.metric_name === "recall" ? "Fraud caught" : "Ring transactions captured";

    resultEl.innerHTML = `
      <div class="grid grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-4 mt-3 max-w-[700px]">
        <div>
          <div class="text-[0.72rem] uppercase tracking-wide text-muted">Transactions reviewed</div>
          <div class="font-data text-xl font-bold">${data.n_reviewed.toLocaleString()}</div>
        </div>
        <div>
          <div class="text-[0.72rem] uppercase tracking-wide text-muted">${metricLabel} (by risk order)</div>
          <div class="font-data text-xl font-bold">${fmtPct(data.model_metric)}</div>
        </div>
        <div>
          <div class="text-[0.72rem] uppercase tracking-wide text-muted">${metricLabel} (random order)</div>
          <div class="font-data text-xl font-bold">${fmtPct(data.random_metric)}</div>
        </div>
      </div>
      <p class="mt-3 text-xs text-muted max-w-[60ch] ${data.is_ground_truth_available ? "" : "border-l-[3px] border-risk-high pl-3"}">${escapeHtml(data.note)}</p>
    `;
  } catch (err) {
    console.error("Failed to load budget simulation", err);
    resultEl.innerHTML = `<p class="text-muted text-sm">Could not load simulation.</p>`;
  }
}

// --- wiring --------------------------------------------------------------

let searchDebounceTimer = null;

document.getElementById("filter-form").addEventListener("input", (e) => {
  if (e.target.id === "page-size") {
    currentLimit = Number(e.target.value);
    resetToFirstPage();
    return;
  }
  if (e.target.id === "txn-search") {
    // Debounced separately from the other filters - this fires on every
    // keystroke, unlike a discrete checkbox/number change, so it needs
    // the same "wait for a pause" treatment the budget slider gets.
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(resetToFirstPage, 300);
    return;
  }
  resetToFirstPage();
});

document.getElementById("prev-page-btn").addEventListener("click", () => {
  currentOffset = Math.max(0, currentOffset - currentLimit);
  loadCases();
});

document.getElementById("next-page-btn").addEventListener("click", () => {
  if (currentOffset + currentLimit < currentTotal) {
    currentOffset += currentLimit;
    loadCases();
  }
});

document.getElementById("budget-slider").addEventListener("input", (e) => {
  document.getElementById("budget-pct-label").textContent = `${e.target.value}%`;
  debounceBudgetFetch();
});

loadSummary();
loadCases();
loadBudgetSimulation();