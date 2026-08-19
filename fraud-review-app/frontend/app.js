// app.js - fraud-review-app frontend
// Fetches from the same-origin FastAPI backend, renders the paginated,
// sortable case queue, the case detail panel (SHAP bar chart, D3 cluster
// diagram, status controls), the risk distribution chart, and the budget
// simulator. No build step, no framework. Styling is Tailwind utility
// classes referencing CSS custom properties (defined in index.html/docs.html)
// for the structural colors, so dark mode is a single class toggle rather
// than a second copy of every color class.

const API_BASE = "";

let selectedTransactionId = null;
let budgetDebounceTimer = null;
let searchDebounceTimer = null;

// pagination state
let currentOffset = 0;
let currentLimit = 20;
let currentTotal = 0;

// sort state
let currentSortBy = "risk_score";
let currentSortDir = "desc";

// --- formatting helpers ---------------------------------------------

const RISK_BADGE_BASE =
  "inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-data text-xs font-bold";

function riskTier(score) {
  if (score >= 0.75) return { label: "Critical", cls: `${RISK_BADGE_BASE} bg-risk-critical text-black`, hex: "#D55E00" };
  if (score >= 0.5) return { label: "High", cls: `${RISK_BADGE_BASE} bg-risk-high text-black`, hex: "#E69F00" };
  if (score >= 0.25) return { label: "Medium", cls: `${RISK_BADGE_BASE} bg-risk-medium text-white`, hex: "#0072B2" };
  return { label: "Low", cls: `${RISK_BADGE_BASE} bg-risk-low text-black`, hex: "#009E73" };
}

const STATUS_LABELS = {
  reviewed: "Reviewed",
  escalated: "Escalated",
  dismissed: "Dismissed",
};

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

// --- theme (dark mode) ---------------------------------------------------

function applyThemeButtonLabel() {
  const isDark = document.documentElement.classList.contains("dark");
  const btn = document.getElementById("theme-toggle-btn");
  const label = document.getElementById("theme-toggle-label");
  if (!btn || !label) return;
  label.textContent = isDark ? "Light mode" : "Dark mode";
  btn.setAttribute("aria-pressed", String(isDark));
}

function initThemeToggle() {
  // index.html/docs.html already applied the saved/preferred theme before
  // first paint (inline script in <head>) - this just wires the button and
  // syncs its label/aria-pressed to whatever state that resulted in.
  applyThemeButtonLabel();
  const btn = document.getElementById("theme-toggle-btn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const nowDark = document.documentElement.classList.toggle("dark");
    localStorage.setItem("theme", nowDark ? "dark" : "light");
    applyThemeButtonLabel();
  });
}

// --- summary strip + risk distribution ----------------------------------

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

async function loadRiskDistribution() {
  const svgEl = document.getElementById("risk-distribution-chart");
  if (!svgEl) return;
  try {
    const res = await fetch(`${API_BASE}/api/risk-distribution`);
    if (!res.ok) return;
    const data = await res.json();
    renderRiskDistributionChart(data.buckets);
  } catch (err) {
    console.error("Failed to load risk distribution", err);
  }
}

function renderRiskDistributionChart(buckets) {
  const svgEl = document.getElementById("risk-distribution-chart");
  const width = 220;
  const height = 44;
  const gap = 4;
  const barWidth = (width - gap * (buckets.length - 1)) / buckets.length;
  const maxCount = Math.max(1, ...buckets.map((b) => b.count));

  // buckets arrive Critical->High->Medium->Low (highest risk first, same
  // order as the legend) - reversed here so the chart reads low-to-high
  // risk left-to-right, the more conventional axis direction.
  const ordered = [...buckets].reverse();

  const bars = ordered
    .map((b, i) => {
      const tier = riskTier(b.min_score + 0.01); // nudge into the bucket's own range for color lookup
      const barHeight = Math.max(2, (b.count / maxCount) * (height - 2));
      const x = i * (barWidth + gap);
      const y = height - barHeight;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" fill="${tier.hex}" rx="1.5">
        <title>${b.label}: ${b.count.toLocaleString()} case${b.count === 1 ? "" : "s"}</title>
      </rect>`;
    })
    .join("");

  svgEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svgEl.innerHTML = bars;
}

// --- case queue (paginated, sortable) ------------------------------------

async function loadCases() {
  try {
    const minRiskInput = document.getElementById("min-risk-score").value;
    const minRisk = minRiskInput === "" ? "0" : minRiskInput;
    const clusterOnly = document.getElementById("cluster-only").checked;
    const search = document.getElementById("txn-search").value.trim();
    const params = new URLSearchParams({
      min_risk_score: minRisk,
      cluster_only: String(clusterOnly),
      sort_by: currentSortBy,
      sort_dir: currentSortDir,
      limit: String(currentLimit),
      offset: String(currentOffset),
    });
    if (search) {
      params.set("search", search);
    }

    const res = await fetch(`${API_BASE}/api/cases?${params}`);
    if (!res.ok) return;
    const body = await res.json();
    currentTotal = body.total;
    renderCaseTable(body.items);
    renderPaginationControls();
  } catch (err) {
    // Deliberately covers the DOM reads above too, not just the fetch:
    // a missing expected element (e.g. index.html out of sync with a
    // newer app.js) used to throw before this function ever reached its
    // try block, silently killing the whole render with no logged error
    // at all. Now it's at least loud and diagnosable.
    console.error("Failed to load cases", err);
  }
}

function currentFilterParams() {
  // Shared by CSV export - must match loadCases()'s filter params (not
  // sort/pagination, which export intentionally ignores) so "export what
  // I'm currently looking at" actually means what it says.
  const minRiskInput = document.getElementById("min-risk-score").value;
  const minRisk = minRiskInput === "" ? "0" : minRiskInput;
  const clusterOnly = document.getElementById("cluster-only").checked;
  const search = document.getElementById("txn-search").value.trim();
  const params = new URLSearchParams({
    min_risk_score: minRisk,
    cluster_only: String(clusterOnly),
  });
  if (search) {
    params.set("search", search);
  }
  return params;
}

function resetToFirstPage() {
  currentOffset = 0;
  loadCases();
}

function updateSortIndicators() {
  document.querySelectorAll(".sort-indicator").forEach((el) => {
    const key = el.dataset.sortIndicatorFor;
    if (key === currentSortBy) {
      el.textContent = currentSortDir === "asc" ? "\u25B2" : "\u25BC"; // ▲ / ▼ (not emoji - geometric shapes)
    } else {
      el.textContent = "";
    }
  });
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

  const cellBase = "px-3 py-2 border-b border-[var(--c-line)] whitespace-nowrap";

  for (const c of cases) {
    const tier = riskTier(c.risk_score);
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.className =
      "cursor-pointer hover:bg-[var(--c-canvas)] aria-selected:bg-[#EFEDE7] dark:aria-selected:bg-[#2A2C31] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--c-ink)] focus-visible:outline-offset-[-3px]";
    tr.dataset.transactionId = String(c.TransactionID);
    tr.setAttribute("aria-selected", String(c.TransactionID === selectedTransactionId));

    const statusBadge = c.status
      ? `<span class="text-xs font-semibold">${STATUS_LABELS[c.status]}</span>`
      : `<span class="text-xs text-[var(--c-muted)]">&ndash;</span>`;

    tr.innerHTML = `
      <td class="${cellBase} font-data">${c.TransactionID}</td>
      <td class="${cellBase} font-data">${fmtAmount(c.TransactionAmt)}</td>
      <td class="${cellBase}"><span class="${tier.cls}">${tier.label} &middot; ${c.risk_score.toFixed(2)}</span></td>
      <td class="${cellBase}">${statusBadge}</td>
      <td class="${cellBase}">${c.cluster_id != null ? `<span class="text-xs text-[var(--c-muted)]">Ring #${c.cluster_id}</span>` : ""}</td>
      <td class="px-3 py-2 border-b border-[var(--c-line)] whitespace-normal min-w-[220px] text-[var(--c-muted)]">${escapeHtml(c.top_reason)}</td>
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

// --- CSV export ------------------------------------------------------

function exportCasesCsv() {
  const params = currentFilterParams();
  window.location.href = `${API_BASE}/api/cases/export?${params}`;
}

// --- case detail (incl. status controls) --------------------------------

function updateUrlHashForCase(transactionId) {
  // replaceState, not a hash assignment or pushState: a hash assignment
  // and pushState both add a browser-history entry per case selected,
  // which would make the back button step through every case a reviewer
  // clicked rather than leaving the page - replaceState updates the
  // shareable URL without polluting history.
  const url = new URL(window.location.href);
  url.hash = `case=${transactionId}`;
  history.replaceState(null, "", url);
}

function getCaseIdFromUrlHash() {
  const match = window.location.hash.match(/^#case=(\d+)$/);
  return match ? Number(match[1]) : null;
}

async function selectCase(transactionId, options = {}) {
  selectedTransactionId = transactionId;
  if (!options.skipHashUpdate) {
    updateUrlHashForCase(transactionId);
  }

  document.querySelectorAll("#case-table-body tr").forEach((tr) => {
    tr.setAttribute("aria-selected", String(Number(tr.dataset.transactionId) === transactionId));
  });

  const panel = document.getElementById("detail-panel");
  panel.innerHTML = `<p class="text-[var(--c-muted)] text-sm">Loading case ${transactionId}&hellip;</p>`;

  try {
    const res = await fetch(`${API_BASE}/api/cases/${transactionId}`);
    if (!res.ok) {
      panel.innerHTML = `<p class="text-[var(--c-muted)] text-sm">Could not load case ${transactionId}.</p>`;
      return;
    }
    const detail = await res.json();
    renderDetailPanel(detail);
  } catch (err) {
    console.error("Failed to load case detail", err);
    panel.innerHTML = `<p class="text-[var(--c-muted)] text-sm">Could not load case ${transactionId}.</p>`;
  }
}

async function setCaseStatus(transactionId, status) {
  try {
    const res = await fetch(`${API_BASE}/api/cases/${transactionId}/status`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!res.ok) {
      console.error("Failed to set case status", await res.text());
      return;
    }
    // Refresh both the detail panel (status pill) and the queue row
    // (status column) so neither view goes stale after the change.
    await selectCase(transactionId, { skipHashUpdate: true });
    await loadCases();
  } catch (err) {
    console.error("Failed to set case status", err);
  }
}

function renderStatusControls(detail) {
  const options = [
    { value: "", label: "No status" },
    { value: "reviewed", label: "Reviewed" },
    { value: "escalated", label: "Escalated" },
    { value: "dismissed", label: "Dismissed" },
  ];
  const optionsHtml = options
    .map((o) => `<option value="${o.value}" ${(detail.status || "") === o.value ? "selected" : ""}>${o.label}</option>`)
    .join("");

  return `
    <div class="mt-3">
      <label for="case-status-select" class="text-xs text-[var(--c-muted)] block mb-1">Case status</label>
      <select id="case-status-select"
              class="text-sm px-2 py-1 border border-[var(--c-line-strong)] rounded bg-[var(--c-surface)] text-[var(--c-ink)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--c-ink)]">
        ${optionsHtml}
      </select>
    </div>
  `;
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
    <p class="text-[var(--c-muted)] text-sm">${escapeHtml(detail.top_reason)}</p>
    ${renderStatusControls(detail)}
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
          <span class="bg-[var(--c-canvas)] border border-[var(--c-line)] rounded h-3.5 overflow-hidden">
            <span class="${barColor} h-full block" style="width: ${widthPct.toFixed(1)}%"></span>
          </span>
          <span class="font-data text-xs text-[var(--c-muted)] text-right">${isPositive ? "+" : ""}${f.shap_value.toFixed(2)}</span>
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
      <p class="text-[var(--c-muted)] text-sm">Shared attribute: <strong class="text-[var(--c-ink)]">${escapeHtml(ci.shared_attribute)}</strong>
        &middot; Cluster fraud rate: <span class="font-data font-semibold">${fmtPct(ci.cluster_fraud_rate)}</span>
        &middot; ${totalMembers.toLocaleString()} linked transaction${totalMembers === 1 ? "" : "s"}</p>
      <ul class="pl-4 text-sm max-h-48 overflow-y-auto">
        ${ci.member_transaction_ids.map((id) => `<li class="mb-1">Transaction ${id}</li>`).join("")}
      </ul>
      <div class="mt-3">
        <svg id="cluster-diagram" role="img" class="w-full h-[220px] border border-[var(--c-line)] rounded-md bg-[var(--c-canvas)]"
             aria-label="Network diagram of transaction ${detail.TransactionID} and up to ${Math.min(totalMembers, MAX_DIAGRAM_NODES)} of its ${totalMembers} linked transactions"></svg>
        <p class="text-xs text-[var(--c-muted)] mt-2" id="diagram-caption"></p>
      </div>
    `;
  }

  panel.innerHTML = html;

  const statusSelect = document.getElementById("case-status-select");
  if (statusSelect) {
    statusSelect.addEventListener("change", (e) => {
      setCaseStatus(detail.TransactionID, e.target.value || null);
    });
  }

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
      resultEl.innerHTML = `<p class="text-[var(--c-muted)] text-sm">Could not load simulation.</p>`;
      return;
    }
    const data = await res.json();

    const metricLabel = data.metric_name === "recall" ? "Fraud caught" : "Ring transactions captured";

    resultEl.innerHTML = `
      <div class="grid grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-4 mt-3 max-w-[700px]">
        <div>
          <div class="text-[0.72rem] uppercase tracking-wide text-[var(--c-muted)]">Transactions reviewed</div>
          <div class="font-data text-xl font-bold">${data.n_reviewed.toLocaleString()}</div>
        </div>
        <div>
          <div class="text-[0.72rem] uppercase tracking-wide text-[var(--c-muted)]">${metricLabel} (by risk order)</div>
          <div class="font-data text-xl font-bold">${fmtPct(data.model_metric)}</div>
        </div>
        <div>
          <div class="text-[0.72rem] uppercase tracking-wide text-[var(--c-muted)]">${metricLabel} (random order)</div>
          <div class="font-data text-xl font-bold">${fmtPct(data.random_metric)}</div>
        </div>
      </div>
      <p class="mt-3 text-xs text-[var(--c-muted)] max-w-[60ch]">${data.is_ground_truth_available ? "" : '<strong class="font-semibold text-[var(--c-ink)]">Note: </strong>'}${escapeHtml(data.note)}</p>
    `;
  } catch (err) {
    console.error("Failed to load budget simulation", err);
    resultEl.innerHTML = `<p class="text-[var(--c-muted)] text-sm">Could not load simulation.</p>`;
  }
}

// --- wiring --------------------------------------------------------------

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

document.querySelectorAll(".sort-header").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sortKey;
    if (currentSortBy === key) {
      currentSortDir = currentSortDir === "asc" ? "desc" : "asc";
    } else {
      currentSortBy = key;
      currentSortDir = "desc";
    }
    updateSortIndicators();
    resetToFirstPage();
  });
});

document.getElementById("export-csv-btn").addEventListener("click", exportCasesCsv);

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

initThemeToggle();
updateSortIndicators();
loadSummary();
loadRiskDistribution();
loadCases().then(() => {
  // Shareable case links: if the URL already has #case=ID on load (e.g.
  // someone followed a shared link), select it automatically - regardless
  // of whether that case happens to be on the first page of results, since
  // /api/cases/{id} doesn't require pagination context to look one up.
  const hashCaseId = getCaseIdFromUrlHash();
  if (hashCaseId != null) {
    selectCase(hashCaseId, { skipHashUpdate: true });
  }
});
loadBudgetSimulation();