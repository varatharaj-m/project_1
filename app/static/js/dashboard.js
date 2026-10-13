let riskChart;
let typeChart;
let activityChart;

const chartColors = ["#4f7cff", "#38bdf8", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444"];

function drawDoughnut(target, chart, labels, values) {
  if (chart) chart.destroy();
  const hasData = values.some((value) => value > 0);
  return new Chart(document.getElementById(target), {
    type: "doughnut",
    data: {
      labels: hasData ? labels.map(titleize) : ["No data"],
      datasets: [{ data: hasData ? values : [1], backgroundColor: hasData ? chartColors : ["rgba(148,163,184,.18)"], borderWidth: 0 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", boxWidth: 10, padding: 14 } } },
    },
  });
}

function drawActivity(labels, values) {
  if (activityChart) activityChart.destroy();
  const hasData = values.some((value) => value > 0);
  activityChart = new Chart(document.getElementById("activityChart"), {
    type: "line",
    data: {
      labels: hasData ? labels : ["No data"],
      datasets: [{
        label: "Analyses",
        data: hasData ? values : [0],
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56,189,248,.16)",
        fill: true,
        tension: .35,
        pointRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,.08)" } },
        y: { ticks: { color: "#94a3b8", precision: 0 }, grid: { color: "rgba(148,163,184,.08)" }, beginAtZero: true },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderModels(modelHealth) {
  const entries = Object.entries(modelHealth || {});
  document.getElementById("modelHealth").innerHTML = entries.length
    ? entries
      .map(([key, state]) => {
        const status = state.status || "unavailable";
        return `
        <article class="model-chip model-${escapeHtml(status)}">
          <div>
            <strong>${escapeHtml(titleize(key))}</strong>
            <span>${escapeHtml(state.name || "not configured")}</span>
          </div>
          <em>${escapeHtml(status)}</em>
        </article>
      `;
      })
      .join("")
    : `<div class="empty-state">Model health has not been reported yet.</div>`;
}

function renderRecent(recent) {
  document.getElementById("historyCount").textContent = `${recent.length} recent`;
  document.getElementById("recentRows").innerHTML =
    recent
      .map((item) => {
        const first = item.results?.[0];
        const risk = first ? first.risk_category : "pending";
        return `
      <tr>
        <td>#${escapeHtml(item.id)}</td>
        <td><span class="type-pill">${escapeHtml(titleize(item.content_type))}</span></td>
        <td>${escapeHtml(titleize(item.status))}</td>
        <td><span class="risk-pill risk-${escapeHtml(risk)}">${escapeHtml(titleize(risk))}</span></td>
        <td>${escapeHtml(first?.model || "pending")}</td>
        <td><button class="btn btn-sm btn-outline-light" type="button" data-report-id="${escapeHtml(item.id)}">Download</button></td>
      </tr>
    `;
      })
      .join("") ||
    `<tr><td colspan="6" class="empty-state">No analyses yet. Run a text, news, or media scan to populate this queue.</td></tr>`;
}

function renderFindings(items) {
  document.getElementById("findingCount").textContent = `${items.length} findings`;
  document.getElementById("highRiskRows").innerHTML = items.length
    ? items.map((item) => `
      <article class="finding-item">
        <div>
          <strong>#${escapeHtml(item.analysis_id)} ${escapeHtml(titleize(item.content_type))}</strong>
          <span>${escapeHtml(titleize(item.module))} · ${escapeHtml(formatDate(item.created_at))}</span>
        </div>
        <span class="risk-pill risk-${escapeHtml(item.risk_category)}">${escapeHtml(titleize(item.risk_category))}</span>
      </article>
    `).join("")
    : `<div class="empty-state">No high-risk findings are currently stored.</div>`;
}

async function downloadReport(id) {
  const data = await apiJson(`/api/v1/reports/${id}`);
  downloadJson(`shieldvision-report-${id}.json`, data.report);
}

async function loadStats() {
  const data = await apiJson("/api/v1/dashboard/stats");
  const byType = data.analyses_by_type || {};
  document.getElementById("totalAnalyses").textContent = data.total_analyses;
  document.getElementById("highPriority").textContent = data.high_priority;
  document.getElementById("imageVideoCount").textContent = (byType.image || 0) + (byType.video || 0);
  document.getElementById("audioTextCount").textContent = (byType.audio || 0) + (byType.text || 0) + (byType.news || 0);
  document.getElementById("availableModels").textContent = data.available_models || 0;
  document.getElementById("modelCount").textContent = `${Object.keys(data.model_health || {}).length} registered`;

  const riskLabels = Object.keys(data.risk_distribution || {});
  const typeLabels = Object.keys(byType);
  const activityLabels = Object.keys(data.activity_by_day || {}).sort();
  riskChart = drawDoughnut("riskChart", riskChart, riskLabels, Object.values(data.risk_distribution || {}));
  typeChart = drawDoughnut("typeChart", typeChart, typeLabels, Object.values(byType));
  drawActivity(activityLabels, activityLabels.map((key) => data.activity_by_day[key]));
  renderModels(data.model_health);
  renderRecent(data.recent || []);
  renderFindings(data.high_risk_recent || []);
  if (window.lucide) window.lucide.createIcons();
}

document.getElementById("refreshStats")?.addEventListener("click", loadStats);
document.getElementById("recentRows")?.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-report-id]");
  if (!button) return;
  try {
    await downloadReport(button.dataset.reportId);
  } catch (_error) {
    showToast("Report could not be downloaded.");
  }
});

loadStats().catch((error) => {
  document.getElementById("recentRows").innerHTML =
    `<tr><td colspan="6">${escapeHtml(error.error || "Unable to load dashboard.")}</td></tr>`;
});
