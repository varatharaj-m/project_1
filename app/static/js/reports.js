let latestReportId = null;
let reportRiskChart;
let reportTypeChart;

function drawReportChart(target, chart, labels, values) {
  if (chart) chart.destroy();
  const hasData = values.some((value) => value > 0);
  return new Chart(document.getElementById(target), {
    type: "bar",
    data: {
      labels: hasData ? labels.map(titleize) : ["No data"],
      datasets: [{ data: hasData ? values : [0], backgroundColor: ["#4f7cff", "#38bdf8", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444"], borderRadius: 8 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8", precision: 0 }, grid: { color: "rgba(148,163,184,.08)" }, beginAtZero: true },
      },
      plugins: { legend: { display: false } },
    },
  });
}

async function loadReports() {
  const stats = await apiJson("/api/v1/dashboard/stats");
  const history = await apiJson("/api/v1/analyses?per_page=8");
  const byType = stats.analyses_by_type || {};
  latestReportId = history.items?.[0]?.id || null;
  document.getElementById("reportTotal").textContent = stats.total_analyses || 0;
  document.getElementById("reportHigh").textContent = stats.high_priority || 0;
  document.getElementById("reportModels").textContent = stats.available_models || 0;
  document.getElementById("latestReportId").textContent = latestReportId ? `#${latestReportId}` : "-";
  document.getElementById("reportTypes").textContent = Object.keys(byType).length;
  reportRiskChart = drawReportChart("reportRiskChart", reportRiskChart, Object.keys(stats.risk_distribution || {}), Object.values(stats.risk_distribution || {}));
  reportTypeChart = drawReportChart("reportTypeChart", reportTypeChart, Object.keys(byType), Object.values(byType));
  document.getElementById("reportRows").innerHTML = history.items?.length
    ? history.items.map((item) => {
      const result = item.results?.[0] || {};
      return `
        <tr>
          <td>#${escapeHtml(item.id)}</td>
          <td><span class="type-pill">${escapeHtml(titleize(item.content_type))}</span></td>
          <td><span class="risk-pill risk-${escapeHtml(result.risk_category || "pending")}">${escapeHtml(titleize(result.risk_category || "pending"))}</span></td>
          <td>${escapeHtml(result.model || "Not reported")}</td>
          <td>${escapeHtml(formatDate(item.created_at))}</td>
          <td><button class="btn btn-sm btn-outline-light" type="button" data-export="${escapeHtml(item.id)}">Export</button></td>
        </tr>
      `;
    }).join("")
    : `<tr><td colspan="6" class="empty-state">No analyses are available for reporting yet.</td></tr>`;
}

async function exportReport(id) {
  if (!id) {
    showToast("No report is available yet.");
    return;
  }
  const data = await apiJson(`/api/v1/reports/${id}`);
  downloadJson(`shieldvision-report-${id}.json`, data.report);
}

document.getElementById("downloadLatestReport")?.addEventListener("click", () => exportReport(latestReportId));
document.getElementById("reportRows")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-export]");
  if (button) exportReport(button.dataset.export);
});

loadReports().catch(() => {
  document.getElementById("reportRows").innerHTML = `<tr><td colspan="6" class="empty-state">Report data could not be loaded.</td></tr>`;
});
