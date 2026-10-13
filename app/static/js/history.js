let historyItems = [];

function firstResult(item) {
  return item.results?.[0] || {};
}

function riskWeight(risk) {
  return { high: 4, needs_review: 3, medium: 2, low: 1 }[risk] || 0;
}

function filteredHistory() {
  const query = document.getElementById("historySearch").value.toLowerCase().trim();
  const type = document.getElementById("historyType").value;
  const risk = document.getElementById("historyRisk").value;
  const sort = document.getElementById("historySort").value;
  let items = historyItems.filter((item) => {
    const result = firstResult(item);
    const haystack = [item.id, item.content_type, item.status, result.risk_category, result.model, ...(result.labels || [])]
      .join(" ")
      .toLowerCase();
    return (!query || haystack.includes(query))
      && (!type || item.content_type === type)
      && (!risk || result.risk_category === risk);
  });
  items = items.sort((a, b) => {
    if (sort === "oldest") return new Date(a.created_at) - new Date(b.created_at);
    if (sort === "risk") return riskWeight(firstResult(b).risk_category) - riskWeight(firstResult(a).risk_category);
    return new Date(b.created_at) - new Date(a.created_at);
  });
  return items;
}

function renderHistory() {
  const items = filteredHistory();
  document.getElementById("historyTotal").textContent = `${items.length} items`;
  document.getElementById("historyRows").innerHTML = items.length
    ? items.map((item) => {
      const result = firstResult(item);
      return `
        <tr>
          <td>#${escapeHtml(item.id)}</td>
          <td><span class="type-pill">${escapeHtml(titleize(item.content_type))}</span></td>
          <td>${escapeHtml(formatDate(item.created_at))}</td>
          <td><span class="risk-pill risk-${escapeHtml(result.risk_category || "pending")}">${escapeHtml(titleize(result.risk_category || "pending"))}</span></td>
          <td>${escapeHtml(titleize((result.labels || ["uncertain"])[0]))}</td>
          <td>${escapeHtml(titleize(result.model_status || item.status))}</td>
          <td>
            <button class="btn btn-sm btn-outline-light" type="button" data-report="${escapeHtml(item.id)}">Report</button>
            <button class="btn btn-sm btn-outline-light" type="button" data-delete="${escapeHtml(item.id)}">Delete</button>
          </td>
        </tr>
      `;
    }).join("")
    : `<tr><td colspan="7" class="empty-state">No saved analyses match the current filters.</td></tr>`;
}

async function loadHistory() {
  const params = new URLSearchParams({ per_page: "50" });
  const data = await apiJson(`/api/v1/analyses?${params}`);
  historyItems = data.items || [];
  const query = new URLSearchParams(window.location.search).get("q");
  if (query) document.getElementById("historySearch").value = query;
  renderHistory();
}

["historySearch", "historyType", "historyRisk", "historySort"].forEach((id) => {
  document.getElementById(id)?.addEventListener("input", renderHistory);
});

document.getElementById("historyRows")?.addEventListener("click", async (event) => {
  const report = event.target.closest("[data-report]");
  const remove = event.target.closest("[data-delete]");
  if (report) {
    const data = await apiJson(`/api/v1/reports/${report.dataset.report}`);
    downloadJson(`shieldvision-report-${report.dataset.report}.json`, data.report);
  }
  if (remove && window.confirm("Delete this analysis?")) {
    await apiJson(`/api/v1/analyses/${remove.dataset.delete}`, { method: "DELETE" });
    historyItems = historyItems.filter((item) => String(item.id) !== String(remove.dataset.delete));
    renderHistory();
    showToast("Analysis deleted.");
  }
});

loadHistory().catch(() => {
  document.getElementById("historyRows").innerHTML = `<tr><td colspan="7" class="empty-state">History could not be loaded.</td></tr>`;
});
