function updateCounter(input, charsId, wordsId) {
  const text = input.value || "";
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  document.getElementById(charsId).textContent = text.length;
  document.getElementById(wordsId).textContent = words;
}

async function postText(formId, resultId, endpoint) {
  const form = document.getElementById(formId);
  const result = document.getElementById(resultId);
  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    renderLoading(result, "Processing analysis...");
    const payload = Object.fromEntries(new FormData(form));
    if (formId === "newsForm") {
      payload.text = [payload.headline, payload.text].filter(Boolean).join("\n\n");
    }
    try {
      renderAnalysisResult(result, await apiJson(endpoint, { method: "POST", body: JSON.stringify(payload) }));
      showToast("Analysis completed.");
    } catch (error) {
      renderError(result, error);
    }
  });
}

postText("textForm", "textResult", "/api/v1/analyze/text");
postText("newsForm", "newsResult", "/api/v1/analyze/news");

const textInput = document.getElementById("textInput");
textInput?.addEventListener("input", () => updateCounter(textInput, "textChars", "textWords"));
if (textInput) updateCounter(textInput, "textChars", "textWords");

const newsInput = document.getElementById("newsInput");
newsInput?.addEventListener("input", () => updateCounter(newsInput, "newsChars", "newsWords"));
if (newsInput) updateCounter(newsInput, "newsChars", "newsWords");

document.getElementById("textExample")?.addEventListener("click", () => {
  textInput.value = "You are not welcome in this group, and everyone should ignore your posts because they are embarrassing.";
  textInput.dispatchEvent(new Event("input"));
});

document.getElementById("newsExample")?.addEventListener("click", () => {
  const headline = document.getElementById("headlineInput");
  headline.value = "Viral post claims a miracle device can cure every illness overnight";
  newsInput.value = "The article uses urgent language, offers no verifiable sources, and asks readers to share before independent confirmation is available.";
  newsInput.dispatchEvent(new Event("input"));
});

document.querySelectorAll("[data-clear-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.clearTarget);
    if (!target) return;
    target.value = "";
    target.dispatchEvent(new Event("input"));
  });
});

document.querySelectorAll(".upload-form").forEach((form) => {
  const input = form.querySelector('input[type="file"]');
  const dropzone = form.querySelector(".dropzone");
  const preview = form.querySelector(".preview-frame");
  const meta = form.querySelector(".file-meta");
  const box = form.querySelector(".result-box");

  function resetFile() {
    input.value = "";
    meta.textContent = "";
    preview.innerHTML = `<span class="preview-empty">${titleize(form.dataset.kind)} preview</span>`;
  }

  function renderPreview(file) {
    if (!file) return resetFile();
    meta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type || "unknown type"}`;
    const url = URL.createObjectURL(file);
    if (form.dataset.kind === "image") {
      preview.innerHTML = `<img src="${url}" alt="Uploaded image preview">`;
    } else if (form.dataset.kind === "video") {
      preview.innerHTML = `<video src="${url}" controls></video>`;
    } else {
      preview.innerHTML = `<audio src="${url}" controls></audio>`;
    }
  }

  input?.addEventListener("change", () => renderPreview(input.files[0]));
  form.querySelector(".remove-file")?.addEventListener("click", resetFile);

  ["dragenter", "dragover"].forEach((name) => {
    dropzone?.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropzone?.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone?.addEventListener("drop", (event) => {
    if (!event.dataTransfer.files.length) return;
    input.files = event.dataTransfer.files;
    renderPreview(input.files[0]);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    renderLoading(box, "Uploading and analyzing...");
    try {
      const response = await fetch(form.dataset.endpoint, { method: "POST", body: new FormData(form), credentials: "same-origin" });
      const data = await response.json();
      if (!response.ok) {
        renderError(box, data);
        return;
      }
      renderAnalysisResult(box, data);
      showToast("Analysis completed.");
    } catch (_error) {
      renderError(box, { error: "upload_failed", message: "The file could not be uploaded." });
    }
  });
});

function renderLoading(target, message) {
  target.classList.remove("result-empty");
  target.innerHTML = `<div class="result-loading">${escapeHtml(message)}</div>`;
}

function renderError(target, error) {
  const message = error?.message || error?.error || "Something went wrong.";
  target.classList.remove("result-empty");
  target.innerHTML = `
    <div class="result-summary">
      <span class="risk-pill risk-needs_review">Needs Review</span>
      <strong>${escapeHtml(titleize(error?.error || "Request Failed"))}</strong>
    </div>
    <p class="result-explanation">${escapeHtml(message)}</p>
  `;
}

function confidenceValue(value) {
  if (typeof value === "number") return value <= 1 ? `${Math.round(value * 100)}%` : String(value);
  if (typeof value === "string") return value;
  return null;
}

function renderAnalysisResult(target, data) {
  const result = data.results?.[0];
  if (!result) {
    renderError(target, { error: "empty_result", message: "No result was returned." });
    return;
  }
  const confidence = Object.entries(result.confidence || {})
    .filter(([, value]) => typeof value !== "object")
    .map(([key, value]) => {
      const formatted = confidenceValue(value);
      return formatted ? `<span><strong>${escapeHtml(titleize(key))}</strong>${escapeHtml(formatted)}</span>` : "";
    })
    .join("");
  const labels = (result.labels || []).map((label) => `<span class="type-pill">${escapeHtml(titleize(label))}</span>`).join("");
  const allResults = data.results.length > 1
    ? `<div class="result-modules">${data.results.map(renderModuleMini).join("")}</div>`
    : "";
  target.classList.remove("result-empty");
  target.innerHTML = `
    <div class="result-summary">
      <span class="risk-pill risk-${escapeHtml(result.risk_category)}">${escapeHtml(titleize(result.risk_category))}</span>
      <strong>${escapeHtml(titleize(result.module))}</strong>
      <em>${escapeHtml(titleize(result.model_status))}</em>
    </div>
    <div class="label-row">${labels || '<span class="type-pill">Uncertain</span>'}</div>
    <div class="confidence-row">${confidence || "<span><strong>Confidence</strong>Not reported</span>"}</div>
    <p class="result-explanation">${escapeHtml(result.explanation)}</p>
    <div class="counter-row"><span>Analysis #${escapeHtml(data.analysis_id)}</span><span>${escapeHtml(formatDate(new Date().toISOString()))}</span></div>
    <div class="button-row"><button class="btn btn-outline-light" type="button" data-download-report="${escapeHtml(data.analysis_id)}"><i data-lucide="download"></i> Download Report</button></div>
    ${allResults}
  `;
  if (window.lucide) window.lucide.createIcons();
}

function renderModuleMini(result) {
  const confidence = result.confidence || {};
  const candidate = confidence.candidate_label;
  const classScores = ["real", "fake"]
    .filter((label) => typeof confidence[label] === "number")
    .map((label) => `${titleize(label)} ${confidenceValue(confidence[label])}`)
    .join(" · ");
  const candidateDetail = candidate
    ? `<small>Model candidate: ${escapeHtml(titleize(candidate))}${classScores ? ` · ${escapeHtml(classScores)}` : ""}</small>`
    : "";
  return `
    <article>
      <strong>${escapeHtml(titleize(result.module))}</strong>
      <span class="risk-pill risk-${escapeHtml(result.risk_category)}">${escapeHtml(titleize(result.risk_category))}</span>
      <small>${escapeHtml(titleize(result.model_status))}</small>
      ${candidateDetail}
    </article>
  `;
}

document.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-download-report]");
  if (!button) return;
  try {
    const id = button.dataset.downloadReport;
    const data = await apiJson(`/api/v1/reports/${id}`);
    downloadJson(`shieldvision-report-${id}.json`, data.report);
  } catch (_error) {
    showToast("Report could not be downloaded.");
  }
});
