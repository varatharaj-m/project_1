async function apiJson(url, options = {}) {
  const headers = options.body instanceof FormData
    ? { ...(options.headers || {}) }
    : { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(url, { credentials: "same-origin", headers, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw data;
  return data;
}

function pretty(data) {
  return JSON.stringify(data, null, 2);
}

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      })[character],
  );
}

function titleize(value) {
  return String(value || "unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatDate(value) {
  if (!value) return "Not recorded";
  return new Date(value).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function showToast(message) {
  const stack = document.getElementById("toastStack");
  if (!stack) return;
  const toast = document.createElement("div");
  toast.className = "toast-message";
  toast.textContent = message;
  stack.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function downloadJson(filename, data) {
  const blob = new Blob([pretty(data)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function refreshSystemStatus() {
  const status = document.getElementById("systemStatus");
  if (!status) return;
  try {
    const data = await apiJson("/api/v1/health");
    const models = Object.values(data.models || {});
    const available = models.filter((model) => model.status === "available").length;
    status.innerHTML = `<span></span> ${available}/${models.length} services available`;
  } catch (_error) {
    status.innerHTML = `<span></span> Model status unavailable`;
  }
}

document.getElementById("mobileMenu")?.addEventListener("click", () => {
  document.getElementById("sidebar")?.classList.toggle("open");
});

document.querySelectorAll(".side-link").forEach((link) => {
  link.addEventListener("click", () => document.getElementById("sidebar")?.classList.remove("open"));
});

document.getElementById("themeToggle")?.addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next;
  localStorage.setItem("shieldvision-theme", next);
});

document.getElementById("globalSearch")?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  const query = event.currentTarget.value.trim();
  window.location.href = query ? `/history?q=${encodeURIComponent(query)}` : "/history";
});

const savedTheme = localStorage.getItem("shieldvision-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

if (window.lucide) window.lucide.createIcons();
refreshSystemStatus();
