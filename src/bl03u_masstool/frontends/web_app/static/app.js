const PROJECT_SETTINGS_STORAGE_KEY = "bl03u_web_project_settings_v1";

function readStorage(key, fallback = "") {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    if (value == null || value === "") localStorage.removeItem(key);
    else localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

const state = {
  projectSettings: loadWebProjectSettings(),
  jobId: null,
  summary: null,
  result: null,
  currentMz: null,
  currentCurve: null,
  currentFit: null,
  tableMode: "species",
  allRows: [],
  picsRows: [],
  currentPics: null,
  fitMode: "auto",
  candidateRows: [],
  selectedCandidateIds: new Set(),
  candidateCoefficients: {},
  candidateCurves: {},
  manualPreviewFit: null,
  previewFrame: null,
  lockedCandidateIds: new Set(),
  picsLibraryId: readStorage("bl03u_pics_library_id"),
  progressValue: 0,
  uploadProgressValue: 0,
  logEntries: [],
  uploadLogEntries: [],
  picsFilterText: "",
  candidateFilterText: "",
  candidateLoading: false,
  resizeFrame: null,
};

const API_BASE_STORAGE_KEY = "bl03u_api_base_url";

function resolveApiBase() {
  const defaultBase = window.location.origin && window.location.origin !== "null"
    ? window.location.origin
    : "http://127.0.0.1:8000";
  let queryBase = "";
  try {
    const params = new URLSearchParams(window.location.search);
    queryBase = params.get("api") || params.get("api_base") || "";
  } catch {
    queryBase = "";
  }
  let storedBase = readStorage(API_BASE_STORAGE_KEY);
  const globalBase = typeof window.BL03U_API_BASE_URL === "string" ? window.BL03U_API_BASE_URL : "";
  const base = queryBase || globalBase || storedBase || defaultBase;
  if (queryBase) {
    writeStorage(API_BASE_STORAGE_KEY, queryBase);
  }
  return base.replace(/\/+$/, "");
}

const API_BASE = resolveApiBase();

function apiUrl(path) {
  const value = String(path || "");
  if (/^https?:\/\//i.test(value)) return value;
  const normalizedBase = API_BASE.endsWith("/") ? API_BASE : `${API_BASE}/`;
  const normalizedPath = value.startsWith("/") ? value.slice(1) : value;
  return new URL(normalizedPath, normalizedBase).toString();
}

const $ = (id) => document.getElementById(id);

function defaultWebProjectSettings() {
  return {
    version: 1,
    project: {
      project_name: "",
      system: "",
      description: "",
      output_dir: "output",
    },
    data_sources: {
      pie_scan_folder: "tests/fixtures/bl03u_sample/C6F11O2H/PIE_Scan/1050",
      pics_database_path: "",
      manual_peak_file: "config/peak_integration.yaml",
    },
    normalization: {
      pie_photon_mode: "first",
      light_source: "io",
      mass_discrimination: 1,
    },
    function_params: {
      pie: {
        energy_decimals: 1,
        recursive: true,
        prefer_gaussian: true,
        multi_folder_mode: false,
        merge_method: "low_energy_dominant",
      },
      pics: {
        query_limit: 100,
        tolerance: 0,
      },
    },
  };
}

function mergeWebProjectSettings(stored) {
  const defaults = defaultWebProjectSettings();
  const source = stored && typeof stored === "object" ? stored : {};
  return {
    ...defaults,
    ...source,
    project: { ...defaults.project, ...(source.project || {}) },
    data_sources: { ...defaults.data_sources, ...(source.data_sources || {}) },
    normalization: { ...defaults.normalization, ...(source.normalization || {}) },
    function_params: {
      ...defaults.function_params,
      ...(source.function_params || {}),
      pie: { ...defaults.function_params.pie, ...((source.function_params || {}).pie || {}) },
      pics: { ...defaults.function_params.pics, ...((source.function_params || {}).pics || {}) },
    },
  };
}

function loadWebProjectSettings() {
  try {
    const raw = readStorage(PROJECT_SETTINGS_STORAGE_KEY);
    return mergeWebProjectSettings(raw ? JSON.parse(raw) : null);
  } catch {
    return defaultWebProjectSettings();
  }
}

function saveWebProjectSettings(settings = state.projectSettings) {
  try {
    return writeStorage(PROJECT_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    return false;
  }
}

function setFieldValue(id, value) {
  const field = $(id);
  if (!field) return;
  if (field.type === "checkbox") field.checked = Boolean(value);
  else field.value = value ?? "";
}

function fieldValue(id, fallback = "") {
  const field = $(id);
  if (!field) return fallback;
  if (field.type === "checkbox") return field.checked;
  return field.value;
}

function numericFieldValue(id, fallback, { min = -Infinity, max = Infinity } = {}) {
  const value = Number(fieldValue(id, fallback));
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function applyProjectSettingsToForms() {
  const settings = state.projectSettings;
  setFieldValue("project-name", settings.project.project_name);
  setFieldValue("project-system", settings.project.system);
  setFieldValue("project-description", settings.project.description);
  setFieldValue("project-output-dir", settings.project.output_dir);
  setFieldValue("folder", settings.data_sources.pie_scan_folder);
  setFieldValue("manual-peak-path", settings.data_sources.manual_peak_file);
  setFieldValue("pics-database-path", settings.data_sources.pics_database_path);
  setFieldValue("photon-mode", settings.normalization.pie_photon_mode);
  setFieldValue("light-source", settings.normalization.light_source);
  setFieldValue("mass-discrimination", settings.normalization.mass_discrimination);
  setFieldValue("energy-decimals", settings.function_params.pie.energy_decimals);
  setFieldValue("recursive", settings.function_params.pie.recursive);
  setFieldValue("gaussian", settings.function_params.pie.prefer_gaussian);
  setFieldValue("project-pics-tolerance", settings.function_params.pics.tolerance);
  setFieldValue("project-pics-limit", settings.function_params.pics.query_limit);
  setFieldValue("pics-tolerance", settings.function_params.pics.tolerance);
  setFieldValue("pics-limit", settings.function_params.pics.query_limit);
  updateProjectSettingSummaries();
}

function collectProjectSettingsFromForms() {
  const current = state.projectSettings;
  return mergeWebProjectSettings({
    ...current,
    project: {
      project_name: fieldValue("project-name").trim(),
      system: fieldValue("project-system").trim(),
      description: fieldValue("project-description").trim(),
      output_dir: fieldValue("project-output-dir", "output").trim() || "output",
    },
    data_sources: {
      pie_scan_folder: fieldValue("folder").trim(),
      manual_peak_file: fieldValue("manual-peak-path").trim(),
      pics_database_path: fieldValue("pics-database-path").trim(),
    },
    normalization: {
      pie_photon_mode: fieldValue("photon-mode", "first"),
      light_source: fieldValue("light-source", "io"),
      mass_discrimination: numericFieldValue("mass-discrimination", 1, { min: 0.000001 }),
    },
    function_params: {
      ...current.function_params,
      pie: {
        ...current.function_params.pie,
        energy_decimals: numericFieldValue("energy-decimals", 1, { min: 0, max: 6 }),
        recursive: Boolean(fieldValue("recursive", true)),
        prefer_gaussian: Boolean(fieldValue("gaussian", true)),
      },
      pics: {
        ...current.function_params.pics,
        tolerance: numericFieldValue("project-pics-tolerance", 0, { min: 0 }),
        query_limit: numericFieldValue("project-pics-limit", 100, { min: 1, max: 500 }),
      },
    },
  });
}

function settingsStatus(text, kind = "") {
  const status = $("project-settings-status");
  if (!status) return;
  status.textContent = text;
  status.dataset.status = kind;
}

function updateProjectSettingSummaries() {
  const settings = state.projectSettings;
  const projectName = settings.project.project_name || "未命名项目";
  const pieSummary = $("pie-settings-summary");
  if (pieSummary) {
    pieSummary.textContent = `${projectName}；${settings.data_sources.pie_scan_folder || "未设置 PIE 目录"}；${settings.normalization.light_source} / D=${settings.normalization.mass_discrimination}；能量小数 ${settings.function_params.pie.energy_decimals}`;
  }
  const libraryText = state.picsLibraryId ? `临时工作库 ${state.picsLibraryId.slice(0, 8)}` : "服务器维护库";
  const picsSummary = $("pics-settings-summary");
  if (picsSummary) {
    picsSummary.textContent = `${libraryText}；容差 ${settings.function_params.pics.tolerance}；最多 ${settings.function_params.pics.query_limit} 条`;
  }
}

function commitProjectSettings(save = false) {
  state.projectSettings = collectProjectSettingsFromForms();
  if (save) {
    const ok = saveWebProjectSettings();
    settingsStatus(ok ? "设置已保存并应用。" : "设置已应用，但浏览器无法保存。", ok ? "success" : "error");
  } else {
    settingsStatus("设置已应用到当前工具。", "success");
  }
  applyProjectSettingsToForms();
}

function setTabIndex(selector, activePredicate) {
  document.querySelectorAll(selector).forEach((item) => {
    item.tabIndex = activePredicate(item) ? 0 : -1;
  });
}

function bindRovingControls(selector) {
  document.querySelectorAll(selector).forEach((control) => {
    control.addEventListener("keydown", (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', ' ', 'Enter'].includes(event.key)) return;
      const controls = Array.from(document.querySelectorAll(selector)).filter((item) => !item.disabled && !item.hidden && item.offsetParent !== null);
      const currentIndex = controls.indexOf(event.currentTarget);
      if (currentIndex < 0) return;
      if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        event.currentTarget.click();
        return;
      }
      event.preventDefault();
      let nextIndex = currentIndex;
      if (event.key === 'Home') nextIndex = 0;
      else if (event.key === 'End') nextIndex = controls.length - 1;
      else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (currentIndex + 1) % controls.length;
      else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (currentIndex - 1 + controls.length) % controls.length;
      controls[nextIndex].focus();
    });
  });
}

function switchProjectSettingsTab(tabId) {
  document.querySelectorAll(".settings-section").forEach((section) => {
    const active = section.id === tabId;
    section.classList.toggle("active", active);
    section.classList.toggle("hidden", !active);
    section.hidden = !active;
  });
  document.querySelectorAll(".settings-tab-button").forEach((button) => {
    const active = button.dataset.settingsTab === tabId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
}

function editProjectSettings(tabId) {
  switchPage("project-page");
  switchProjectSettingsTab(tabId);
}

function statusPresentation(status) {
  const map = {
    idle: ["idle", "Idle"],
    queued: ["running", "Queued"],
    running: ["running", "Running"],
    done: ["done", "Done"],
    error: ["error", "Error"],
  };
  return map[status] || ["idle", String(status || "Idle")];
}

function setStatusView(config, status, text) {
  const pill = $(config.pillId);
  const label = $(config.textId);
  const [className, statusLabel] = statusPresentation(status);
  if (pill) {
    pill.className = `pill ${className}`;
    pill.textContent = statusLabel;
  }
  if (label) label.textContent = text;
}

function setProgressView(config, value, step, status = "running", extra = "") {
  const clamped = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  state[config.stateKey] = clamped;
  const bar = $(config.barId);
  if (!bar) return;
  bar.style.width = `${clamped}%`;
  bar.classList.toggle("error", status === "error");
  const percent = $(config.percentId);
  if (percent) percent.textContent = `${clamped}%`;
  const stepLabel = $(config.stepId);
  const stepText = step || config.defaultStep;
  if (stepLabel) stepLabel.textContent = stepText;
  if (config.extraId) {
    const extraLabel = $(config.extraId);
    if (extraLabel) extraLabel.textContent = extra || "--";
  }
  const meter = bar.closest(".progress-meter");
  if (meter) {
    meter.setAttribute("role", "progressbar");
    meter.setAttribute("aria-valuemin", "0");
    meter.setAttribute("aria-valuemax", "100");
    meter.setAttribute("aria-valuenow", String(clamped));
    meter.setAttribute("aria-valuetext", `${stepText} ${clamped}%`);
  }
}

function appendLog(config, message) {
  const stamp = new Date().toLocaleTimeString();
  const text = String(message || "").trim();
  if (!text) return;
  if (state[config.stateKey].length && state[config.stateKey][state[config.stateKey].length - 1].endsWith(`  ${text}`)) {
    return;
  }
  state[config.stateKey].push(`${stamp}  ${text}`);
  state[config.stateKey] = state[config.stateKey].slice(-10);
  const view = $(config.viewId);
  if (!view) return;
  view.textContent = state[config.stateKey].join("\n");
  view.scrollTop = view.scrollHeight;
}

const pieStatusView = {
  pillId: "status-pill",
  textId: "status-text",
  barId: "progress-bar",
  percentId: "progress-percent",
  stepId: "progress-step",
  extraId: "progress-elapsed",
  stateKey: "progressValue",
  defaultStep: "等待任务",
  logStateKey: "logEntries",
  logViewId: "progress-log",
};

const uploadStatusView = {
  pillId: "upload-status-pill",
  textId: "upload-status-text",
  barId: "upload-progress-bar",
  percentId: "upload-progress-percent",
  stepId: "upload-progress-step",
  stateKey: "uploadProgressValue",
  defaultStep: "等待上传",
  logStateKey: "uploadLogEntries",
  logViewId: "upload-progress-log",
};

function setStatus(status, text) {
  setStatusView(pieStatusView, status, text);
}

function setProgress(value, step, status = "running", elapsedText = "") {
  setProgressView(pieStatusView, value, step, status, elapsedText);
}

function log(message) {
  appendLog({ stateKey: pieStatusView.logStateKey, viewId: pieStatusView.logViewId }, message);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value, digits = 6) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return Math.abs(number) >= 1000 ? number.toFixed(2) : Number(number.toPrecision(digits)).toString();
}

function fitCanvasText(ctx, text, maxWidth) {
  const raw = String(text ?? "");
  if (ctx.measureText(raw).width <= maxWidth) return raw;
  let clipped = raw;
  while (clipped.length > 4 && ctx.measureText(`${clipped}...`).width > maxWidth) {
    clipped = clipped.slice(0, -1);
  }
  return `${clipped}...`;
}

function cssToken(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function chartTheme() {
  return {
    axis: cssToken("--chart-axis", "#c8d2df"),
    grid: cssToken("--chart-grid", "#e6edf5"),
    text: cssToken("--chart-text", "#526174"),
    primary: cssToken("--chart-series-primary", "#2563eb"),
    fit: cssToken("--chart-series-fit", "#f97316"),
    alts: [
      cssToken("--chart-series-alt-1", "#059669"),
      cssToken("--chart-series-alt-2", "#7c3aed"),
      cssToken("--chart-series-alt-3", "#dc2626"),
    ],
    title: cssToken("--ink", "#111827"),
  };
}

function setupChartCanvas(canvas, smallHeight, largeHeight) {
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(220, parent.clientWidth - 24);
  const height = width < 560 ? smallHeight : largeHeight;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  return { ctx, width, height };
}

function chartBounds(xValues, yValues, minY = 0, minXPad = 0.1, minYPad = 0.1) {
  let xMin = Math.min(...xValues);
  let xMax = Math.max(...xValues);
  let yMin = Math.min(minY, Math.min(...yValues));
  let yMax = Math.max(...yValues);
  const xPad = Math.max(minXPad, (xMax - xMin) * 0.08);
  const yPad = Math.max(minYPad, (yMax - yMin) * 0.12);
  xMin -= xPad;
  xMax += xPad;
  yMax += yPad;
  if (xMin === xMax) xMax = xMin + 1;
  if (yMin === yMax) yMax = yMin + 1;
  return { xMin, xMax, yMin, yMax };
}

function chartScales(width, height, margin, bounds) {
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  return {
    plotW,
    plotH,
    sx: (value) => margin.left + ((value - bounds.xMin) / (bounds.xMax - bounds.xMin)) * plotW,
    sy: (value) => margin.top + plotH - ((value - bounds.yMin) / (bounds.yMax - bounds.yMin)) * plotH,
  };
}

function chartMargin(ctx, width, top, yTicks) {
  const yLabelWidth = Math.max(...yTicks.map((value) => ctx.measureText(fmt(value, 3)).width));
  return width < 360
    ? { left: Math.max(64, Math.ceil(yLabelWidth) + 42), right: 12, top, bottom: 48 }
    : { left: Math.max(90, Math.ceil(yLabelWidth) + 50), right: 22, top, bottom: 54 };
}

function drawAxes(ctx, width, height, margin, scales, bounds, yTicks, xLabel, yLabel, theme) {
  ctx.strokeStyle = theme.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + scales.plotH);
  ctx.lineTo(margin.left + scales.plotW, margin.top + scales.plotH);
  ctx.stroke();

  ctx.fillStyle = theme.text;
  ctx.textAlign = "right";
  yTicks.forEach((value) => {
    const y = scales.sy(value);
    ctx.strokeStyle = theme.grid;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + scales.plotW, y);
    ctx.stroke();
    ctx.fillText(fmt(value, 3), margin.left - 12, y + 4);
  });
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i += 1) {
    const value = bounds.xMin + ((bounds.xMax - bounds.xMin) * i) / 5;
    ctx.fillText(fmt(value, 4), scales.sx(value), height - 24);
  }
  ctx.fillText(xLabel, margin.left + scales.plotW / 2, height - 7);
  ctx.save();
  ctx.translate(18, margin.top + scales.plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.textAlign = "start";
}

function drawLineSeries(ctx, series, scales) {
  series.forEach((item) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.dash ? 1.6 : 2.4;
    ctx.setLineDash(item.dash ? [6, 4] : []);
    ctx.beginPath();
    let started = false;
    const count = Math.min(item.x.length, item.y.length);
    for (let i = 0; i < count; i += 1) {
      const x = Number(item.x[i]);
      const y = Number(item.y[i]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (!started) {
        ctx.moveTo(scales.sx(x), scales.sy(y));
        started = true;
      } else {
        ctx.lineTo(scales.sx(x), scales.sy(y));
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (item.points) {
      ctx.fillStyle = item.color;
      for (let i = 0; i < count; i += 1) {
        const x = Number(item.x[i]);
        const y = Number(item.y[i]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        ctx.beginPath();
        ctx.arc(scales.sx(x), scales.sy(y), item.pointRadius || 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });
}

function legendRows(ctx, series, availableWidth) {
  let rows = 1;
  let rowWidth = 0;
  series.forEach((item) => {
    const itemWidth = Math.min(260, ctx.measureText(String(item.label)).width + 50);
    if (rowWidth > 0 && rowWidth + itemWidth > availableWidth) {
      rows += 1;
      rowWidth = 0;
    }
    rowWidth += itemWidth;
  });
  return rows;
}

function drawLegend(ctx, series, width, margin, availableWidth, theme) {
  let legendX = margin.left;
  let legendY = 22;
  series.forEach((item) => {
    const label = fitCanvasText(ctx, item.label, Math.min(220, availableWidth - 44));
    const itemWidth = Math.min(260, ctx.measureText(label).width + 50);
    if (legendX > margin.left && legendX + itemWidth > width - margin.right) {
      legendX = margin.left;
      legendY += 20;
    }
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 22, legendY);
    ctx.stroke();
    ctx.fillStyle = theme.title;
    ctx.fillText(label, legendX + 28, legendY + 4);
    legendX += itemWidth;
  });
}

function validNumbers(values) {
  return values.map(Number).filter(Number.isFinite);
}

function yTicksFor(bounds) {
  return Array.from({ length: 6 }, (_, i) => bounds.yMin + ((bounds.yMax - bounds.yMin) * i) / 5);
}

function interpolatePieSeriesColor(index, theme) {
  return theme.alts[index % theme.alts.length];
}

function interpolateSeries(sourceX, sourceY, targetX) {
  const pairs = sourceX
    .map((x, index) => [Number(x), Number(sourceY[index])])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    .sort((a, b) => a[0] - b[0]);
  if (pairs.length < 2) return targetX.map(() => 0);
  return targetX.map((value) => {
    const x = Number(value);
    if (!Number.isFinite(x) || x < pairs[0][0] || x > pairs[pairs.length - 1][0]) return 0;
    let right = 1;
    while (right < pairs.length && pairs[right][0] < x) right += 1;
    if (right >= pairs.length) return pairs[pairs.length - 1][1];
    const [x0, y0] = pairs[right - 1];
    const [x1, y1] = pairs[right];
    if (x1 === x0) return y0;
    return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  });
}

function solveNonnegativeLeastSquares(design, target, lockedCoefficients = {}) {
  const rowCount = target.length;
  const colCount = design.length;
  const coefficients = Array(colCount).fill(0);
  const locked = new Set(Object.keys(lockedCoefficients).map(Number));
  for (const [indexText, value] of Object.entries(lockedCoefficients)) {
    const index = Number(indexText);
    if (index >= 0 && index < colCount) coefficients[index] = Math.max(0, Number(value) || 0);
  }
  const fitted = () => {
    const values = Array(rowCount).fill(0);
    for (let col = 0; col < colCount; col += 1) {
      const coefficient = coefficients[col];
      if (!coefficient) continue;
      for (let row = 0; row < rowCount; row += 1) {
        values[row] += design[col][row] * coefficient;
      }
    }
    return values;
  };
  for (let iter = 0; iter < 180; iter += 1) {
    let maxChange = 0;
    for (let col = 0; col < colCount; col += 1) {
      if (locked.has(col)) continue;
      const basis = design[col];
      let numerator = 0;
      let denominator = 0;
      const current = coefficients[col];
      const currentFit = fitted();
      for (let row = 0; row < rowCount; row += 1) {
        numerator += basis[row] * (target[row] - currentFit[row] + basis[row] * current);
        denominator += basis[row] * basis[row];
      }
      if (denominator <= 0) continue;
      const next = Math.max(0, numerator / denominator);
      maxChange = Math.max(maxChange, Math.abs(next - current));
      coefficients[col] = next;
    }
    if (maxChange < 1e-6) break;
  }
  return coefficients;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(apiUrl(url), options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || data.error || response.statusText);
    error.status = response.status;
    error.detail = data.detail || data.error || "";
    throw error;
  }
  return data;
}

function isMissingPicsLibraryError(error) {
  if (!state.picsLibraryId || !error) return false;
  const text = String(error.detail || error.message || "");
  return (
    error.status === 404
    && (
      text.includes("PICS library not found")
      || text.includes("临时 PICS 工作库")
      || text.includes("临时工作库")
      || text.includes("工作库不存在")
    )
  );
}

function handleMissingPicsLibrary(error, writer = log) {
  if (!isMissingPicsLibraryError(error)) return false;
  const shortId = state.picsLibraryId.slice(0, 8);
  setPicsLibrary("");
  const message = `临时工作库 ${shortId} 已失效，已切回服务器维护库`;
  writer(message);
  return true;
}

function payloadFromForm() {
  state.projectSettings = collectProjectSettingsFromForms();
  const settings = state.projectSettings;
  const peakSource = $("peak-source").value;
  return {
    folder: settings.data_sources.pie_scan_folder.trim(),
    library_id: state.picsLibraryId || null,
    peak_source: peakSource,
    manual_peak_path: peakSource === "manual" ? settings.data_sources.manual_peak_file.trim() || null : null,
    target_mz: $("target-mz").value.trim() || null,
    recursive: Boolean(settings.function_params.pie.recursive),
    energy_decimals: Number(settings.function_params.pie.energy_decimals || 1),
    gaussian: Boolean(settings.function_params.pie.prefer_gaussian),
    photon_mode: settings.normalization.pie_photon_mode,
    light_source: settings.normalization.light_source,
    mass_discrimination: Number(settings.normalization.mass_discrimination || 1),
  };
}

function currentFitButtonLabel() {
  if (state.fitMode !== "manual") return "拟合当前曲线";
  return $("coefficient-mode").value === "manual" ? "应用手动系数" : "按候选拟合";
}

function updateFitActionState() {
  const selectedCount = selectedCandidateIds().length;
  const lockedCount = state.lockedCandidateIds.size;
  const label = currentFitButtonLabel();
  const fitButton = $("fit-button");
  fitButton.textContent = label;
  fitButton.disabled = !state.currentCurve;
  fitButton.title = state.currentCurve ? "拟合当前 m/z 曲线" : "请先生成或上传 PIE 曲线";

  const manualButton = $("manual-fit-button");
  if (manualButton) {
    manualButton.textContent = label;
    manualButton.disabled = state.fitMode !== "manual" || !state.currentCurve || selectedCount === 0;
    manualButton.title = !state.currentCurve
      ? "请先生成或上传 PIE 曲线"
      : selectedCount === 0
        ? "请先选择至少一个 PICS 候选"
        : "使用当前候选和系数设置拟合";
  }
  const manualCaption = $("manual-fit-caption");
  if (manualCaption) {
    manualCaption.textContent = selectedCount
      ? `已选择 ${selectedCount} 条候选${lockedCount ? `，锁定 ${lockedCount} 条系数` : ""}${state.manualPreviewFit ? "，图中为实时预览" : ""}`
      : "尚未选择候选";
  }
}

function setBusy(busy) {
  $("start-button").disabled = busy;
  $("upload-curve-button").disabled = busy;
  if (busy) {
    $("fit-button").disabled = true;
    $("manual-fit-button").disabled = true;
  } else {
    updateFitActionState();
  }
}

function estimateProgress(data) {
  const status = data.status;
  if (status === "done") return 100;
  if (status === "error") return Math.max(state.progressValue, 100);
  if (status === "queued") return Math.max(state.progressValue, 12);
  const message = String(data.message || "");
  const elapsed = Number(data.elapsed) || 0;
  let floor = 24;
  if (message.includes("读取")) floor = 34;
  if (message.includes("生成")) floor = 68;
  if (message.includes("拟合")) floor = 72;
  return Math.max(state.progressValue, Math.min(92, floor + Math.floor(elapsed * 3)));
}

function resetPieState(caption = "任务提交中") {
  state.jobId = null;
  state.summary = null;
  state.result = null;
  state.currentCurve = null;
  state.currentFit = null;
  state.allRows = [];
  state.candidateRows = [];
  state.selectedCandidateIds = new Set();
  state.candidateCoefficients = {};
  state.candidateCurves = {};
  state.manualPreviewFit = null;
  state.lockedCandidateIds = new Set();
  state.progressValue = 0;
  state.logEntries = [];
  $("mz-select").innerHTML = "";
  $("mz-select").disabled = true;
  $("fit-button").disabled = true;
  $("manual-fit-button").disabled = true;
  $("show-components").disabled = true;
  $("source-caption").textContent = caption;
  renderSummary({});
  clearChart();
  renderFitSummary(null);
  renderCandidatePanel();
  renderTable();
  updateFitActionState();
  setProgress(0, caption, "idle");
  $("progress-log").textContent = "";
}

async function startJob(event) {
  event.preventDefault();
  resetPieState("任务提交中");

  try {
    setBusy(true);
    setStatus("running", "提交任务");
    setProgress(8, "提交任务", "running");
    const data = await fetchJson("/api/pie/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    state.jobId = data.job_id;
    log(`任务已提交: ${state.jobId}`);
    setProgress(14, "等待服务器开始计算", "running");
    pollProgress();
  } catch (error) {
    handleMissingPicsLibrary(error);
    setBusy(false);
    setStatus("error", error.message);
    setProgress(100, "提交失败", "error");
    log(error.message);
  }
}

async function uploadPieCurve(event) {
  event.preventDefault();
  const file = $("pie-curve-file").files[0];
  if (!file) {
    setStatus("error", "请选择 PIE 曲线文件");
    log("未选择文件");
    return;
  }
  resetPieState("正在上传曲线");
  try {
    setBusy(true);
    setStatus("running", "上传 PIE 曲线");
    setProgress(18, "读取上传文件", "running");
    const params = new URLSearchParams({ filename: file.name });
    if (state.picsLibraryId) params.set("library_id", state.picsLibraryId);
    const data = await fetchJson(`/api/pie/upload_curve?${params.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: await file.arrayBuffer(),
    });
    state.jobId = data.job_id;
    state.summary = data.summary;
    log(`已载入 ${data.summary.curve_count || 0} 条 PIE 曲线`);
    setProgress(76, "载入曲线结果", "running");
    await loadResult();
    setProgress(100, "PIE 曲线已载入", "done");
    setStatus("done", "PIE 曲线已载入");
  } catch (error) {
    handleMissingPicsLibrary(error);
    setStatus("error", error.message);
    setProgress(100, "上传失败", "error");
    log(error.message);
  } finally {
    setBusy(false);
  }
}

async function pollProgress() {
  if (!state.jobId) return;
  try {
    const data = await fetchJson(`/api/pie/progress/${encodeURIComponent(state.jobId)}`);
    setStatus(data.status, data.message);
    setProgress(
      estimateProgress(data),
      data.status === "done" ? "计算完成，正在载入结果" : data.message,
      data.status,
      `${data.elapsed}s`,
    );
    log(data.message);
    if (data.status === "done") {
      state.summary = data.summary;
      await loadResult();
      setProgress(100, "结果已载入", "done", `${data.elapsed}s`);
      log("结果已载入");
      setBusy(false);
      return;
    }
    if (data.status === "error") {
      handleMissingPicsLibrary({ status: 404, message: data.error, detail: data.error });
      setBusy(false);
      setProgress(100, data.error || "任务失败", "error", `${data.elapsed}s`);
      log(data.error || "任务失败");
      return;
    }
    state._pollRetries = 0;
    setTimeout(pollProgress, 900);
  } catch (error) {
    handleMissingPicsLibrary(error);
    state._pollRetries = (state._pollRetries || 0) + 1;
    if (state._pollRetries <= 5) {
      log(`读取进度失败，${state._pollRetries}/5 次重试...`);
      setTimeout(pollProgress, 2000);
    } else {
      setBusy(false);
      setStatus("error", error.message);
      setProgress(100, "读取进度失败", "error");
      log(error.message);
    }
  }
}

async function loadResult() {
  const result = await fetchJson(`/api/pie/result/${encodeURIComponent(state.jobId)}`);
  state.result = result;
  renderSummary(result.summary || {});
  $("source-caption").textContent = `${result.source_folder || ""} | 卡峰来源: ${result.peak_source || ""}`;
  const mzSelect = $("mz-select");
  mzSelect.innerHTML = "";
  for (const mz of result.summary.mz_values || []) {
    const option = document.createElement("option");
    option.value = mz;
    option.textContent = `m/z ${mz}`;
    mzSelect.appendChild(option);
  }
  mzSelect.disabled = !mzSelect.options.length;
  if (mzSelect.options.length) {
    await loadCurve(mzSelect.value);
  }
}

function renderSummary(summary) {
  const grid = $("summary-grid");
  if (!grid) return;
  const values = [
    ["提取曲线数", summary.curve_count || 0],
    ["能量点数", summary.energy_count || 0],
    ["积分记录数", summary.row_count || 0],
    ["原始文件数", summary.file_count || 0],
  ];
  grid.innerHTML = values
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

async function loadCurve(mz) {
  if (!state.jobId || !mz) return;
  const data = await fetchJson(`/api/pie/curve/${encodeURIComponent(state.jobId)}/${encodeURIComponent(mz)}`);
  state.currentMz = Number(mz);
  state.currentCurve = data.curve;
  state.currentFit = data.fit || null;
  state.candidateRows = [];
  state.selectedCandidateIds = new Set();
  state.candidateCoefficients = {};
  state.candidateCurves = {};
  state.manualPreviewFit = null;
  state.lockedCandidateIds = new Set();
  storeCoefficientsFromFit(state.currentFit);
  $("show-components").disabled = !state.currentFit;
  updateFitActionState();
  renderFitSummary(state.currentFit);
  if (state.fitMode === "manual") {
    await loadCandidates();
  } else {
    renderCandidatePanel();
  }
  drawChart();
  renderTable();
}

function selectedCandidateIds() {
  return Array.from(state.selectedCandidateIds).map(Number).filter(Number.isFinite);
}

function selectedCoefficientPayload() {
  const payload = {};
  for (const id of selectedCandidateIds()) {
    const value = Number(state.candidateCoefficients[id] ?? 0);
    payload[id] = Number.isFinite(value) ? Math.max(0, value) : 0;
  }
  return payload;
}

function storeCoefficientsFromFit(fit) {
  if (!fit || !Array.isArray(fit.species)) return false;
  const next = { ...state.candidateCoefficients };
  let changed = false;
  fit.species.forEach((item) => {
    if (item.coefficients_by_id && typeof item.coefficients_by_id === "object") {
      Object.entries(item.coefficients_by_id).forEach(([idText, coefficientValue]) => {
        const id = Number(idText);
        const coefficient = Number(coefficientValue);
        if (Number.isFinite(id) && Number.isFinite(coefficient)) {
          next[id] = coefficient;
          changed = true;
        }
      });
      return;
    }
    const ids = Array.isArray(item.ids) ? item.ids.map(Number).filter(Number.isFinite) : [];
    if (!ids.length) return;
    const coefficient = Number(item.coefficient);
    if (!Number.isFinite(coefficient)) return;
    const share = coefficient / ids.length;
    ids.forEach((id) => {
      next[id] = share;
      changed = true;
    });
  });
  state.candidateCoefficients = next;
  return changed;
}

async function loadCandidateCurves() {
  if (!state.jobId || !state.currentMz || !state.candidateRows.length) {
    state.candidateCurves = {};
    return;
  }
  const ids = state.candidateRows.map((row) => Number(row.id)).filter(Number.isFinite);
  const data = await fetchJson(
    `/api/pie/candidate_curves/${encodeURIComponent(state.jobId)}/${encodeURIComponent(state.currentMz)}?species_ids=${ids.join(",")}`,
  );
  state.candidateCurves = {};
  (data.rows || []).forEach((row) => {
    state.candidateCurves[Number(row.id)] = row;
  });
}

function buildManualPreviewFit() {
  if (state.fitMode !== "manual" || !state.currentCurve) return null;
  const selected = selectedCandidateIds().filter((id) => state.candidateCurves[id]);
  if (!selected.length) return null;

  const energies = state.currentCurve.energies.map(Number).filter(Number.isFinite);
  const experimental = state.currentCurve.intensities.slice(0, energies.length).map(Number);
  if (!energies.length || experimental.some((value) => !Number.isFinite(value))) return null;

  const design = selected.map((id) => {
    const curve = state.candidateCurves[id];
    return interpolateSeries(curve.energies || [], curve.cross_sections || [], energies);
  });
  const mode = $("coefficient-mode").value;
  let coefficients = [];
  if (mode === "manual") {
    coefficients = selected.map((id) => Math.max(0, Number(state.candidateCoefficients[id] ?? 0) || 0));
  } else if (mode === "locked_fit") {
    const locked = {};
    selected.forEach((id, index) => {
      if (state.lockedCandidateIds.has(id)) {
        locked[index] = Math.max(0, Number(state.candidateCoefficients[id] ?? 0) || 0);
      }
    });
    coefficients = solveNonnegativeLeastSquares(design, experimental, locked);
  } else {
    coefficients = solveNonnegativeLeastSquares(design, experimental);
  }

  const fitted = energies.map((_, row) => design.reduce((sum, basis, col) => sum + basis[row] * coefficients[col], 0));
  const residuals = experimental.map((value, index) => value - fitted[index]);
  const mean = experimental.reduce((sum, value) => sum + value, 0) / experimental.length;
  const ssTot = experimental.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const ssRes = residuals.reduce((sum, value) => sum + value ** 2, 0);
  const fittedArea = fitted.reduce((sum, value) => sum + value, 0);
  const species = selected.map((id, index) => {
    const curve = state.candidateCurves[id];
    const component = design[index].map((value) => value * coefficients[index]);
    const componentArea = component.reduce((sum, value) => sum + value, 0);
    return {
      ids: [id],
      coefficients_by_id: { [id]: coefficients[index] },
      mz: Number(curve.mz),
      species: curve.species,
      ie: curve.ie,
      coefficient: coefficients[index],
      contribution_percent: fittedArea > 0 ? (100 * componentArea) / fittedArea : 0,
      r_squared: ssTot > 0 ? 1 - ssRes / ssTot : 0,
      component_intensities: component,
    };
  }).filter((item) => item.coefficient > 0.001);

  return {
    preview: true,
    energies,
    experimental,
    fitted,
    residuals,
    r_squared: ssTot > 0 ? 1 - ssRes / ssTot : 0,
    rmse: Math.sqrt(residuals.reduce((sum, value) => sum + value ** 2, 0) / residuals.length),
    mae: residuals.reduce((sum, value) => sum + Math.abs(value), 0) / residuals.length,
    candidate_count: selected.length,
    coefficient_mode: mode,
    locked_species_ids: Array.from(state.lockedCandidateIds).filter((id) => selected.includes(id)),
    species,
  };
}

function scheduleManualPreview() {
  if (state.previewFrame) cancelAnimationFrame(state.previewFrame);
  state.previewFrame = requestAnimationFrame(() => {
    state.previewFrame = null;
    state.manualPreviewFit = buildManualPreviewFit();
    renderFitSummary(state.manualPreviewFit || state.currentFit);
    drawChart();
    updateFitActionState();
  });
}

async function fitCurrentCurve() {
  if (!state.jobId || !state.currentMz) return;
  try {
    $("fit-button").disabled = true;
    $("manual-fit-button").disabled = true;
    setStatus("running", `拟合 m/z ${state.currentMz}`);
    setProgress(45, `拟合 m/z ${state.currentMz}`, "running");
    const options = { method: "POST" };
    if (state.fitMode === "manual") {
      const selected = selectedCandidateIds();
      if (!selected.length) {
        throw new Error("请至少选择一个 PICS 候选");
      }
      const coefficientMode = $("coefficient-mode").value;
      const coefficients = selectedCoefficientPayload();
      if (
        coefficientMode === "manual"
        && !Object.values(coefficients).some((value) => Number(value) > 0)
      ) {
        throw new Error("使用手动系数时，请至少填写一个大于 0 的系数");
      }
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify({
        species_ids: selected,
        coefficient_mode: coefficientMode,
        coefficients,
        locked_species_ids: Array.from(state.lockedCandidateIds).filter((id) => state.selectedCandidateIds.has(id)),
      });
    }
    const data = await fetchJson(
      `/api/pie/fit/${encodeURIComponent(state.jobId)}/${encodeURIComponent(state.currentMz)}`,
      options,
    );
    state.currentFit = data.fit;
    state.manualPreviewFit = null;
    if (state.fitMode === "manual") {
      storeCoefficientsFromFit(state.currentFit);
      renderCandidatePanel();
    }
    $("show-components").disabled = !(state.currentFit && state.currentFit.species && state.currentFit.species.length);
    renderFitSummary(state.currentFit);
    drawChart();
    renderTable();
    setStatus("done", `m/z ${state.currentMz} 拟合完成`);
    setProgress(100, `m/z ${state.currentMz} 拟合完成`, "done");
    log(`m/z ${state.currentMz} 拟合完成`);
  } catch (error) {
    setStatus("error", error.message);
    setProgress(100, "拟合失败", "error");
    log(error.message);
  } finally {
    updateFitActionState();
  }
}

function renderFitSummary(fit) {
  const values = [
    ["当前 m/z", state.currentMz || "-"],
    ["候选 PICS", fit ? fit.candidate_count || 0 : 0],
    ["R²", fit ? fmt(fit.r_squared, 5) : "未拟合"],
    ["RMSE", fit ? fmt(fit.rmse, 5) : "-"],
  ];
  $("fit-summary").innerHTML = values
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

async function loadCandidates() {
  if (!state.jobId || !state.currentMz) {
    renderCandidatePanel();
    return;
  }
  state.candidateLoading = true;
  $("candidate-caption").textContent = "正在读取";
  renderCandidatePanel();
  try {
    const data = await fetchJson(
      `/api/pie/candidates/${encodeURIComponent(state.jobId)}/${encodeURIComponent(state.currentMz)}`,
    );
    state.candidateRows = data.rows || [];
    state.selectedCandidateIds = new Set(state.candidateRows.map((row) => Number(row.id)));
    await loadCandidateCurves();
    scheduleManualPreview();
  } finally {
    state.candidateLoading = false;
    renderCandidatePanel();
  }
}

function filterRows(rows, filterText) {
  const query = String(filterText || "").trim().toLowerCase();
  if (!query) return rows;
  return rows.filter((row) => Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query)));
}

function setCaptionChip(id, text) {
  const chip = $(id);
  if (chip) chip.textContent = text;
}

function renderCandidatePanel() {
  const panel = $("candidate-panel");
  panel.classList.toggle("hidden", state.fitMode !== "manual");
  if (state.fitMode !== "manual") {
    updateFitActionState();
    return;
  }

  const allRows = state.candidateRows || [];
  const rows = filterRows(allRows, state.candidateFilterText);
  const selectedCount = selectedCandidateIds().length;
  const lockedCount = state.lockedCandidateIds.size;
  $("candidate-caption").textContent = allRows.length
    ? `m/z ${state.currentMz} | ${allRows.length} 条`
    : state.candidateLoading ? "正在读取" : "无候选";
  setCaptionChip(
    "candidate-stats",
    state.candidateLoading
      ? "读取候选中"
      : `${rows.length}/${allRows.length} 条显示 · 已选 ${selectedCount} · 锁定 ${lockedCount}`,
  );
  const view = $("candidate-view");
  if (state.candidateLoading) {
    view.innerHTML = '<div class="empty loading-state">正在读取当前 m/z 的 PICS 候选...</div>';
    updateFitActionState();
    return;
  }
  if (!allRows.length) {
    view.innerHTML = '<div class="empty">当前 m/z 没有 PICS 候选。</div>';
    updateFitActionState();
    return;
  }
  if (!rows.length) {
    view.innerHTML = '<div class="empty">没有匹配当前筛选条件的候选。</div>';
    updateFitActionState();
    return;
  }
  const mode = $("coefficient-mode").value;
  const coefficientClass = mode === "manual" ? " emphasized-cell" : "";
  const lockClass = mode === "locked_fit" ? " emphasized-cell" : "";
  const tbody = rows.map((row) => {
    const id = Number(row.id);
    const checked = state.selectedCandidateIds.has(id) ? " checked" : "";
    const locked = state.lockedCandidateIds.has(id) ? " checked" : "";
    const coefficient = state.candidateCoefficients[id] ?? "";
    return `<tr>
      <td><input type="checkbox" aria-label="选择候选 ${escapeHtml(row.name)}" data-candidate-check="${row.id}"${checked}></td>
      <td class="${coefficientClass.trim()}"><input class="coefficient-input" type="number" min="0" step="0.000001" value="${escapeHtml(coefficient)}" aria-label="${escapeHtml(row.name)} 的拟合系数" data-candidate-coefficient="${row.id}"></td>
      <td class="lock-cell${lockClass}"><input type="checkbox" aria-label="锁定 ${escapeHtml(row.name)} 的拟合系数" data-candidate-lock="${row.id}"${locked}></td>
      <td>${row.id}</td>
      <td>${row.mz}</td>
      <td>${escapeHtml(row.name)}</td>
      <td>${row.ionization_energy == null ? "" : fmt(row.ionization_energy, 5)}</td>
      <td>${row.energy_min == null ? "" : `${fmt(row.energy_min, 4)}-${fmt(row.energy_max, 4)}`}</td>
      <td>${row.point_count}</td>
      <td>${row.cross_section_max == null ? "" : fmt(row.cross_section_max, 6)}</td>
    </tr>`;
  }).join("");
  view.innerHTML = `
    <table aria-label="PICS 候选表">
      <thead><tr><th scope="col">启用</th><th scope="col">系数</th><th scope="col">锁定</th><th scope="col">id</th><th scope="col">m/z</th><th scope="col">物种</th><th scope="col">IE(eV)</th><th scope="col">能量范围</th><th scope="col">点数</th><th scope="col">最大截面</th></tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
  updateFitActionState();
}

async function setFitMode(mode) {
  state.fitMode = mode;
  if (mode !== "manual") state.manualPreviewFit = null;
  document.querySelectorAll(".segment-button").forEach((button) => {
    const active = button.dataset.fitMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  updateFitActionState();
  if (mode === "manual" && state.currentCurve && !state.candidateRows.length) {
    try {
      await loadCandidates();
    } catch (error) {
      setStatus("error", error.message);
      log(error.message);
      renderCandidatePanel();
    }
  } else {
    renderCandidatePanel();
    scheduleManualPreview();
  }
}

function clearChart() {
  const canvas = $("pie-chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawChart() {
  const canvas = $("pie-chart");
  const { ctx, width, height } = setupChartCanvas(canvas, 320, 420);
  canvas.setAttribute(
    "aria-label",
    state.currentCurve ? `PIE 曲线图，当前 m/z ${state.currentMz}` : "PIE 曲线图，暂无数据",
  );

  if (!state.currentCurve) {
    drawEmptyChart(ctx, width, height, "生成曲线后显示 PIE 图");
    return;
  }

  const theme = chartTheme();
  const series = [
    {
      label: "实验 PIE",
      x: state.currentCurve.energies,
      y: state.currentCurve.intensities,
      color: theme.primary,
      points: true,
    },
  ];
  const chartFit = state.manualPreviewFit || state.currentFit;
  if (chartFit && chartFit.fitted && chartFit.fitted.length) {
    series.push({
      label: chartFit.preview ? "实时预览" : "PICS 总拟合",
      x: chartFit.energies,
      y: chartFit.fitted,
      color: theme.fit,
      points: false,
    });
  }
  if ($("show-components").checked && chartFit && chartFit.species) {
    chartFit.species.slice(0, 6).forEach((item, index) => {
      series.push({
        label: item.species,
        x: chartFit.energies,
        y: item.component_intensities || [],
        color: interpolatePieSeriesColor(index, theme),
        points: false,
        dash: true,
      });
    });
  }

  const allX = validNumbers(series.flatMap((item) => item.x));
  const allY = validNumbers(series.flatMap((item) => item.y));
  if (!allX.length || !allY.length) {
    drawEmptyChart(ctx, width, height, "当前曲线没有有效数据");
    return;
  }

  const bounds = chartBounds(allX, allY, 0, 0.1, 1);
  const yTicks = yTicksFor(bounds);
  const margin = chartMargin(ctx, width, 34, yTicks);
  const legendAvailableWidth = Math.max(120, width - margin.left - margin.right);
  margin.top += legendRows(ctx, series, legendAvailableWidth) * 20;
  const scales = chartScales(width, height, margin, bounds);
  drawAxes(ctx, width, height, margin, scales, bounds, yTicks, "Photon energy (eV)", "Normalized intensity", theme);
  drawLineSeries(ctx, series, scales);
  drawLegend(ctx, series, width, margin, legendAvailableWidth, theme);
}

function drawEmptyChart(ctx, width, height, text) {
  const theme = chartTheme();
  ctx.fillStyle = theme.text;
  ctx.font = "14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
  ctx.textAlign = "start";
}

function rowsForTable() {
  if (state.tableMode === "species") {
    return state.currentFit && state.currentFit.species
      ? state.currentFit.species.map((item) => ({
          "候选ID": Array.isArray(item.ids) ? item.ids.join(";") : "",
          "物种": item.species,
          "电离能(eV)": item.ie == null ? "" : fmt(item.ie, 5),
          "系数": fmt(item.coefficient, 6),
          "贡献(%)": fmt(item.contribution_percent, 5),
          "R²": fmt(item.r_squared, 5),
        }))
      : [];
  }
  if (state.tableMode === "curve") {
    return state.currentCurve ? state.currentCurve.rows || [] : [];
  }
  return state.allRows || [];
}

function renderTable() {
  const rows = rowsForTable();
  const view = $("table-view");
  const downloadButton = $("download-button");
  if (!rows.length) {
    view.innerHTML = `<div class="empty">${state.tableMode === "species" ? "当前曲线尚未拟合或无匹配物种。" : "没有可显示的数据。"}</div>`;
    downloadButton.disabled = true;
    downloadButton.title = "当前表暂无数据";
    return;
  }
  const columns = Object.keys(rows[0]);
  const thead = `<thead><tr>${columns.map((col) => `<th scope="col">${escapeHtml(col)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((row) => `<tr>${columns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  view.innerHTML = `<table>${thead}${tbody}</table>`;
  downloadButton.disabled = false;
  downloadButton.title = "下载当前表格";
}

async function setTableMode(mode) {
  state.tableMode = mode;
  document.querySelectorAll(".tab-button").forEach((button) => {
    const active = button.dataset.table === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  if (mode === "all" && state.jobId && !state.allRows.length) {
    const data = await fetchJson(`/api/pie/export/${encodeURIComponent(state.jobId)}`);
    state.allRows = data.rows || [];
  }
  renderTable();
}

function downloadCurrentTable() {
  const rows = rowsForTable();
  if (!rows.length) return;
  const columns = Object.keys(rows[0]);
  const lines = [
    columns.join(","),
    ...rows.map((row) => columns.map((col) => csvCell(row[col])).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `pie_${state.tableMode}_${state.currentMz || "all"}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

const WORKFLOW_PAGES = new Set(["project-page", "pie-page", "pics-upload-page"]);

function switchPage(pageId) {
  const inWorkflow = WORKFLOW_PAGES.has(pageId);
  document.querySelectorAll(".page").forEach((page) => {
    const active = page.id === pageId;
    page.classList.toggle("active", active);
    page.hidden = !active;
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    const active = button.id === "workflow-tab" ? inWorkflow : button.dataset.page === pageId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll(".workflow-tab-button").forEach((button) => {
    const active = button.dataset.page === pageId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  const workflowTabs = $("workflow-tabs");
  workflowTabs.hidden = !inWorkflow;
  workflowTabs.classList.toggle("hidden", !inWorkflow);
  if (pageId === "pie-page") {
    drawChart();
  } else if (pageId === "pics-page") {
    drawPicsChart();
  }
}

function picsSearchParams() {
  const params = new URLSearchParams();
  const mz = $("pics-mz").value.trim();
  const tolerance = $("pics-tolerance").value.trim();
  const name = $("pics-name").value.trim();
  const ieMin = $("pics-ie-min").value.trim();
  const ieMax = $("pics-ie-max").value.trim();
  params.set("limit", $("pics-limit").value.trim() || "100");
  if (state.picsLibraryId) params.set("library_id", state.picsLibraryId);
  if (mz) params.set("mz", mz);
  if (tolerance) params.set("tolerance", tolerance);
  if (name) params.set("name", name);
  if (ieMin) params.set("ie_min", ieMin);
  if (ieMax) params.set("ie_max", ieMax);
  return params;
}

async function searchPics(event) {
  event.preventDefault();
  $("pics-search-button").disabled = true;
  $("pics-caption").textContent = "正在查询";
  state.picsRows = [];
  state.currentPics = null;
  renderPicsSummary();
  drawPicsChart();
  renderPicsTable();
  try {
    const data = await fetchJson(`/api/pics/search?${picsSearchParams().toString()}`);
    state.picsRows = data.rows || [];
    $("pics-caption").textContent = `命中 ${state.picsRows.length} 条记录`;
    renderPicsSummary();
    renderPicsTable();
    if (state.picsRows.length) {
      await loadPicsSpecies(state.picsRows[0].id);
    }
  } catch (error) {
    if (handleMissingPicsLibrary(error)) {
      state.picsRows = [];
      state.currentPics = null;
      renderPicsSummary();
      drawPicsChart();
      renderPicsTable();
    }
    $("pics-caption").textContent = error.message;
  } finally {
    $("pics-search-button").disabled = false;
  }
}

function renderPicsSummary() {
  const species = state.currentPics ? state.currentPics.species : null;
  const points = state.currentPics ? state.currentPics.points || [] : [];
  const values = [
    ["命中物种", state.picsRows.length],
    ["当前 m/z", species ? species.mz : "-"],
    ["数据点", points.length],
    ["IE(eV)", species && species.ionization_energy != null ? fmt(species.ionization_energy, 5) : "-"],
  ];
  $("pics-summary-grid").innerHTML = values
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

async function loadPicsSpecies(speciesId) {
  const params = new URLSearchParams();
  if (state.picsLibraryId) params.set("library_id", state.picsLibraryId);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  let data;
  try {
    data = await fetchJson(`/api/pics/species/${encodeURIComponent(speciesId)}${suffix}`);
  } catch (error) {
    if (handleMissingPicsLibrary(error)) {
      state.currentPics = null;
      renderPicsSummary();
      drawPicsChart();
      renderPicsTable();
    }
    $("pics-caption").textContent = error.message;
    throw error;
  }
  state.currentPics = data;
  renderPicsSummary();
  drawPicsChart();
  renderPicsTable();
}

function renderPicsTable() {
  const view = $("pics-table-view");
  const downloadButton = $("pics-download-button");
  const rows = filterRows(state.picsRows, state.picsFilterText);
  setCaptionChip("pics-result-stats", `${rows.length}/${state.picsRows.length} 条显示`);
  if (!state.picsRows.length) {
    view.innerHTML = '<div class="empty">输入条件后查询 PICS 数据库。</div>';
    downloadButton.disabled = true;
    downloadButton.title = "当前表暂无数据";
    return;
  }
  if (!rows.length) {
    view.innerHTML = '<div class="empty">没有匹配当前筛选条件的 PICS 记录。</div>';
    downloadButton.disabled = false;
    downloadButton.title = "下载全部查询结果";
    return;
  }
  const headers = ["id", "m/z", "物种", "IE(eV)", "能量范围(eV)", "点数", "最大截面", "操作"];
  const tbody = rows.map((row) => {
    const active = state.currentPics && state.currentPics.species.id === row.id ? " class=\"selected-row\"" : "";
    return `<tr${active}>
      <td>${row.id}</td>
      <td>${row.mz}</td>
      <td>${escapeHtml(row.name)}</td>
      <td>${row.ionization_energy == null ? "" : fmt(row.ionization_energy, 5)}</td>
      <td>${row.energy_min == null ? "" : `${fmt(row.energy_min, 4)}-${fmt(row.energy_max, 4)}`}</td>
      <td>${row.point_count}</td>
      <td>${row.cross_section_max == null ? "" : fmt(row.cross_section_max, 6)}</td>
      <td><button class="small-button" type="button" aria-label="查看 ${escapeHtml(row.name)} 的 PICS 曲线" data-pics-id="${row.id}">查看</button></td>
    </tr>`;
  }).join("");
  view.innerHTML = `<table aria-label="PICS 查询结果表"><thead><tr>${headers.map((item) => `<th scope="col">${item}</th>`).join("")}</tr></thead><tbody>${tbody}</tbody></table>`;
  downloadButton.disabled = false;
  downloadButton.title = "下载全部查询结果";
}

function drawPicsChart() {
  const canvas = $("pics-chart");
  if (!canvas) return;
  const { ctx, width, height } = setupChartCanvas(canvas, 300, 380);
  canvas.setAttribute(
    "aria-label",
    state.currentPics && state.currentPics.species
      ? `PICS 曲线图，当前物种 ${state.currentPics.species.name}`
      : "PICS 曲线图，暂无数据",
  );
  if (!state.currentPics || !state.currentPics.points || !state.currentPics.points.length) {
    drawEmptyChart(ctx, width, height, "选择物种后显示 PICS 曲线");
    return;
  }
  const xValues = validNumbers(state.currentPics.points.map((point) => point.energy_ev));
  const yValues = validNumbers(state.currentPics.points.map((point) => point.cross_section));
  if (!xValues.length || !yValues.length) {
    drawEmptyChart(ctx, width, height, "当前物种没有有效 PICS 数据");
    return;
  }

  const theme = chartTheme();
  const bounds = chartBounds(xValues, yValues, 0, 0.1, 0.1);
  const yTicks = yTicksFor(bounds);
  const margin = chartMargin(ctx, width, 48, yTicks);
  const scales = chartScales(width, height, margin, bounds);
  drawAxes(ctx, width, height, margin, scales, bounds, yTicks, "Photon energy (eV)", "Cross section", theme);
  drawLineSeries(ctx, [{
    x: state.currentPics.points.map((point) => point.energy_ev),
    y: state.currentPics.points.map((point) => point.cross_section),
    color: theme.primary,
    points: true,
    pointRadius: 2.8,
  }], scales);
  const species = state.currentPics.species;
  ctx.fillStyle = theme.title;
  ctx.fillText(fitCanvasText(ctx, `${species.name} | m/z ${species.mz}`, width - margin.left - margin.right), margin.left, 24);
}

function downloadPicsRows() {
  if (!state.picsRows.length) return;
  const rows = state.picsRows.map((row) => ({
    id: row.id,
    mz: row.mz,
    name: row.name,
    ionization_energy: row.ionization_energy,
    point_count: row.point_count,
    energy_min: row.energy_min,
    energy_max: row.energy_max,
    cross_section_max: row.cross_section_max,
  }));
  const columns = Object.keys(rows[0]);
  const lines = [
    columns.join(","),
    ...rows.map((row) => columns.map((col) => csvCell(row[col])).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "pics_query.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

function setUploadStatus(status, text) {
  setStatusView(uploadStatusView, status, text);
}

function setUploadProgress(value, step, status = "running") {
  setProgressView(uploadStatusView, value, step, status);
}

function uploadLog(message) {
  appendLog({ stateKey: uploadStatusView.logStateKey, viewId: uploadStatusView.logViewId }, message);
}

function setPicsLibrary(libraryId) {
  state.picsLibraryId = libraryId || "";
  writeStorage("bl03u_pics_library_id", state.picsLibraryId);
  updatePicsLibraryCaptions();
  updateProjectSettingSummaries();
}

function updatePicsLibraryCaptions() {
  const text = state.picsLibraryId
    ? `当前使用临时工作库 ${state.picsLibraryId.slice(0, 8)}`
    : "当前使用服务器维护库";
  const queryCaption = $("pics-library-caption");
  if (queryCaption) queryCaption.textContent = text;
  const activeLine = $("active-pics-library");
  if (activeLine) activeLine.textContent = `${text}。`;
  const useServerButton = $("use-server-library");
  if (useServerButton) {
    useServerButton.disabled = !state.picsLibraryId;
    useServerButton.title = state.picsLibraryId
      ? "切回服务器维护的 PICS 数据库"
      : "当前已经使用服务器维护库";
  }
}

function updateUploadScopeUi() {
  const isServer = $("pics-upload-scope").value === "server";
  $("admin-token-wrap").classList.toggle("hidden", !isServer);
  $("pics-write-mode-wrap").classList.toggle("hidden", !isServer);
  $("overwrite-confirm-wrap").classList.toggle(
    "hidden",
    !isServer || $("pics-upload-mode").value !== "overwrite_all",
  );
  $("pics-upload-button").textContent = isServer ? "写入服务器维护库" : "上传为临时工作库";
}

async function uploadPics(event) {
  event.preventDefault();
  const file = $("pics-upload-file").files[0];
  if (!file) {
    setUploadStatus("error", "请选择文件");
    setUploadProgress(100, "未选择文件", "error");
    uploadLog("未选择 PICS 数据文件");
    return;
  }
  const scope = $("pics-upload-scope").value;
  const mode = $("pics-upload-mode").value;
  const confirmOverwrite = $("overwrite-confirm").checked;
  const adminToken = $("admin-token").value.trim();
  if (scope === "server" && !adminToken) {
    setUploadStatus("error", "需要管理员 token");
    setUploadProgress(100, "缺少管理员 token", "error");
    uploadLog("写入服务器维护库前需要填写管理员 token");
    return;
  }
  if (scope === "server" && mode === "overwrite_all" && !confirmOverwrite) {
    setUploadStatus("error", "需要确认覆盖");
    setUploadProgress(100, "等待覆盖确认", "error");
    uploadLog("覆盖整个库前需要勾选确认");
    return;
  }

  $("pics-upload-button").disabled = true;
  setUploadStatus("running", "上传中");
  setUploadProgress(28, "读取上传文件", "running");
  uploadLog(file.name);
  try {
    const params = new URLSearchParams({
      filename: file.name,
      scope,
      mode,
      confirm_overwrite: confirmOverwrite ? "true" : "false",
    });
    const headers = { "Content-Type": "application/octet-stream" };
    if (scope === "server") headers["X-Admin-Token"] = adminToken;
    const result = await fetchJson(`/api/pics/upload?${params.toString()}`, {
      method: "POST",
      headers,
      body: await file.arrayBuffer(),
    });
    setUploadProgress(82, "写入 PICS 库", "running");
    if (result.scope === "session") {
      setPicsLibrary(result.library_id);
      setUploadStatus("done", "临时工作库已启用");
      setUploadProgress(100, "临时工作库已启用", "done");
      uploadLog(`临时导入 ${result.inserted_species} 个物种，${result.inserted_points} 个点`);
    } else {
      setPicsLibrary("");
      setUploadStatus("done", "服务器维护库已更新");
      setUploadProgress(100, "服务器维护库已更新", "done");
      uploadLog(`写入 ${result.inserted_species} 个物种，${result.inserted_points} 个点`);
    }
    renderUploadResult(result);
  } catch (error) {
    setUploadStatus("error", "上传失败");
    setUploadProgress(100, "上传失败", "error");
    uploadLog(error.message);
  } finally {
    $("pics-upload-button").disabled = false;
  }
}

function renderUploadResult(result) {
  const rows = [
    ["使用范围", result.scope === "server" ? "服务器维护库" : "临时工作库"],
    ["写入模式", result.scope === "server" ? result.mode : "临时覆盖"],
    ["解析物种", result.parsed_species],
    ["写入物种", result.inserted_species],
    ["替换旧记录", result.replaced_species],
    ["写入数据点", result.inserted_points],
    ["备份", result.backup ? "已创建" : "无"],
  ];
  if (result.scope === "session") {
    rows.push(["当前状态", "临时工作库已启用，查询和拟合将优先使用该库"]);
  }
  $("upload-result").innerHTML = `
    <table aria-label="PICS 上传结果统计">
      <tbody>
        ${rows.map(([key, value]) => `<tr><th scope="row">${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("")}
      </tbody>
    </table>
  `;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function activePageId() {
  const page = document.querySelector(".page.active");
  return page ? page.id : "pics-page";
}

function requestActiveSubmit() {
  const pageId = activePageId();
  if (pageId === "pie-page") {
    const form = $("curve-upload-form").classList.contains("hidden") ? $("query-form") : $("curve-upload-form");
    form.requestSubmit();
  } else if (pageId === "pics-page") {
    $("pics-form").requestSubmit();
  } else if (pageId === "pics-upload-page") {
    $("pics-upload-form").requestSubmit();
  }
}

function downloadActiveTable() {
  const pageId = activePageId();
  if (pageId === "pie-page" && !$("download-button").disabled) {
    downloadCurrentTable();
    return true;
  }
  if (pageId === "pics-page" && !$("pics-download-button").disabled) {
    downloadPicsRows();
    return true;
  }
  return false;
}

document.querySelectorAll("[data-page]").forEach((button) => {
  button.addEventListener("click", () => switchPage(button.dataset.page));
});
document.querySelectorAll(".source-button").forEach((button) => {
  button.addEventListener("click", () => {
    const source = button.dataset.pieSource;
    document.querySelectorAll(".source-button").forEach((item) => {
      const active = item.dataset.pieSource === source;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
      item.tabIndex = active ? 0 : -1;
    });
    $("query-form").classList.toggle("hidden", source !== "folder");
    $("query-form").classList.toggle("active", source === "folder");
    $("curve-upload-form").classList.toggle("hidden", source !== "upload");
    $("curve-upload-form").classList.toggle("active", source === "upload");
  });
});
document.querySelectorAll(".settings-tab-button").forEach((button) => {
  button.addEventListener("click", () => switchProjectSettingsTab(button.dataset.settingsTab));
});
$("save-project-settings").addEventListener("click", () => commitProjectSettings(true));
$("apply-project-settings").addEventListener("click", () => commitProjectSettings(false));
$("reset-project-settings").addEventListener("click", () => {
  state.projectSettings = defaultWebProjectSettings();
  saveWebProjectSettings();
  applyProjectSettingsToForms();
  settingsStatus("已恢复默认设置。", "success");
});
$("edit-pie-settings").addEventListener("click", () => editProjectSettings("project-settings-data"));
$("edit-pics-settings").addEventListener("click", () => editProjectSettings("project-settings-function"));

$("query-form").addEventListener("submit", startJob);
$("curve-upload-form").addEventListener("submit", uploadPieCurve);
$("peak-source").addEventListener("change", updateProjectSettingSummaries);
$("mz-select").addEventListener("change", (event) => loadCurve(event.target.value));
$("fit-button").addEventListener("click", fitCurrentCurve);
$("manual-fit-button").addEventListener("click", fitCurrentCurve);
document.querySelectorAll(".segment-button").forEach((button) => {
  button.addEventListener("click", () => setFitMode(button.dataset.fitMode));
});
$("coefficient-mode").addEventListener("change", () => {
  scheduleManualPreview();
  updateFitActionState();
});
$("candidate-view").addEventListener("change", (event) => {
  const input = event.target.closest("[data-candidate-check]");
  if (input) {
    const id = Number(input.dataset.candidateCheck);
    if (input.checked) state.selectedCandidateIds.add(id);
    else {
      state.selectedCandidateIds.delete(id);
      state.lockedCandidateIds.delete(id);
    }
    scheduleManualPreview();
    updateFitActionState();
    return;
  }
  const lockInput = event.target.closest("[data-candidate-lock]");
  if (lockInput) {
    const id = Number(lockInput.dataset.candidateLock);
    if (lockInput.checked) {
      state.lockedCandidateIds.add(id);
      state.selectedCandidateIds.add(id);
    } else {
      state.lockedCandidateIds.delete(id);
    }
    renderCandidatePanel();
    scheduleManualPreview();
    updateFitActionState();
  }
});
$("candidate-view").addEventListener("input", (event) => {
  const input = event.target.closest("[data-candidate-coefficient]");
  if (!input) return;
  const id = Number(input.dataset.candidateCoefficient);
  const value = Number(input.value);
  state.candidateCoefficients[id] = Number.isFinite(value) ? Math.max(0, value) : 0;
  scheduleManualPreview();
});
$("candidate-filter").addEventListener("input", (event) => {
  state.candidateFilterText = event.target.value;
  renderCandidatePanel();
});
$("select-all-candidates").addEventListener("click", () => {
  state.selectedCandidateIds = new Set(state.candidateRows.map((row) => Number(row.id)));
  renderCandidatePanel();
  scheduleManualPreview();
});
$("clear-candidates").addEventListener("click", () => {
  state.selectedCandidateIds = new Set();
  state.lockedCandidateIds = new Set();
  renderCandidatePanel();
  scheduleManualPreview();
});
$("seed-coefficients").addEventListener("click", () => {
  if (!storeCoefficientsFromFit(state.currentFit)) {
    setStatus("error", "当前没有可读取的拟合系数");
    log("先执行一次自动求系数，或手动填写候选系数");
  }
  renderCandidatePanel();
  scheduleManualPreview();
});
$("zero-coefficients").addEventListener("click", () => {
  state.candidateCoefficients = {};
  renderCandidatePanel();
  scheduleManualPreview();
});
$("show-components").addEventListener("change", drawChart);
$("download-button").addEventListener("click", downloadCurrentTable);
document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setTableMode(button.dataset.table));
});
$("pics-form").addEventListener("submit", searchPics);
$("pics-download-button").addEventListener("click", downloadPicsRows);
$("pics-table-view").addEventListener("click", (event) => {
  const button = event.target.closest("[data-pics-id]");
  if (button) {
    loadPicsSpecies(button.dataset.picsId).catch((error) => {
      if (!isMissingPicsLibraryError(error)) $("pics-caption").textContent = error.message;
    });
  }
});
$("pics-filter").addEventListener("input", (event) => {
  state.picsFilterText = event.target.value;
  renderPicsTable();
});
$("pics-upload-form").addEventListener("submit", uploadPics);
$("pics-upload-scope").addEventListener("change", updateUploadScopeUi);
$("pics-upload-mode").addEventListener("change", () => {
  updateUploadScopeUi();
});
$("use-server-library").addEventListener("click", () => {
  setPicsLibrary("");
  setUploadStatus("idle", "已切回服务器维护库");
  setUploadProgress(0, "等待上传", "idle");
  uploadLog("后续查询和拟合将使用服务器维护库");
});
function scheduleChartResize() {
  if (state.resizeFrame) cancelAnimationFrame(state.resizeFrame);
  state.resizeFrame = requestAnimationFrame(() => {
    state.resizeFrame = null;
    drawChart();
    drawPicsChart();
  });
}

document.addEventListener("keydown", (event) => {
  const commandKey = event.ctrlKey || event.metaKey;
  if (!commandKey) return;
  if (event.key === "Enter") {
    event.preventDefault();
    requestActiveSubmit();
  }
  if (event.key.toLowerCase() === "s" && downloadActiveTable()) {
    event.preventDefault();
  }
});
window.addEventListener("resize", scheduleChartResize);
document.querySelectorAll(".page").forEach((page) => {
  page.hidden = !page.classList.contains("active");
});
bindRovingControls(".nav-button");
bindRovingControls(".workflow-tab-button");
bindRovingControls(".settings-tab-button");
bindRovingControls(".source-button");
bindRovingControls(".segment-button");
bindRovingControls(".tab-button");
setTabIndex(".nav-button", (item) => item.classList.contains("active"));
setTabIndex(".workflow-tab-button", (item) => item.classList.contains("active"));
setTabIndex(".settings-tab-button", (item) => item.classList.contains("active"));
setTabIndex(".source-button", (item) => item.classList.contains("active"));
setTabIndex(".segment-button", (item) => item.classList.contains("active"));
setTabIndex(".tab-button", (item) => item.classList.contains("active"));
applyProjectSettingsToForms();
updatePicsLibraryCaptions();
updateUploadScopeUi();
setProgress(0, "等待任务", "idle");
setUploadProgress(0, "等待上传", "idle");
drawChart();
drawPicsChart();
