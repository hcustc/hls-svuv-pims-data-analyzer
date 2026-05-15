const state = {
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
  picsLibraryId: localStorage.getItem("bl03u_pics_library_id") || "",
  progressValue: 0,
  uploadProgressValue: 0,
  logEntries: [],
  uploadLogEntries: [],
};

const $ = (id) => document.getElementById(id);

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

function setStatus(status, text) {
  const pill = $("status-pill");
  const [className, label] = statusPresentation(status);
  pill.className = `pill ${className}`;
  pill.textContent = label;
  $("status-text").textContent = text;
}

function setProgress(value, step, status = "running", elapsedText = "") {
  const clamped = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  state.progressValue = clamped;
  const bar = $("progress-bar");
  bar.style.width = `${clamped}%`;
  bar.classList.toggle("error", status === "error");
  $("progress-percent").textContent = `${clamped}%`;
  $("progress-step").textContent = step || "等待任务";
  $("progress-elapsed").textContent = elapsedText || "--";
  const meter = bar.closest(".progress-meter");
  if (meter) {
    meter.setAttribute("role", "progressbar");
    meter.setAttribute("aria-valuemin", "0");
    meter.setAttribute("aria-valuemax", "100");
    meter.setAttribute("aria-valuenow", String(clamped));
    meter.setAttribute("aria-valuetext", `${step || "任务进度"} ${clamped}%`);
  }
}

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  const text = String(message || "").trim();
  if (!text) return;
  if (state.logEntries.length && state.logEntries[state.logEntries.length - 1].endsWith(`  ${text}`)) {
    return;
  }
  state.logEntries.push(`${stamp}  ${text}`);
  state.logEntries = state.logEntries.slice(-10);
  const view = $("progress-log");
  view.textContent = state.logEntries.join("\n");
  view.scrollTop = view.scrollHeight;
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
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.error || response.statusText);
  }
  return data;
}

function payloadFromForm() {
  return {
    folder: $("folder").value.trim(),
    library_id: state.picsLibraryId || null,
    peak_source: $("peak-source").value,
    manual_peak_path: $("manual-peak-path").value.trim() || null,
    target_mz: $("target-mz").value.trim() || null,
    recursive: $("recursive").checked,
    energy_decimals: Number($("energy-decimals").value || 1),
    gaussian: $("gaussian").checked,
    photon_mode: $("photon-mode").value,
    light_source: $("light-source").value,
    mass_discrimination: Number($("mass-discrimination").value || 1),
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
  $("fit-button").textContent = label;
  $("fit-button").disabled = !state.currentCurve;

  const manualButton = $("manual-fit-button");
  if (manualButton) {
    manualButton.textContent = label;
    manualButton.disabled = state.fitMode !== "manual" || !state.currentCurve || selectedCount === 0;
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
      setBusy(false);
      setProgress(100, data.error || "任务失败", "error", `${data.elapsed}s`);
      log(data.error || "任务失败");
      return;
    }
    setTimeout(pollProgress, 900);
  } catch (error) {
    setBusy(false);
    setStatus("error", error.message);
    setProgress(100, "读取进度失败", "error");
    log(error.message);
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
  $("candidate-caption").textContent = "正在读取";
  const data = await fetchJson(
    `/api/pie/candidates/${encodeURIComponent(state.jobId)}/${encodeURIComponent(state.currentMz)}`,
  );
  state.candidateRows = data.rows || [];
  state.selectedCandidateIds = new Set(state.candidateRows.map((row) => Number(row.id)));
  await loadCandidateCurves();
  scheduleManualPreview();
  renderCandidatePanel();
}

function renderCandidatePanel() {
  const panel = $("candidate-panel");
  panel.classList.toggle("hidden", state.fitMode !== "manual");
  if (state.fitMode !== "manual") {
    updateFitActionState();
    return;
  }

  const rows = state.candidateRows || [];
  $("candidate-caption").textContent = rows.length
    ? `m/z ${state.currentMz} | ${rows.length} 条`
    : "无候选";
  const view = $("candidate-view");
  if (!rows.length) {
    view.innerHTML = '<div class="empty">当前 m/z 没有 PICS 候选。</div>';
    updateFitActionState();
    return;
  }
  const tbody = rows.map((row) => {
    const id = Number(row.id);
    const checked = state.selectedCandidateIds.has(id) ? " checked" : "";
    const locked = state.lockedCandidateIds.has(id) ? " checked" : "";
    const coefficient = state.candidateCoefficients[id] ?? "";
    return `<tr>
      <td><input type="checkbox" aria-label="选择候选 ${escapeHtml(row.name)}" data-candidate-check="${row.id}"${checked}></td>
      <td><input class="coefficient-input" type="number" min="0" step="0.000001" value="${escapeHtml(coefficient)}" aria-label="${escapeHtml(row.name)} 的拟合系数" data-candidate-coefficient="${row.id}"></td>
      <td class="lock-cell"><input type="checkbox" aria-label="锁定 ${escapeHtml(row.name)} 的拟合系数" data-candidate-lock="${row.id}"${locked}></td>
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
    button.classList.toggle("active", button.dataset.fitMode === mode);
    button.setAttribute("aria-pressed", String(button.dataset.fitMode === mode));
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
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(220, parent.clientWidth - 24);
  const height = width < 560 ? 320 : 420;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.setAttribute(
    "aria-label",
    state.currentCurve ? `PIE 曲线图，当前 m/z ${state.currentMz}` : "PIE 曲线图，暂无数据",
  );
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (!state.currentCurve) {
    drawEmptyChart(ctx, width, height, "生成曲线后显示 PIE 图");
    return;
  }

  const series = [
    {
      label: "实验 PIE",
      x: state.currentCurve.energies,
      y: state.currentCurve.intensities,
      color: "#1d5fbf",
      points: true,
    },
  ];
  const chartFit = state.manualPreviewFit || state.currentFit;
  if (chartFit && chartFit.fitted && chartFit.fitted.length) {
    series.push({
      label: chartFit.preview ? "实时预览" : "PICS 总拟合",
      x: chartFit.energies,
      y: chartFit.fitted,
      color: "#f97316",
      points: false,
    });
  }
  if ($("show-components").checked && chartFit && chartFit.species) {
    const colors = ["#059669", "#7c3aed", "#dc2626"];
    chartFit.species.slice(0, 6).forEach((item, index) => {
      series.push({
        label: item.species,
        x: chartFit.energies,
        y: item.component_intensities || [],
        color: colors[index % colors.length],
        points: false,
        dash: true,
      });
    });
  }

  const allX = series.flatMap((item) => item.x.map(Number)).filter(Number.isFinite);
  const allY = series.flatMap((item) => item.y.map(Number)).filter(Number.isFinite);
  if (!allX.length || !allY.length) {
    drawEmptyChart(ctx, width, height, "当前曲线没有有效数据");
    return;
  }

  let xMin = Math.min(...allX);
  let xMax = Math.max(...allX);
  let yMin = Math.min(0, Math.min(...allY));
  let yMax = Math.max(...allY);
  const xPad = Math.max(0.1, (xMax - xMin) * 0.08);
  const yPad = Math.max(1, (yMax - yMin) * 0.12);
  xMin -= xPad;
  xMax += xPad;
  yMax += yPad;
  if (xMin === xMax) xMax = xMin + 1;
  if (yMin === yMax) yMax = yMin + 1;

  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  const yTicks = Array.from({ length: 6 }, (_, i) => yMin + ((yMax - yMin) * i) / 5);
  const yLabelWidth = Math.max(...yTicks.map((value) => ctx.measureText(fmt(value, 3)).width));
  const margin = width < 360
    ? { left: Math.max(64, Math.ceil(yLabelWidth) + 42), right: 12, top: 30, bottom: 48 }
    : { left: Math.max(90, Math.ceil(yLabelWidth) + 50), right: 22, top: 34, bottom: 54 };
  const legendAvailableWidth = Math.max(120, width - margin.left - margin.right);
  let legendRows = 1;
  let legendRowWidth = 0;
  series.forEach((item) => {
    const itemWidth = Math.min(260, ctx.measureText(String(item.label)).width + 50);
    if (legendRowWidth > 0 && legendRowWidth + itemWidth > legendAvailableWidth) {
      legendRows += 1;
      legendRowWidth = 0;
    }
    legendRowWidth += itemWidth;
  });
  margin.top += legendRows * 20;
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const sx = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value) => margin.top + plotH - ((value - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#667085";
  ctx.textAlign = "right";
  yTicks.forEach((value) => {
    const y = sy(value);
    ctx.strokeStyle = "#eef1f5";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.fillText(fmt(value, 3), margin.left - 12, y + 4);
  });
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / 5;
    const x = sx(value);
    ctx.fillText(fmt(value, 4), x, height - 24);
  }
  ctx.fillText("Photon energy (eV)", margin.left + plotW / 2, height - 7);
  ctx.save();
  ctx.translate(18, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Normalized intensity", 0, 0);
  ctx.restore();
  ctx.textAlign = "start";

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
        ctx.moveTo(sx(x), sy(y));
        started = true;
      } else {
        ctx.lineTo(sx(x), sy(y));
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
        ctx.arc(sx(x), sy(y), 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });

  let legendX = margin.left;
  let legendY = 22;
  series.forEach((item) => {
    const label = fitCanvasText(ctx, item.label, Math.min(220, legendAvailableWidth - 44));
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
    ctx.fillStyle = "#344054";
    ctx.fillText(label, legendX + 28, legendY + 4);
    legendX += itemWidth;
  });
}

function drawEmptyChart(ctx, width, height, text) {
  ctx.fillStyle = "#667085";
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
  if (!rows.length) {
    view.innerHTML = `<div class="empty">${state.tableMode === "species" ? "当前曲线尚未拟合或无匹配物种。" : "没有可显示的数据。"}</div>`;
    $("download-button").disabled = true;
    return;
  }
  const columns = Object.keys(rows[0]);
  const thead = `<thead><tr>${columns.map((col) => `<th scope="col">${escapeHtml(col)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((row) => `<tr>${columns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  view.innerHTML = `<table>${thead}${tbody}</table>`;
  $("download-button").disabled = false;
}

async function setTableMode(mode) {
  state.tableMode = mode;
  document.querySelectorAll(".tab-button").forEach((button) => {
    const active = button.dataset.table === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
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

function switchPage(pageId) {
  document.querySelectorAll(".page").forEach((page) => {
    const active = page.id === pageId;
    page.classList.toggle("active", active);
    page.hidden = !active;
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    const active = button.dataset.page === pageId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
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
  const data = await fetchJson(`/api/pics/species/${encodeURIComponent(speciesId)}${suffix}`);
  state.currentPics = data;
  renderPicsSummary();
  drawPicsChart();
  renderPicsTable();
}

function renderPicsTable() {
  const view = $("pics-table-view");
  if (!state.picsRows.length) {
    view.innerHTML = '<div class="empty">输入条件后查询 PICS 数据库。</div>';
    $("pics-download-button").disabled = true;
    return;
  }
  const rows = state.picsRows;
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
  $("pics-download-button").disabled = false;
}

function drawPicsChart() {
  const canvas = $("pics-chart");
  if (!canvas) return;
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(220, parent.clientWidth - 24);
  const height = width < 560 ? 300 : 380;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.setAttribute(
    "aria-label",
    state.currentPics && state.currentPics.species
      ? `PICS 曲线图，当前物种 ${state.currentPics.species.name}`
      : "PICS 曲线图，暂无数据",
  );
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!state.currentPics || !state.currentPics.points || !state.currentPics.points.length) {
    drawEmptyChart(ctx, width, height, "选择物种后显示 PICS 曲线");
    return;
  }
  const xValues = state.currentPics.points.map((point) => Number(point.energy_ev)).filter(Number.isFinite);
  const yValues = state.currentPics.points.map((point) => Number(point.cross_section)).filter(Number.isFinite);
  if (!xValues.length || !yValues.length) {
    drawEmptyChart(ctx, width, height, "当前物种没有有效 PICS 数据");
    return;
  }
  let xMin = Math.min(...xValues);
  let xMax = Math.max(...xValues);
  let yMin = Math.min(0, Math.min(...yValues));
  let yMax = Math.max(...yValues);
  const xPad = Math.max(0.1, (xMax - xMin) * 0.08);
  const yPad = Math.max(0.1, (yMax - yMin) * 0.12);
  xMin -= xPad;
  xMax += xPad;
  yMax += yPad;
  if (xMin === xMax) xMax = xMin + 1;
  if (yMin === yMax) yMax = yMin + 1;
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  const yTicks = Array.from({ length: 6 }, (_, i) => yMin + ((yMax - yMin) * i) / 5);
  const yLabelWidth = Math.max(...yTicks.map((value) => ctx.measureText(fmt(value, 3)).width));
  const margin = width < 360
    ? { left: Math.max(64, Math.ceil(yLabelWidth) + 42), right: 12, top: 48, bottom: 48 }
    : { left: Math.max(90, Math.ceil(yLabelWidth) + 50), right: 22, top: 48, bottom: 54 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const sx = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value) => margin.top + plotH - ((value - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#667085";
  ctx.textAlign = "right";
  yTicks.forEach((value) => {
    const y = sy(value);
    ctx.strokeStyle = "#eef1f5";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.fillText(fmt(value, 3), margin.left - 12, y + 4);
  });
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / 5;
    const x = sx(value);
    ctx.fillText(fmt(value, 4), x, height - 24);
  }
  ctx.fillText("Photon energy (eV)", margin.left + plotW / 2, height - 7);
  ctx.save();
  ctx.translate(18, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Cross section", 0, 0);
  ctx.restore();
  ctx.textAlign = "start";

  ctx.strokeStyle = "#1d5fbf";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  state.currentPics.points.forEach((point, index) => {
    const x = sx(Number(point.energy_ev));
    const y = sy(Number(point.cross_section));
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#1d5fbf";
  state.currentPics.points.forEach((point) => {
    const x = sx(Number(point.energy_ev));
    const y = sy(Number(point.cross_section));
    ctx.beginPath();
    ctx.arc(x, y, 2.8, 0, Math.PI * 2);
    ctx.fill();
  });
  const species = state.currentPics.species;
  ctx.fillStyle = "#344054";
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
  const pill = $("upload-status-pill");
  const [className, label] = statusPresentation(status);
  pill.className = `pill ${className}`;
  pill.textContent = label;
  $("upload-status-text").textContent = text;
}

function setUploadProgress(value, step, status = "running") {
  const clamped = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  state.uploadProgressValue = clamped;
  const bar = $("upload-progress-bar");
  bar.style.width = `${clamped}%`;
  bar.classList.toggle("error", status === "error");
  $("upload-progress-percent").textContent = `${clamped}%`;
  $("upload-progress-step").textContent = step || "等待上传";
  const meter = bar.closest(".progress-meter");
  if (meter) {
    meter.setAttribute("role", "progressbar");
    meter.setAttribute("aria-valuemin", "0");
    meter.setAttribute("aria-valuemax", "100");
    meter.setAttribute("aria-valuenow", String(clamped));
    meter.setAttribute("aria-valuetext", `${step || "上传进度"} ${clamped}%`);
  }
}

function uploadLog(message) {
  const stamp = new Date().toLocaleTimeString();
  const text = String(message || "").trim();
  if (!text) return;
  if (
    state.uploadLogEntries.length
    && state.uploadLogEntries[state.uploadLogEntries.length - 1].endsWith(`  ${text}`)
  ) {
    return;
  }
  state.uploadLogEntries.push(`${stamp}  ${text}`);
  state.uploadLogEntries = state.uploadLogEntries.slice(-10);
  const view = $("upload-progress-log");
  view.textContent = state.uploadLogEntries.join("\n");
  view.scrollTop = view.scrollHeight;
}

function setPicsLibrary(libraryId) {
  state.picsLibraryId = libraryId || "";
  if (state.picsLibraryId) {
    localStorage.setItem("bl03u_pics_library_id", state.picsLibraryId);
  } else {
    localStorage.removeItem("bl03u_pics_library_id");
  }
  updatePicsLibraryCaptions();
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
  if (useServerButton) useServerButton.disabled = !state.picsLibraryId;
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
  return page ? page.id : "pie-page";
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

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => switchPage(button.dataset.page));
});
document.querySelectorAll(".source-button").forEach((button) => {
  button.addEventListener("click", () => {
    const source = button.dataset.pieSource;
    document.querySelectorAll(".source-button").forEach((item) => {
      const active = item.dataset.pieSource === source;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    $("query-form").classList.toggle("hidden", source !== "folder");
    $("query-form").classList.toggle("active", source === "folder");
    $("curve-upload-form").classList.toggle("hidden", source !== "upload");
    $("curve-upload-form").classList.toggle("active", source === "upload");
  });
});
$("query-form").addEventListener("submit", startJob);
$("curve-upload-form").addEventListener("submit", uploadPieCurve);
$("peak-source").addEventListener("change", () => {
  $("manual-peak-wrap").classList.toggle("hidden", $("peak-source").value !== "manual");
});
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
  if (button) loadPicsSpecies(button.dataset.picsId);
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
window.addEventListener("resize", () => {
  drawChart();
  drawPicsChart();
});
document.querySelectorAll(".page").forEach((page) => {
  page.hidden = !page.classList.contains("active");
});
updatePicsLibraryCaptions();
updateUploadScopeUi();
setProgress(0, "等待任务", "idle");
setUploadProgress(0, "等待上传", "idle");
drawChart();
drawPicsChart();
