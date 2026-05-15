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
  picsLibraryId: localStorage.getItem("bl03u_pics_library_id") || "",
};

const $ = (id) => document.getElementById(id);

function setStatus(status, text) {
  const pill = $("status-pill");
  pill.className = `pill ${status}`;
  pill.textContent = status === "idle" ? "Idle" : status;
  $("status-text").textContent = text;
}

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  $("progress-log").textContent = `${stamp}  ${message}`;
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

function setBusy(busy) {
  $("start-button").disabled = busy;
  $("upload-curve-button").disabled = busy;
  $("fit-button").disabled = busy || !state.currentCurve;
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
  $("mz-select").innerHTML = "";
  $("mz-select").disabled = true;
  $("fit-button").disabled = true;
  $("show-components").disabled = true;
  $("source-caption").textContent = caption;
  renderSummary({});
  clearChart();
  renderFitSummary(null);
  renderCandidatePanel();
  renderTable();
}

async function startJob(event) {
  event.preventDefault();
  resetPieState("任务提交中");

  try {
    setBusy(true);
    setStatus("running", "提交任务");
    const data = await fetchJson("/api/pie/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    state.jobId = data.job_id;
    log(`任务已提交: ${state.jobId}`);
    pollProgress();
  } catch (error) {
    setBusy(false);
    setStatus("error", error.message);
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
    await loadResult();
    setStatus("done", "PIE 曲线已载入");
  } catch (error) {
    setStatus("error", error.message);
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
    log(`${data.message} | ${data.elapsed}s`);
    if (data.status === "done") {
      state.summary = data.summary;
      await loadResult();
      setBusy(false);
      return;
    }
    if (data.status === "error") {
      setBusy(false);
      log(data.error || "任务失败");
      return;
    }
    setTimeout(pollProgress, 900);
  } catch (error) {
    setBusy(false);
    setStatus("error", error.message);
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
  $("fit-button").disabled = false;
  $("show-components").disabled = !state.currentFit;
  renderFitSummary(state.currentFit);
  if (state.fitMode === "manual") {
    await loadCandidates();
  } else {
    renderCandidatePanel();
  }
  drawChart();
  renderTable();
}

async function fitCurrentCurve() {
  if (!state.jobId || !state.currentMz) return;
  try {
    $("fit-button").disabled = true;
    setStatus("running", `拟合 m/z ${state.currentMz}`);
    const options = { method: "POST" };
    if (state.fitMode === "manual") {
      const selected = Array.from(state.selectedCandidateIds);
      if (!selected.length) {
        throw new Error("请至少选择一个 PICS 候选");
      }
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify({ species_ids: selected });
    }
    const data = await fetchJson(
      `/api/pie/fit/${encodeURIComponent(state.jobId)}/${encodeURIComponent(state.currentMz)}`,
      options,
    );
    state.currentFit = data.fit;
    $("show-components").disabled = !(state.currentFit && state.currentFit.species && state.currentFit.species.length);
    renderFitSummary(state.currentFit);
    drawChart();
    renderTable();
    setStatus("done", `m/z ${state.currentMz} 拟合完成`);
  } catch (error) {
    setStatus("error", error.message);
    log(error.message);
  } finally {
    $("fit-button").disabled = false;
  }
}

function renderFitSummary(fit) {
  const values = [
    ["当前 m/z", state.currentMz || "-"],
    ["候选 PICS", fit ? fit.candidate_count || 0 : 0],
    ["拟合物种", fit && fit.species ? fit.species.length : 0],
    ["R²", fit ? fmt(fit.r_squared, 5) : "未拟合"],
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
  renderCandidatePanel();
}

function renderCandidatePanel() {
  const panel = $("candidate-panel");
  panel.classList.toggle("hidden", state.fitMode !== "manual");
  if (state.fitMode !== "manual") return;

  const rows = state.candidateRows || [];
  $("candidate-caption").textContent = rows.length
    ? `m/z ${state.currentMz} | ${rows.length} 条`
    : "无候选";
  const view = $("candidate-view");
  if (!rows.length) {
    view.innerHTML = '<div class="empty">当前 m/z 没有 PICS 候选。</div>';
    return;
  }
  const tbody = rows.map((row) => {
    const checked = state.selectedCandidateIds.has(Number(row.id)) ? " checked" : "";
    return `<tr>
      <td><input type="checkbox" data-candidate-check="${row.id}"${checked}></td>
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
    <table>
      <thead><tr><th></th><th>id</th><th>m/z</th><th>物种</th><th>IE(eV)</th><th>能量范围</th><th>点数</th><th>最大截面</th></tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
}

async function setFitMode(mode) {
  state.fitMode = mode;
  document.querySelectorAll(".segment-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.fitMode === mode);
  });
  $("fit-button").textContent = mode === "manual" ? "按勾选拟合" : "拟合当前曲线";
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
  const width = Math.max(640, parent.clientWidth - 24);
  const height = 420;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
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
  if (state.currentFit && state.currentFit.fitted && state.currentFit.fitted.length) {
    series.push({
      label: "PICS 总拟合",
      x: state.currentFit.energies,
      y: state.currentFit.fitted,
      color: "#f97316",
      points: false,
    });
  }
  if ($("show-components").checked && state.currentFit && state.currentFit.species) {
    const colors = ["#059669", "#7c3aed", "#dc2626"];
    state.currentFit.species.slice(0, 3).forEach((item, index) => {
      series.push({
        label: item.species,
        x: state.currentFit.energies,
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

  const margin = { left: 64, right: 22, top: 26, bottom: 52 };
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
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  for (let i = 0; i <= 5; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 5;
    const y = sy(value);
    ctx.strokeStyle = "#eef1f5";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.fillText(fmt(value, 3), 8, y + 4);
  }
  for (let i = 0; i <= 5; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / 5;
    const x = sx(value);
    ctx.fillText(fmt(value, 4), x - 14, height - 22);
  }
  ctx.fillText("Photon energy (eV)", margin.left + plotW / 2 - 48, height - 6);
  ctx.save();
  ctx.translate(14, margin.top + plotH / 2 + 44);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Normalized intensity", 0, 0);
  ctx.restore();

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

  let legendX = margin.left + 10;
  const legendY = margin.top + 8;
  series.forEach((item) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 22, legendY);
    ctx.stroke();
    ctx.fillStyle = "#344054";
    ctx.fillText(String(item.label).slice(0, 24), legendX + 28, legendY + 4);
    legendX += 148;
  });
}

function drawEmptyChart(ctx, width, height, text) {
  ctx.fillStyle = "#667085";
  ctx.font = "14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.fillText(text, width / 2 - 90, height / 2);
}

function rowsForTable() {
  if (state.tableMode === "species") {
    return state.currentFit && state.currentFit.species
      ? state.currentFit.species.map((item) => ({
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
  const thead = `<thead><tr>${columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows
    .map((row) => `<tr>${columns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`)
    .join("")}</tbody>`;
  view.innerHTML = `<table>${thead}${tbody}</table>`;
  $("download-button").disabled = false;
}

async function setTableMode(mode) {
  state.tableMode = mode;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.table === mode);
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
    page.classList.toggle("active", page.id === pageId);
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === pageId);
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
      <td><button class="small-button" type="button" data-pics-id="${row.id}">查看</button></td>
    </tr>`;
  }).join("");
  view.innerHTML = `<table><thead><tr>${headers.map((item) => `<th>${item}</th>`).join("")}</tr></thead><tbody>${tbody}</tbody></table>`;
  $("pics-download-button").disabled = false;
}

function drawPicsChart() {
  const canvas = $("pics-chart");
  if (!canvas) return;
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(640, parent.clientWidth - 24);
  const height = 380;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
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
  const margin = { left: 64, right: 22, top: 28, bottom: 52 };
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
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  for (let i = 0; i <= 5; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 5;
    const y = sy(value);
    ctx.strokeStyle = "#eef1f5";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.fillText(fmt(value, 3), 8, y + 4);
  }
  for (let i = 0; i <= 5; i += 1) {
    const value = xMin + ((xMax - xMin) * i) / 5;
    const x = sx(value);
    ctx.fillText(fmt(value, 4), x - 14, height - 22);
  }
  ctx.fillText("Photon energy (eV)", margin.left + plotW / 2 - 48, height - 6);
  ctx.save();
  ctx.translate(14, margin.top + plotH / 2 + 48);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Cross section", 0, 0);
  ctx.restore();

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
  ctx.fillText(`${species.name} | m/z ${species.mz}`, margin.left + 10, margin.top + 2);
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
  pill.className = `pill ${status}`;
  pill.textContent = status === "idle" ? "Idle" : status;
  $("upload-status-text").textContent = text;
}

function uploadLog(message) {
  const stamp = new Date().toLocaleTimeString();
  $("upload-progress-log").textContent = `${stamp}  ${message}`;
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
    uploadLog("未选择 PICS 数据文件");
    return;
  }
  const scope = $("pics-upload-scope").value;
  const mode = $("pics-upload-mode").value;
  const confirmOverwrite = $("overwrite-confirm").checked;
  const adminToken = $("admin-token").value.trim();
  if (scope === "server" && !adminToken) {
    setUploadStatus("error", "需要管理员 token");
    uploadLog("写入服务器维护库前需要填写管理员 token");
    return;
  }
  if (scope === "server" && mode === "overwrite_all" && !confirmOverwrite) {
    setUploadStatus("error", "需要确认覆盖");
    uploadLog("覆盖整个库前需要勾选确认");
    return;
  }

  $("pics-upload-button").disabled = true;
  setUploadStatus("running", "上传中");
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
    if (result.scope === "session") {
      setPicsLibrary(result.library_id);
      setUploadStatus("done", "临时工作库已启用");
      uploadLog(`临时导入 ${result.inserted_species} 个物种，${result.inserted_points} 个点`);
    } else {
      setPicsLibrary("");
      setUploadStatus("done", "服务器维护库已更新");
      uploadLog(`写入 ${result.inserted_species} 个物种，${result.inserted_points} 个点`);
    }
    renderUploadResult(result);
  } catch (error) {
    setUploadStatus("error", "上传失败");
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
    <table>
      <tbody>
        ${rows.map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("")}
      </tbody>
    </table>
  `;
}

function csvCell(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => switchPage(button.dataset.page));
});
document.querySelectorAll(".source-button").forEach((button) => {
  button.addEventListener("click", () => {
    const source = button.dataset.pieSource;
    document.querySelectorAll(".source-button").forEach((item) => {
      item.classList.toggle("active", item.dataset.pieSource === source);
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
document.querySelectorAll(".segment-button").forEach((button) => {
  button.addEventListener("click", () => setFitMode(button.dataset.fitMode));
});
$("candidate-view").addEventListener("change", (event) => {
  const input = event.target.closest("[data-candidate-check]");
  if (!input) return;
  const id = Number(input.dataset.candidateCheck);
  if (input.checked) state.selectedCandidateIds.add(id);
  else state.selectedCandidateIds.delete(id);
});
$("select-all-candidates").addEventListener("click", () => {
  state.selectedCandidateIds = new Set(state.candidateRows.map((row) => Number(row.id)));
  renderCandidatePanel();
});
$("clear-candidates").addEventListener("click", () => {
  state.selectedCandidateIds = new Set();
  renderCandidatePanel();
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
  uploadLog("后续查询和拟合将使用服务器维护库");
});
window.addEventListener("resize", () => {
  drawChart();
  drawPicsChart();
});
updatePicsLibraryCaptions();
updateUploadScopeUi();
drawChart();
drawPicsChart();
