const state = {
  source: "sample",
  samples: [],
  busy: false,
  modelAvailable: false,
  activeTab: "waveform",
  logs: [],
};

const el = {};

function byId(id) {
  return document.getElementById(id);
}

function bindElements() {
  [
    "modelBadge",
    "modelName",
    "checkpointPath",
    "sampleCount",
    "sampleSourceButton",
    "uploadSourceButton",
    "sampleControls",
    "uploadControls",
    "sampleSelect",
    "uploadFile",
    "outputDir",
    "waveformSeconds",
    "waveformSecondsText",
    "enhanceButton",
    "enhanceButtonText",
    "busySpinner",
    "resetButton",
    "statusMessage",
    "resultStamp",
    "inputName",
    "inputPath",
    "outputName",
    "outputPath",
    "originalAudio",
    "enhancedAudio",
    "downloadLink",
    "outputList",
    "initialLoadMetric",
    "readyMetric",
    "processMetric",
    "totalMetric",
    "waveformImage",
    "powerSpectrumImage",
    "melSpectrumImage",
    "metricsTable",
    "logList",
    "logCount",
  ].forEach((id) => {
    el[id] = byId(id);
  });
}

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} 秒` : "-";
}

function setBadge(kind, text) {
  el.modelBadge.className = `status-badge ${kind}`;
  el.modelBadge.textContent = text;
}

function setStatus(text, isError = false) {
  el.statusMessage.textContent = text;
  el.statusMessage.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function appendLog(message) {
  const timestamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  state.logs.push(`[${timestamp}] ${message}`);
  state.logs = state.logs.slice(-80);
  el.logList.replaceChildren();
  state.logs.forEach((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    el.logList.appendChild(item);
  });
  el.logCount.textContent = String(state.logs.length);
  el.logList.scrollTop = el.logList.scrollHeight;
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const detail = payload?.detail || response.statusText || "请求失败";
    throw new Error(detail);
  }
  return payload;
}

function selectedSample() {
  return state.samples.find((sample) => sample.path === el.sampleSelect.value) || null;
}

function sampleAudioUrl(path) {
  return `/api/sample-audio?path=${encodeURIComponent(path)}`;
}

function updateEnhanceButton() {
  const hasSample = state.source === "sample" && Boolean(selectedSample());
  const hasUpload = state.source === "upload" && el.uploadFile.files.length > 0;
  el.enhanceButton.disabled = state.busy || !state.modelAvailable || !(hasSample || hasUpload);
}

function setBusy(isBusy) {
  state.busy = isBusy;
  el.enhanceButtonText.textContent = isBusy ? "处理中" : "开始分离";
  el.busySpinner.classList.toggle("hidden", !isBusy);
  el.sampleSelect.disabled = isBusy;
  el.uploadFile.disabled = isBusy;
  el.outputDir.disabled = isBusy;
  el.waveformSeconds.disabled = isBusy;
  el.sampleSourceButton.disabled = isBusy;
  el.uploadSourceButton.disabled = isBusy;
  el.resetButton.disabled = isBusy;
  updateEnhanceButton();
}

function setSource(source) {
  state.source = source;
  const sampleActive = source === "sample";
  el.sampleSourceButton.classList.toggle("is-active", sampleActive);
  el.uploadSourceButton.classList.toggle("is-active", !sampleActive);
  el.sampleSourceButton.setAttribute("aria-pressed", String(sampleActive));
  el.uploadSourceButton.setAttribute("aria-pressed", String(!sampleActive));
  el.sampleControls.classList.toggle("hidden", !sampleActive);
  el.uploadControls.classList.toggle("hidden", sampleActive);
  resetResult(false);
  updateInputPreview();
  updateEnhanceButton();
}

function updateInputPreview() {
  if (state.source === "sample") {
    const sample = selectedSample();
    if (!sample) {
      el.originalAudio.removeAttribute("src");
      el.inputName.textContent = "未选择";
      el.inputPath.textContent = "-";
      return;
    }
    el.originalAudio.src = sampleAudioUrl(sample.path);
    el.inputName.textContent = sample.name;
    el.inputPath.textContent = sample.path;
    return;
  }

  const file = el.uploadFile.files[0];
  if (!file) {
    el.originalAudio.removeAttribute("src");
    el.inputName.textContent = "未选择";
    el.inputPath.textContent = "-";
    return;
  }
  el.originalAudio.src = URL.createObjectURL(file);
  el.inputName.textContent = file.name;
  el.inputPath.textContent = file.name;
}

function clearAnalysisPanels() {
  [
    [el.waveformImage, byId("waveformPanel")],
    [el.powerSpectrumImage, byId("spectrumPanel")],
    [el.melSpectrumImage, byId("spectrumPanel")],
  ].forEach(([image, panel]) => {
    image.removeAttribute("src");
    panel.classList.remove("has-result");
  });
  el.metricsTable.querySelector("thead").replaceChildren();
  el.metricsTable.querySelector("tbody").replaceChildren();
  byId("metricsPanel").classList.remove("has-result");
}

function resetResult(writeLog = true) {
  el.enhancedAudio.removeAttribute("src");
  el.outputName.textContent = "无结果";
  el.outputName.classList.add("muted");
  el.outputPath.textContent = "-";
  el.outputList.replaceChildren();
  el.downloadLink.href = "#";
  el.downloadLink.classList.add("is-disabled");
  el.downloadLink.setAttribute("aria-disabled", "true");
  el.resultStamp.textContent = "等待处理";
  el.initialLoadMetric.textContent = "-";
  el.readyMetric.textContent = "-";
  el.processMetric.textContent = "-";
  el.totalMetric.textContent = "-";
  clearAnalysisPanels();
  setStatus("");
  if (writeLog) {
    appendLog("已清空当前结果");
  }
}

function renderTiming(timing) {
  el.initialLoadMetric.textContent = formatSeconds(timing.model_initial_load_seconds);
  el.readyMetric.textContent = formatSeconds(timing.model_ready_seconds);
  el.processMetric.textContent = formatSeconds(timing.process_seconds);
  el.totalMetric.textContent = formatSeconds(timing.total_seconds);
}

function renderMetrics(rows) {
  const head = el.metricsTable.querySelector("thead");
  const body = el.metricsTable.querySelector("tbody");
  head.replaceChildren();
  body.replaceChildren();

  if (!rows.length) {
    byId("metricsPanel").classList.remove("has-result");
    return;
  }

  const headerRow = document.createElement("tr");
  Object.keys(rows[0]).forEach((key) => {
    const cell = document.createElement("th");
    cell.textContent = key;
    headerRow.appendChild(cell);
  });
  head.appendChild(headerRow);

  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    Object.values(row).forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      tableRow.appendChild(cell);
    });
    body.appendChild(tableRow);
  });
  byId("metricsPanel").classList.add("has-result");
}

function renderResult(data) {
  const outputs = data.outputs || [];
  const primaryOutput = outputs[0] || null;
  el.enhancedAudio.src = primaryOutput?.audio_url || data.output_audio_url;
  el.outputName.textContent = data.output_name;
  el.outputName.classList.remove("muted");
  el.outputPath.textContent = outputs.map((item) => item.path).join("\n") || data.output_path;
  el.downloadLink.href = primaryOutput?.download_url || data.download_url;
  el.downloadLink.download = primaryOutput?.name || data.output_name;
  el.downloadLink.classList.remove("is-disabled");
  el.downloadLink.setAttribute("aria-disabled", "false");
  el.resultStamp.textContent = data.created_at;
  el.outputList.replaceChildren();

  outputs.forEach((item) => {
    const row = document.createElement("div");
    row.className = "output-row";

    const label = document.createElement("span");
    label.className = "output-row-label";
    label.textContent = item.label;

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = item.audio_url;

    const link = document.createElement("a");
    link.className = "download-link compact";
    link.href = item.download_url;
    link.download = item.name;
    link.textContent = "下载";

    row.append(label, audio, link);
    el.outputList.appendChild(row);
  });

  el.waveformImage.src = data.analysis.waveform_image;
  el.powerSpectrumImage.src = data.analysis.power_spectrum_image;
  el.melSpectrumImage.src = data.analysis.mel_spectrum_image;
  byId("waveformPanel").classList.add("has-result");
  byId("spectrumPanel").classList.add("has-result");
  renderMetrics(data.analysis.metrics_rows || []);
  renderTiming(data.timing);

  data.logs.forEach((line) => appendLog(line));
  setStatus("处理完成");
}

function activateTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab-button").forEach((button) => {
    const active = button.dataset.tab === tabName;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  ["waveform", "spectrum", "metrics"].forEach((name) => {
    byId(`${name}Panel`).classList.toggle("hidden", name !== tabName);
  });
}

async function loadHealth() {
  const health = await apiJson("/api/health");
  state.modelAvailable = Boolean(health.model_available);
  el.modelName.textContent = health.model_name;
  el.checkpointPath.textContent = health.checkpoint_dir;
  el.sampleCount.textContent = String(health.sample_count);
  el.uploadFile.accept = health.supported_extensions.map((extension) => `.${extension}`).join(",");
  el.outputDir.value = health.default_output_dir;

  if (state.modelAvailable) {
    setBadge("status-ready", "模型就绪");
  } else {
    setBadge("status-missing", "模型未就绪");
    setStatus("模型权重目录未就绪", true);
  }
  updateEnhanceButton();
}

async function loadSamples() {
  const payload = await apiJson("/api/samples");
  state.samples = payload.samples || [];
  el.sampleSelect.replaceChildren();

  state.samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.path;
    option.textContent = sample.name;
    el.sampleSelect.appendChild(option);
  });

  if (!state.samples.length) {
    const option = document.createElement("option");
    option.textContent = "未找到示例音频";
    option.value = "";
    el.sampleSelect.appendChild(option);
    el.sampleSelect.disabled = true;
  }
  updateInputPreview();
  updateEnhanceButton();
}

async function runEnhancement() {
  resetResult(false);
  const formData = new FormData();
  formData.append("source_type", state.source);
  formData.append("output_dir", el.outputDir.value);
  formData.append("waveform_seconds", el.waveformSeconds.value);

  if (state.source === "sample") {
    const sample = selectedSample();
    if (!sample) {
      setStatus("未选择示例音频", true);
      return;
    }
    formData.append("sample_path", sample.path);
    appendLog(`开始处理: ${sample.path}`);
  } else {
    const file = el.uploadFile.files[0];
    if (!file) {
      setStatus("未上传音频文件", true);
      return;
    }
    formData.append("file", file);
    appendLog(`开始处理: ${file.name}`);
  }

  setBusy(true);
  setBadge("status-busy", "处理中");
  setStatus("处理请求已提交");

  try {
    const data = await apiJson("/api/separate", {
      method: "POST",
      body: formData,
    });
    renderResult(data);
    setBadge(state.modelAvailable ? "status-ready" : "status-missing", state.modelAvailable ? "模型就绪" : "模型未就绪");
  } catch (error) {
    appendLog(`处理失败: ${error.message}`);
    setStatus(error.message, true);
    setBadge(state.modelAvailable ? "status-ready" : "status-error", state.modelAvailable ? "模型就绪" : "处理失败");
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  el.sampleSourceButton.addEventListener("click", () => setSource("sample"));
  el.uploadSourceButton.addEventListener("click", () => setSource("upload"));
  el.sampleSelect.addEventListener("change", () => {
    resetResult(false);
    updateInputPreview();
    updateEnhanceButton();
  });
  el.uploadFile.addEventListener("change", () => {
    resetResult(false);
    updateInputPreview();
    updateEnhanceButton();
  });
  el.waveformSeconds.addEventListener("input", () => {
    el.waveformSecondsText.textContent = `${el.waveformSeconds.value} 秒`;
  });
  el.enhanceButton.addEventListener("click", runEnhancement);
  el.resetButton.addEventListener("click", () => {
    resetResult(true);
    updateInputPreview();
  });
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });
}

async function init() {
  bindElements();
  bindEvents();
  appendLog("WebUI 已启动");
  try {
    await loadHealth();
    await loadSamples();
  } catch (error) {
    setBadge("status-error", "检查失败");
    setStatus(error.message, true);
    appendLog(`初始化失败: ${error.message}`);
  }
}

document.addEventListener("DOMContentLoaded", init);
