const TOOL_ORDER = ["separate", "enhance", "super_resolve"];
const TOOL_LABELS = {
  enhance: "增强",
  separate: "分离",
  super_resolve: "超分",
  "speaker-1": "说话人 1",
  "speaker-2": "说话人 2",
  enhanced: "增强",
  super_resolved: "超分",
  input: "原音频",
};

const state = {
  source: "sample",
  samples: [],
  tools: [],
  selectedTools: new Set(),
  busy: false,
  activeTab: "waveform",
  logs: [],
  task: null,
  selectedRef: "input",
  selectedLabel: "原音频",
};

const el = {};

function byId(id) {
  return document.getElementById(id);
}

function bindElements() {
  [
    "sceneBadge",
    "toolList",
    "sampleSourceButton",
    "uploadSourceButton",
    "sampleControls",
    "uploadControls",
    "sampleSelect",
    "uploadFile",
    "waveformSeconds",
    "waveformSecondsText",
    "runButton",
    "runButtonText",
    "busySpinner",
    "resetButton",
    "selectedArtifactLabel",
    "taskSelect",
    "openTaskButton",
    "reprocessTool",
    "reprocessButton",
    "statusMessage",
    "resultStamp",
    "inputName",
    "inputPath",
    "outputName",
    "outputPath",
    "originalAudio",
    "resultAudio",
    "downloadLink",
    "stepList",
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
  el.sceneBadge.className = `status-badge ${kind}`;
  el.sceneBadge.textContent = text;
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

function availableTools() {
  return state.tools.filter((tool) => tool.available);
}

function checkedAvailableTools() {
  return TOOL_ORDER.filter((id) => state.selectedTools.has(id) && state.tools.some((tool) => tool.id === id && tool.available));
}

function updateButtons() {
  const hasSample = state.source === "sample" && Boolean(selectedSample());
  const hasUpload = state.source === "upload" && el.uploadFile.files.length > 0;
  const hasSource = hasSample || hasUpload;
  const hasTools = checkedAvailableTools().length > 0;
  el.runButton.disabled = state.busy || !hasSource || !hasTools;
  const reprocessReady = Boolean(state.task) && Boolean(el.reprocessTool.value);
  el.reprocessButton.disabled = state.busy || !reprocessReady;
  if (el.openTaskButton) {
    el.openTaskButton.disabled = state.busy || !el.taskSelect.value;
  }
}

function setBusy(isBusy, label) {
  state.busy = isBusy;
  el.runButtonText.textContent = isBusy ? "处理中" : "开始处理";
  el.busySpinner.classList.toggle("hidden", !isBusy);
  el.sampleSelect.disabled = isBusy;
  el.uploadFile.disabled = isBusy;
  el.waveformSeconds.disabled = isBusy;
  el.sampleSourceButton.disabled = isBusy;
  el.uploadSourceButton.disabled = isBusy;
  el.reprocessTool.disabled = isBusy;
  document.querySelectorAll('#toolList input[type="checkbox"]').forEach((input) => {
    input.disabled = isBusy || input.dataset.available !== "true";
  });
  if (label) {
    setStatus(label);
  }
  updateButtons();
}

function setSource(source) {
  state.source = source;
  el.sampleSourceButton.classList.toggle("is-active", source === "sample");
  el.uploadSourceButton.classList.toggle("is-active", source === "upload");
  el.sampleSourceButton.setAttribute("aria-pressed", String(source === "sample"));
  el.uploadSourceButton.setAttribute("aria-pressed", String(source === "upload"));
  el.sampleControls.classList.toggle("hidden", source !== "sample");
  el.uploadControls.classList.toggle("hidden", source !== "upload");
  resetResult(false);
  updateInputPreview();
  updateButtons();
}

function applySuggestedTools(sample) {
  state.selectedTools = new Set();
  const suggested = sample?.suggested_tools || [];
  suggested.forEach((id) => {
    const tool = state.tools.find((item) => item.id === id);
    if (tool?.available) {
      state.selectedTools.add(id);
    }
  });
  renderToolList();
}

function renderToolList() {
  el.toolList.replaceChildren();
  state.tools.forEach((tool) => {
    const row = document.createElement("div");
    row.className = "tool-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `tool-${tool.id}`;
    checkbox.dataset.available = String(Boolean(tool.available));
    checkbox.checked = state.selectedTools.has(tool.id);
    checkbox.disabled = state.busy || !tool.available;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        state.selectedTools.add(tool.id);
      } else {
        state.selectedTools.delete(tool.id);
      }
      updateButtons();
    });
    const label = document.createElement("label");
    label.htmlFor = checkbox.id;
    label.innerHTML = `<span>${TOOL_LABELS[tool.id] || tool.id}</span><small>${tool.model}</small>`;
    const badge = document.createElement("span");
    badge.className = `status-badge ${tool.available ? "status-ready" : "status-missing"}`;
    badge.textContent = tool.available ? "就绪" : "未就绪";
    row.append(checkbox, label, badge);
    el.toolList.appendChild(row);
  });
  renderReprocessTools();
  updateButtons();
}

function renderReprocessTools() {
  el.reprocessTool.replaceChildren();
  availableTools().forEach((tool) => {
    const option = document.createElement("option");
    option.value = tool.id;
    option.textContent = TOOL_LABELS[tool.id] || tool.id;
    el.reprocessTool.appendChild(option);
  });
}

function updateInputPreview() {
  if (state.task?.input_audio_url) {
    el.inputName.textContent = state.task.input?.name || "当前任务";
    el.inputName.classList.remove("muted");
    el.originalAudio.src = state.task.input_audio_url;
    el.inputPath.textContent = state.task.input_path || "-";
    return;
  }
  if (state.source === "sample") {
    const sample = selectedSample();
    if (sample) {
      el.inputName.textContent = sample.name;
      el.inputName.classList.remove("muted");
      el.originalAudio.src = sample.audio_url;
      el.inputPath.textContent = sample.path;
      return;
    }
  } else if (el.uploadFile.files[0]) {
    const file = el.uploadFile.files[0];
    el.inputName.textContent = file.name;
    el.inputName.classList.remove("muted");
    el.originalAudio.src = URL.createObjectURL(file);
    el.inputPath.textContent = file.name;
    return;
  }
  el.inputName.textContent = "未选择";
  el.originalAudio.removeAttribute("src");
  el.inputPath.textContent = "-";
}

function renderMetrics(rows) {
  const table = el.metricsTable;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.replaceChildren();
  tbody.replaceChildren();
  if (!rows?.length) {
    return;
  }
  const keys = Object.keys(rows[0]);
  const headRow = document.createElement("tr");
  keys.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      td.textContent = row[key] ?? "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderAnalysis(analysis) {
  if (!analysis) {
    return;
  }
  el.waveformImage.src = analysis.waveform_image;
  el.powerSpectrumImage.src = analysis.power_spectrum_image;
  el.melSpectrumImage.src = analysis.mel_spectrum_image;
  byId("waveformPanel").classList.add("has-result");
  byId("spectrumPanel").classList.add("has-result");
  byId("metricsPanel").classList.add("has-result");
  renderMetrics(analysis.metrics_rows || []);
}

function renderTiming(timing) {
  el.initialLoadMetric.textContent = formatSeconds(timing?.model_initial_load_seconds);
  el.readyMetric.textContent = formatSeconds(timing?.model_ready_seconds);
  el.processMetric.textContent = formatSeconds(timing?.process_seconds);
  el.totalMetric.textContent = formatSeconds(timing?.total_seconds);
}

async function selectArtifact(ref, label, audioUrl, downloadUrl, path, options = {}) {
  state.selectedRef = ref;
  state.selectedLabel = label;
  el.selectedArtifactLabel.textContent = `当前选中：${label}`;
  el.outputName.textContent = label;
  el.outputName.classList.remove("muted");
  if (audioUrl) {
    el.resultAudio.src = audioUrl;
  }
  el.outputPath.textContent = path || "-";
  if (downloadUrl) {
    el.downloadLink.href = downloadUrl;
    el.downloadLink.download = label;
    el.downloadLink.classList.remove("is-disabled");
    el.downloadLink.setAttribute("aria-disabled", "false");
  } else {
    el.downloadLink.href = "#";
    el.downloadLink.classList.add("is-disabled");
    el.downloadLink.setAttribute("aria-disabled", "true");
  }
  document.querySelectorAll(".artifact-card").forEach((card) => {
    card.classList.toggle("is-selected", card.dataset.ref === ref);
  });
  updateButtons();
  if (options.skipAnalysisFetch || !state.task || !ref.startsWith("step:")) {
    return;
  }
  const runId = ref.split(":")[1];
  if (!runId) {
    return;
  }
  try {
    const data = await apiJson(
      `/api/tasks/${state.task.task_id}/runs/${runId}?waveform_seconds=${encodeURIComponent(el.waveformSeconds.value)}`,
    );
    renderAnalysis(data.analysis);
  } catch (error) {
    appendLog(`读取步骤分析失败: ${error.message}`);
  }
}

function artifactCard(output, runId) {
  const ref = `step:${runId}:${output.id}`;
  const card = document.createElement("div");
  card.className = "artifact-card";
  card.dataset.ref = ref;
  const title = TOOL_LABELS[output.id] || output.id;
  const heading = document.createElement("button");
  heading.type = "button";
  heading.className = "secondary-button";
  heading.textContent = `选中${title}`;
  heading.addEventListener("click", () => {
    void selectArtifact(ref, title, output.audio_url, output.download_url, output.path);
  });
  const label = document.createElement("strong");
  label.textContent = title;
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "metadata";
  audio.src = output.audio_url;
  const link = document.createElement("a");
  link.className = "download-link";
  link.href = output.download_url;
  link.download = output.name || title;
  link.textContent = `下载${title}`;
  card.append(label, audio, heading, link);
  return card;
}

function renderSteps(task) {
  const steps = task?.steps || [];
  el.stepList.replaceChildren();
  if (!steps.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "尚未开始。勾选工具后点开始处理；之后可对某一步产物再跑另一件工具。";
    el.stepList.appendChild(empty);
    return;
  }
  const inputCard = document.createElement("article");
  inputCard.className = "step-card";
  inputCard.innerHTML = "<h3>导入</h3>";
  const inputWrap = document.createElement("div");
  inputWrap.className = "artifact-card";
  inputWrap.dataset.ref = "input";
  const inputLabel = document.createElement("strong");
  inputLabel.textContent = "原音频";
  const inputSelect = document.createElement("button");
  inputSelect.type = "button";
  inputSelect.className = "secondary-button";
  inputSelect.textContent = "选中原音频";
  inputSelect.addEventListener("click", () => {
    void selectArtifact("input", "原音频", task.input_audio_url, null, task.input_path);
  });
  inputWrap.append(inputLabel, inputSelect);
  const inputGrid = document.createElement("div");
  inputGrid.className = "artifact-grid";
  inputGrid.appendChild(inputWrap);
  inputCard.appendChild(inputGrid);
  el.stepList.appendChild(inputCard);

  steps.forEach((step, index) => {
    const card = document.createElement("article");
    card.className = "step-card";
    const heading = document.createElement("h3");
    heading.textContent = `步骤 ${index + 1} · ${TOOL_LABELS[step.tool] || step.tool}`;
    const meta = document.createElement("p");
    meta.className = "order-hint";
    meta.textContent = `采样率 ${step.input_sample_rate} → ${step.output_sample_rate} Hz`;
    const grid = document.createElement("div");
    grid.className = "artifact-grid";
    (step.outputs || []).forEach((output) => {
      grid.appendChild(artifactCard(output, step.run_id));
    });
    card.append(heading, meta, grid);
    el.stepList.appendChild(card);
  });
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

function resetResult(clearLogs) {
  state.task = null;
  state.selectedRef = "input";
  state.selectedLabel = "原音频";
  el.outputName.textContent = "无结果";
  el.outputName.classList.add("muted");
  el.outputPath.textContent = "-";
  el.resultAudio.removeAttribute("src");
  el.downloadLink.href = "#";
  el.downloadLink.classList.add("is-disabled");
  el.resultStamp.textContent = "等待处理";
  el.selectedArtifactLabel.textContent = "尚未选中产物";
  ["waveformPanel", "spectrumPanel", "metricsPanel"].forEach((id) => byId(id).classList.remove("has-result"));
  el.metricsTable.querySelector("thead").replaceChildren();
  el.metricsTable.querySelector("tbody").replaceChildren();
  renderTiming({});
  renderSteps(null);
  if (clearLogs) {
    state.logs = [];
    el.logList.replaceChildren();
    el.logCount.textContent = "0";
  }
  updateButtons();
}

async function createTask() {
  const formData = new FormData();
  formData.append("source_type", state.source);
  if (state.source === "sample") {
    const sample = selectedSample();
    if (!sample) {
      throw new Error("未选择示例音频");
    }
    formData.append("sample_path", sample.path);
    appendLog(`创建任务: ${sample.path}`);
  } else {
    const file = el.uploadFile.files[0];
    if (!file) {
      throw new Error("未上传音频文件");
    }
    formData.append("file", file);
    appendLog(`创建任务: ${file.name}`);
  }
  const task = await apiJson("/api/tasks", { method: "POST", body: formData });
  state.task = task;
  updateInputPreview();
  renderSteps(task);
  await selectArtifact("input", "原音频", task.input_audio_url, null, task.input_path);
  await loadTaskList(task.task_id);
  return task;
}

async function loadTaskList(selectedId) {
  const payload = await apiJson("/api/tasks");
  const tasks = payload.tasks || [];
  const current = selectedId || state.task?.task_id || el.taskSelect.value;
  el.taskSelect.replaceChildren();
  if (!tasks.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "暂无磁盘任务";
    el.taskSelect.appendChild(option);
    el.openTaskButton.disabled = true;
    return;
  }
  tasks.forEach((task) => {
    const option = document.createElement("option");
    option.value = task.task_id;
    option.textContent = `${task.title} · ${task.step_count} 步`;
    el.taskSelect.appendChild(option);
  });
  if (current && tasks.some((task) => task.task_id === current)) {
    el.taskSelect.value = current;
  }
  el.openTaskButton.disabled = state.busy || !el.taskSelect.value;
}

async function openSelectedTask() {
  const taskId = el.taskSelect.value;
  if (!taskId) {
    setStatus("没有可打开的任务", true);
    return;
  }
  const task = await apiJson(`/api/tasks/${taskId}`);
  state.task = task;
  renderSteps(task);
  updateInputPreview();
  const last = (task.steps || []).at(-1);
  const firstOutput = last?.outputs?.[0];
  if (firstOutput) {
    await selectArtifact(
      `step:${last.run_id}:${firstOutput.id}`,
      TOOL_LABELS[firstOutput.id] || firstOutput.id,
      firstOutput.audio_url,
      firstOutput.download_url,
      firstOutput.path,
    );
  } else {
    await selectArtifact("input", "原音频", task.input_audio_url, null, task.input_path);
  }
  el.resultStamp.textContent = task.created_at;
  setStatus("已打开磁盘任务");
  appendLog(`打开任务: ${task.task_id}`);
  updateButtons();
}

async function runToolOnRef(toolId, inputRef) {
  const formData = new FormData();
  formData.append("tool", toolId);
  formData.append("input_ref", inputRef);
  formData.append("waveform_seconds", el.waveformSeconds.value);
  appendLog(`调用 ${TOOL_LABELS[toolId] || toolId} ← ${inputRef}`);
  const data = await apiJson(`/api/tasks/${state.task.task_id}/runs`, {
    method: "POST",
    body: formData,
  });
  state.task = data.task;
  renderSteps(state.task);
  renderAnalysis(data.analysis);
  renderTiming(data.timing);
  (data.logs || []).forEach((line) => appendLog(line));
  const first = data.outputs?.[0];
  if (first) {
    await selectArtifact(
      `step:${data.run_id}:${first.id}`,
      TOOL_LABELS[first.id] || first.id,
      first.audio_url,
      first.download_url,
      first.path,
      { skipAnalysisFetch: true },
    );
  }
  el.resultStamp.textContent = data.task.created_at;
  return data;
}

async function runCompose() {
  resetResult(false);
  const tools = checkedAvailableTools();
  if (!tools.length) {
    setStatus("请勾选至少一件就绪工具", true);
    return;
  }
  setBusy(true, "处理请求已提交");
  setBadge("status-busy", "处理中");
  try {
    await createTask();
    let workset = [{ ref: "input" }];
    for (const toolId of tools) {
      const next = [];
      for (const item of workset) {
        const data = await runToolOnRef(toolId, item.ref);
        data.outputs.forEach((output) => {
          next.push({ ref: `step:${data.run_id}:${output.id}` });
        });
      }
      workset = next;
    }
    setBadge("status-ready", "工具就绪");
    setStatus("处理完成");
  } catch (error) {
    appendLog(`处理失败: ${error.message}`);
    setStatus(error.message, true);
    setBadge("status-error", "处理失败");
  } finally {
    setBusy(false);
  }
}

async function runReprocess() {
  if (!state.task) {
    setStatus("请先导入并处理一次", true);
    return;
  }
  const toolId = el.reprocessTool.value;
  if (!toolId) {
    setStatus("请选择再处理工具", true);
    return;
  }
  setBusy(true, "再处理中");
  setBadge("status-busy", "处理中");
  try {
    await runToolOnRef(toolId, state.selectedRef);
    setBadge("status-ready", "工具就绪");
    setStatus("再处理完成");
  } catch (error) {
    appendLog(`再处理失败: ${error.message}`);
    setStatus(error.message, true);
    setBadge("status-error", "处理失败");
  } finally {
    setBusy(false);
  }
}

async function loadHealth() {
  const health = await apiJson("/api/health");
  state.tools = health.tools || [];
  const readyCount = availableTools().length;
  if (readyCount === state.tools.length && readyCount > 0) {
    setBadge("status-ready", "工具就绪");
  } else if (readyCount > 0) {
    setBadge("status-ready", `${readyCount}/${state.tools.length} 就绪`);
  } else {
    setBadge("status-missing", "工具未就绪");
    setStatus("没有可用工具权重", true);
  }
  el.uploadFile.accept = (health.supported_extensions || []).map((item) => `.${item}`).join(",");
  renderToolList();
}

async function loadSamples() {
  const payload = await apiJson("/api/samples");
  state.samples = payload.samples || [];
  el.sampleSelect.replaceChildren();
  state.samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.path;
    option.textContent = `${sample.name}（${sample.kind}）`;
    el.sampleSelect.appendChild(option);
  });
  if (!state.samples.length) {
    const option = document.createElement("option");
    option.textContent = "未找到示例音频";
    option.value = "";
    el.sampleSelect.appendChild(option);
    el.sampleSelect.disabled = true;
  } else {
    applySuggestedTools(selectedSample());
  }
  updateInputPreview();
  updateButtons();
}

function bindEvents() {
  el.sampleSourceButton.addEventListener("click", () => setSource("sample"));
  el.uploadSourceButton.addEventListener("click", () => setSource("upload"));
  el.sampleSelect.addEventListener("change", () => {
    resetResult(false);
    applySuggestedTools(selectedSample());
    updateInputPreview();
  });
  el.uploadFile.addEventListener("change", () => {
    resetResult(false);
    updateInputPreview();
    updateButtons();
  });
  el.waveformSeconds.addEventListener("input", () => {
    el.waveformSecondsText.textContent = `${el.waveformSeconds.value} 秒`;
  });
  el.runButton.addEventListener("click", runCompose);
  el.openTaskButton.addEventListener("click", () => {
    void openSelectedTask().catch((error) => {
      setStatus(error.message, true);
      appendLog(`打开任务失败: ${error.message}`);
    });
  });
  el.reprocessButton.addEventListener("click", runReprocess);
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
  appendLog("语音清晰工作台已启动");
  try {
    await loadHealth();
    await loadSamples();
    await loadTaskList();
  } catch (error) {
    setBadge("status-error", "检查失败");
    setStatus(error.message, true);
    appendLog(`初始化失败: ${error.message}`);
  }
}

init();
