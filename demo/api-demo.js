const baseInput = document.querySelector("#apiBaseUrl");

const getApiBaseUrl = () => baseInput?.value?.trim() || "http://127.0.0.1:5100/api";

const output = document.querySelector("#output");
const loadExamplesBtn = document.querySelector("#loadExamples");

const publishBtn = document.querySelector("#publishBtn");
const viewerUrlInput = document.querySelector("#viewerUrl");
const viewerUserInput = document.querySelector("#viewerUser");
const viewerPwdInput = document.querySelector("#viewerPwd");
const publishStatusArea = document.querySelector("#publishStatus");
const statusBadge = document.querySelector("#statusBadge");
const progressTrack = document.querySelector(".progress-track");
const progressBar = document.querySelector("#publishProgressBar");
const progressValue = document.querySelector("#publishProgressValue");

const renderJson = (data) => {
  if (!output) return;
  output.textContent = JSON.stringify(data, null, 2);
};

const renderError = (error) => {
  if (!output) return;
  output.textContent = `Error while calling API:\n${error.message ?? error}`;
};

const fetchExamples = async () => {
  if (!output) return;
  output.textContent = "Loading...";

  try {
    const response = await fetch(`${getApiBaseUrl()}/list_items?class=Currency`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    renderJson(payload);
  } catch (error) {
    renderError(error);
  }
};

if (loadExamplesBtn) {
  loadExamplesBtn.addEventListener("click", fetchExamples);
}

const setStatusBadge = (statusText) => {
  if (!statusBadge) {
    return;
  }

  statusBadge.dataset.status = statusText;
  statusBadge.textContent = statusText;
};

const setPublishStatus = (message) => {
  if (!publishStatusArea) {
    return;
  }

  publishStatusArea.value = message;
};

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const setProgress = (percent) => {
  if (!progressBar) {
    return;
  }

  const safePercent = Number.isFinite(percent) ? clamp(percent, 0, 100) : 0;
  progressBar.style.width = `${safePercent}%`;

  if (progressValue) {
    progressValue.textContent = `${safePercent}%`;
  }

  if (progressTrack) {
    progressTrack.setAttribute("aria-valuenow", String(safePercent));
  }
};

const extractProgress = (logs) => {
  if (!logs) {
    return null;
  }

  const lines = logs.split(/\r?\n/).reverse();
  for (const line of lines) {
    const match = line.match(/PROGRESS\s+(\d+)%/i);
    if (match) {
      return Number.parseInt(match[1], 10);
    }
  }

  return null;
};

const updateProgressFromLogs = (logs) => {
  const percent = extractProgress(logs);
  if (percent !== null) {
    setProgress(percent);
  }
};

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let isPolling = false;

const pollPublishStatus = async (jobId) => {
  if (!jobId) return;
  isPolling = true;

  try {
    while (isPolling) {
      const response = await fetch(`${getApiBaseUrl()}/publish-status?id=${encodeURIComponent(jobId)}`);
      if (!response.ok) {
        throw new Error(`Failed to query status (HTTP ${response.status})`);
      }

      const payload = await response.json();
      const { status, logs } = payload;

      setStatusBadge(status ?? "UNKNOWN");
      setPublishStatus(`${logs ?? ""}`);
      updateProgressFromLogs(logs);

      if (status !== "RUNNING") {
        isPolling = false;
        if (status === "SUCCESS") {
          setProgress(100);
        }
        break;
      }

      await wait(3000);
    }
  } catch (error) {
    setStatusBadge("ERROR");
    setPublishStatus(`Error while tracking publish job:\n${error.message ?? error}`);
    isPolling = false;
    setProgress(0);
  }
};

const triggerPublish = async () => {
  if (!publishBtn) return;

  const url = viewerUrlInput?.value?.trim();
  const user = viewerUserInput?.value?.trim();
  const pwd = viewerPwdInput?.value?.trim();

  const params = new URLSearchParams();
  if (url) params.append("url", url);
  if (user) params.append("user", user);
  if (pwd) params.append("pwd", pwd);

  const query = params.toString();
  const baseUrl = getApiBaseUrl();
  const endpoint = query ? `${baseUrl}/publish?${query}` : `${baseUrl}/publish`;

  isPolling = false;
  publishBtn.disabled = true;
  setStatusBadge("RUNNING");
  setPublishStatus("Starting publish job...");
  setProgress(0);

  try {
    const response = await fetch(endpoint, { method: "GET" });
    if (!response.ok) {
      throw new Error(`Failed to start publish job (HTTP ${response.status})`);
    }

    const payload = await response.json();
    const jobId = payload.jobId;

    if (!jobId) {
      throw new Error("Response does not include jobId.");
    }

    setPublishStatus(`Publish job started. Job ID: ${jobId}\nWaiting for status updates...`);
    setProgress(0);
    await pollPublishStatus(jobId);
  } catch (error) {
    setStatusBadge("ERROR");
    setPublishStatus(`Error while starting publish job:\n${error.message ?? error}`);
    setProgress(0);
  } finally {
    publishBtn.disabled = false;
  }
};

if (publishBtn) {
  publishBtn.addEventListener("click", triggerPublish);
}

window.apiDemo = {
  fetchExamples,
  triggerPublish,
  getApiBaseUrl,
};
