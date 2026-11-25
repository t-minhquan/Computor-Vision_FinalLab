"use strict";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("image-file");
const croppedInput = document.getElementById("cropped-faces");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const predictButton = document.getElementById("predict");
const refreshButton = document.getElementById("refresh");
const statusLabel = document.getElementById("model-status");
const modelHint = document.getElementById("model-hint");
const resultPanel = document.getElementById("result-panel");
const result = document.getElementById("result");
const empty = document.getElementById("result-empty");
const error = document.getElementById("error");
const labels = { neutral: "Bình thường", happiness: "Vui vẻ", sadness: "Buồn bã", surprise: "Ngạc nhiên", anger: "Giận dữ", disgust: "Ghê tởm", fear: "Sợ hãi" };
let selectedFile = null;
let previewUrl = null;
let modelReady = false;
let busy = false;

function updateControls() {
  predictButton.disabled = busy || !selectedFile || !modelReady;
  fileInput.disabled = busy;
  croppedInput.disabled = busy;
  refreshButton.disabled = busy;
  predictButton.textContent = busy ? "Đang nhận diện…" : "Nhận diện cảm xúc";
  resultPanel.setAttribute("aria-busy", String(busy));
}

function showError(message) {
  error.textContent = message;
  error.hidden = false;
}

async function checkHealth() {
  refreshButton.disabled = true;
  modelReady = false;
  statusLabel.textContent = "Đang kiểm tra backend…";
  statusLabel.dataset.state = "checking";
  predictButton.disabled = true;
  try {
    const response = await fetch("/health", { cache: "no-store", signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error("health");
    const health = await response.json();
    modelReady = health.model_ready === true;
    statusLabel.textContent = modelReady ? "Model đã sẵn sàng" : "Backend hoạt động · Chưa có model";
    statusLabel.dataset.state = modelReady ? "ready" : "missing";
    modelHint.textContent = modelReady ? "Bạn có thể chọn ảnh và bắt đầu nhận diện." : "Cần huấn luyện và nạp model, sau đó khởi động lại backend để nhận diện.";
  } catch {
    statusLabel.textContent = "Không kết nối được backend";
    statusLabel.dataset.state = "offline";
    modelHint.textContent = "Kiểm tra backend đang chạy rồi nhấn Kiểm tra lại.";
  } finally {
    updateControls();
  }
}

fileInput.addEventListener("change", () => {
  selectedFile = null;
  result.hidden = true;
  empty.hidden = false;
  error.hidden = true;
  preview.hidden = true;
  placeholder.hidden = false;
  preview.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  const file = fileInput.files[0];
  if (file) {
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      showError("Hãy chọn ảnh JPEG hoặc PNG.");
    } else if (file.size === 0 || file.size > 5 * 1024 * 1024) {
      showError("Ảnh phải có dữ liệu và không vượt quá 5 MiB.");
    } else {
      selectedFile = file;
      previewUrl = URL.createObjectURL(file);
      preview.src = previewUrl;
      preview.hidden = false;
      placeholder.hidden = true;
    }
  }
  updateControls();
});

preview.addEventListener("error", () => {
  selectedFile = null;
  preview.hidden = true;
  placeholder.hidden = false;
  showError("Không đọc được ảnh. Hãy chọn một ảnh JPEG hoặc PNG khác.");
  updateControls();
});

croppedInput.addEventListener("change", () => {
  result.hidden = true;
  empty.hidden = false;
  error.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !selectedFile || !modelReady) return;
  busy = true;
  error.hidden = true;
  result.hidden = true;
  empty.hidden = true;
  updateControls();
  const data = new FormData();
  data.append("file", selectedFile);
  try {
    const endpoint = croppedInput.checked ? "/api/v1/predict?cropped_faces=true" : "/api/v1/predict";
    const response = await fetch(endpoint, { method: "POST", body: data, signal: AbortSignal.timeout(30000) });
    const body = await response.json();
    if (!response.ok) {
      let message = "Không nhận diện được ảnh. Vui lòng thử lại.";
      if (response.status === 503) {
        modelReady = false;
        statusLabel.textContent = "Backend hoạt động · Chưa có model";
        statusLabel.dataset.state = "missing";
        modelHint.textContent = "Cần huấn luyện và nạp model, sau đó khởi động lại backend để nhận diện.";
        message = "Chưa có model đã huấn luyện để nhận diện cảm xúc.";
      } else if (response.status === 413) {
        message = "Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn 5 MiB.";
      } else if (response.status === 422) {
        const count = typeof body.detail === "string" && body.detail.match(/found (\d+)/);
        message = count ? `Ảnh có ${count[1]} khuôn mặt được phát hiện. Hãy chọn ảnh có đúng một khuôn mặt rõ nét.` : "Ảnh không hợp lệ. Dùng JPEG/PNG tối đa 12 megapixel, có đúng một khuôn mặt.";
      }
      showError(message);
      empty.hidden = false;
      return;
    }
    document.getElementById("emotion").textContent = labels[body.emotion] || body.emotion;
    document.getElementById("emotion-label").textContent = body.emotion;
    document.getElementById("latency").textContent = `${body.latency_ms} ms`;
    const box = body.face_box;
    document.getElementById("face-box").textContent = `${box.x}, ${box.y}, ${box.width}, ${box.height}`;
    result.hidden = false;
  } catch {
    showError("Không nhận được phản hồi. Kiểm tra kết nối hoặc thử lại với ảnh nhỏ hơn.");
    empty.hidden = false;
  } finally {
    busy = false;
    updateControls();
  }
});

refreshButton.addEventListener("click", checkHealth);
checkHealth();
