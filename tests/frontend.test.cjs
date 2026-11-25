const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const script = readFileSync(join(__dirname, "../src/api/static/app.js"), "utf8");

async function setup({ ready = true, offline = false } = {}) {
  const elements = new Map();
  const calls = [];
  const reply = { status: 200, body: { emotion: "happiness", latency_ms: 12,
    face_box: { x: 1, y: 2, width: 100, height: 120 } } };
  function get(id) {
    if (!elements.has(id)) elements.set(id, {
      hidden: ["preview", "result", "error"].includes(id), disabled: false,
      textContent: "", dataset: {}, files: [], listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this[name] = value; },
      removeAttribute(name) { delete this[name]; },
      fire(name) { return this.listeners[name]({ preventDefault() {} }); },
    });
    return elements.get(id);
  }
  const context = {
    document: { getElementById: get }, FormData, AbortSignal,
    URL: { createObjectURL: () => "blob:test", revokeObjectURL() {} },
    fetch: async (url, options) => {
      calls.push({ url, options });
      if (offline || reply.networkError) throw new Error("offline");
      if (url === "/health") return { ok: true, json: async () => ({ model_ready: ready }) };
      return { ok: reply.status === 200, status: reply.status, json: async () => reply.body };
    },
  };
  vm.runInNewContext(script, context);
  await new Promise(setImmediate);
  function select(type = "image/png", bytes = "image data") {
    get("image-file").files = [new File([bytes], "face.png", { type })];
    get("image-file").fire("change");
  }
  return { get, calls, reply, select };
}

test("missing model allows preview but disables prediction", async () => {
  const ui = await setup({ ready: false });
  ui.select();
  assert.equal(ui.get("model-status").dataset.state, "missing");
  assert.equal(ui.get("preview").hidden, false);
  assert.equal(ui.get("predict").disabled, true);
  await ui.get("upload-form").fire("submit");
  assert.equal(ui.calls.length, 1);
});

test("upload uses FormData and displays the backend result", async () => {
  const ui = await setup();
  ui.select();
  assert.equal(ui.get("predict").disabled, false);
  const pending = ui.get("upload-form").fire("submit");
  assert.equal(ui.get("image-file").disabled, true);
  assert.equal(ui.get("predict").disabled, true);
  await pending;
  const request = ui.calls.at(-1);
  assert.equal(request.url, "/api/v1/predict");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.body.get("file").name, "face.png");
  assert.equal(ui.get("result").hidden, false);
  assert.equal(ui.get("emotion-label").textContent, "happiness");
  assert.equal(ui.get("latency").textContent, "12 ms");
  assert.equal(ui.get("face-box").textContent, "1, 2, 100, 120");
  assert.equal(ui.get("predict").disabled, false);
  ui.select();
  assert.equal(ui.get("result").hidden, true);
});

test("invalid type, oversize and unreadable preview prevent submission", async () => {
  const ui = await setup();
  ui.select("text/plain");
  assert.equal(ui.get("error").hidden, false);
  assert.equal(ui.get("predict").disabled, true);
  ui.select("image/png", new Uint8Array(5 * 1024 * 1024 + 1));
  assert.equal(ui.get("predict").disabled, true);
  ui.select();
  ui.get("preview").fire("error");
  assert.equal(ui.get("predict").disabled, true);
});

test("server rejection resets busy state and missing model disables retries", async () => {
  for (const status of [413, 422, 503]) {
    const ui = await setup();
    ui.select();
    ui.reply.status = status;
    ui.reply.body = { detail: "Expected exactly one face; found 0." };
    await ui.get("upload-form").fire("submit");
    assert.equal(ui.get("error").hidden, false);
    assert.equal(ui.get("result").hidden, true);
    assert.equal(ui.get("image-file").disabled, false);
    assert.equal(ui.get("predict").disabled, status === 503);
  }
});

test("network errors show an error and keep the interface usable", async () => {
  const offline = await setup({ offline: true });
  assert.equal(offline.get("model-status").dataset.state, "offline");
  const ui = await setup();
  ui.select();
  ui.reply.networkError = true;
  await ui.get("upload-form").fire("submit");
  assert.equal(ui.get("error").hidden, false);
  assert.equal(ui.get("image-file").disabled, false);
  assert.equal(ui.get("predict").disabled, false);
});

test("cropped face mode is explicit and changing mode clears the previous result", async () => {
  const ui = await setup();
  ui.select();
  await ui.get("upload-form").fire("submit");
  assert.equal(ui.calls.at(-1).url, "/api/v1/predict");
  ui.get("cropped-faces").checked = true;
  ui.get("cropped-faces").fire("change");
  assert.equal(ui.get("result").hidden, true);
  const pending = ui.get("upload-form").fire("submit");
  assert.equal(ui.get("cropped-faces").disabled, true);
  await pending;
  assert.equal(ui.calls.at(-1).url, "/api/v1/predict?cropped_faces=true");
  assert.equal(ui.get("cropped-faces").disabled, false);
  assert.equal(ui.get("result").hidden, false);
});
