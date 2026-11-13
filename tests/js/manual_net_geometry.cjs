// Run the shipped browser scripts against a recording canvas. No browser or
// model inference is needed; pytest supplies real calibration/export payloads.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function recordingContext() {
  const strokes = [];
  let points = [];
  const ctx = new Proxy({
    beginPath() { points = []; },
    moveTo(x, y) { points.push([x, y]); },
    lineTo(x, y) { points.push([x, y]); },
    stroke() { strokes.push({color: ctx.strokeStyle, points: structuredClone(points)}); },
  }, {get(target, key) { return key in target ? target[key] : () => {}; }});
  return {ctx, strokes};
}

const source = name => fs.readFileSync(path.join(__dirname, '../../viewer/static', name), 'utf8');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

async function preview() {
  const main = recordingContext(), loupe = recordingContext();
  const elements = new Map();
  function el(id) {
    if (!elements.has(id)) elements.set(id, {
      value: '', style: {}, listeners: {}, naturalWidth: 1600,
      addEventListener(event, handler) { this.listeners[event] = handler; },
      getContext() { return id === 'calibrationLoupe' ? loupe.ctx : main.ctx; },
      getBoundingClientRect() { return input.rect; },
      setPointerCapture() {}, replaceChildren() {}, add() {},
    });
    return elements.get(id);
  }
  el('videoPath').value = 'clip.webm';
  el('previewTime').value = 0;
  el('calibrationMode').value = 'net';
  el('netProfile').value = input.profile;
  const sandbox = {
    document: {getElementById: el}, window: {devicePixelRatio: 1.25},
    structuredClone, URLSearchParams,
    Image: class { constructor() { this.naturalWidth = input.data.frame_width; this.complete = true; } },
    Option: class {},
    fetch: async () => ({ok: true, headers: {get: () => 'application/json'}, json: async () => input.data}),
  };
  vm.createContext(sandbox);
  vm.runInContext(source('court_geometry.js'), sandbox);
  vm.runInContext(source('upload_calibration.js'), sandbox);
  await el('previewCalibration').listeners.click();
  const canvas = el('calibrationCanvas');
  const event = ([x, y]) => ({
    clientX: input.rect.left + x * input.rect.width / canvas.width,
    clientY: input.rect.top + y * input.rect.height / canvas.height,
    pointerId: 1, button: 0, pointerType: 'mouse',
  });
  canvas.listeners.pointerdown(event(input.data.net_tape[1]));
  main.strokes.length = 0;
  canvas.listeners.pointermove(event(input.target));
  canvas.listeners.lostpointercapture();
  const edit = vm.runInContext("uploadCalibration.analysisOptions('clip.webm').calibration", sandbox);
  const snapshot = structuredClone(edit);
  // A caller must not be able to move an endpoint through a returned reference.
  edit.net_tape[0][0] -= 100;
  const after = vm.runInContext("uploadCalibration.analysisOptions('clip.webm').calibration", sandbox);
  return {edit: snapshot, after, strokes: main.strokes};
}

function overlay() {
  const {ctx, strokes} = recordingContext();
  const sandbox = {
    ctx, canvas: input.canvas,
    vid: {videoWidth: input.data.frame_width, videoHeight: input.data.frame_height},
    window: {_annData: input.data}, annotations: {0: {f: 0}}, currentFrameIdx: 0,
    courtEditor: {active: false},
    renderServeScore() {}, renderAttackHeight() {}, updateActiveEvent() {}, drawSetTrajectories() {},
    document: {getElementById: id => ({checked: id === 'togNet', value: 'active'})},
  };
  vm.runInNewContext(source('viewer_overlay.js') + '\nredraw();', sandbox);
  return {strokes};
}

(async () => {
  const result = input.mode === 'preview' ? await preview() : overlay();
  process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
