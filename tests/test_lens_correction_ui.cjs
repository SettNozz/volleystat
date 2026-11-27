// Exercise the shipped upload UI, including profile changes during preview.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const elements = new Map();
const context = new Proxy({}, { get(target, key) { return target[key] ?? (() => {}); } });
function el(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '', checked: false, hidden: false, style: {}, listeners: {},
    naturalWidth: 640,
    addEventListener(event, handler) { this.listeners[event] = handler; },
    getContext() { return context; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; },
    replaceChildren() {}, add() {}, setPointerCapture() {},
  });
  return elements.get(id);
}
Object.entries({ videoPath: 'wide.mp4', previewTime: '0', calibrationMode: 'corners',
  netProfile: 'mixed', lensFov: '130', lensBalance: '0.5' }).forEach(([id, value]) => { el(id).value = value; });
const requests = [], images = [];
const sandbox = {
  document: { getElementById: el }, window: { devicePixelRatio: 1 },
  structuredClone, URLSearchParams, Option: class {},
  Image: class {
    constructor() { this.naturalWidth = 640; this.complete = true; images.push(this); }
  },
  fetch: async (url, options) => {
    const body = JSON.parse(options.body);
    requests.push(body);
    const lens = body.lens_correction;
    return { ok: true, headers: { get: () => 'application/json' }, json: async () => ({
      image: 'data:image/jpeg;base64,preview', frame_width: 640, frame_height: 360,
      last_second: 100, setups: [], court_corners: [[100, 100], [540, 100], [600, 340], [40, 340]],
      net_tape: [[100, 80], [540, 80]],
      // Flask sorts JSON keys; field order must not invalidate a matching profile.
      lens_correction: lens ? { balance: lens.balance, enabled: true, horizontal_fov_deg: lens.horizontal_fov_deg } : null,
    }) };
  },
};
vm.createContext(sandbox);
for (const name of ['court_geometry.js', 'upload_calibration.js']) {
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../viewer/static', name), 'utf8'), sandbox);
}
const options = () => JSON.parse(JSON.stringify(vm.runInContext("uploadCalibration.analysisOptions(document.getElementById('videoPath').value)", sandbox)));
const settle = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  el('lensCorrectionEnabled').checked = true;
  el('lensCorrectionEnabled').listeners.change();
  assert.throws(options, /Wait for/);
  await settle();
  assert.deepEqual(requests.at(-1).lens_correction, { enabled: true, horizontal_fov_deg: 130, balance: 0.5 });
  assert.equal(options().calibration.lens_correction.horizontal_fov_deg, 130);
  assert.deepEqual(JSON.parse(new URLSearchParams(images.at(-1).src.split('?')[1]).get('lens_correction')),
    { enabled: true, horizontal_fov_deg: 130, balance: 0.5 });

  el('lensFov').value = '112.6';
  el('previewTime').value = '20';
  el('lensFov').listeners.change();
  assert.throws(options, /Wait for/);
  await settle();
  assert.equal(options().lens_correction.horizontal_fov_deg, 112.6);
  assert.equal(options().calibration.lens_correction.horizontal_fov_deg, 112.6);
  assert.equal(requests.at(-1).seconds, 20);

  el('videoPath').value = 'another.mp4';
  el('videoPath').listeners.input();
  assert.throws(options, /Preview the lens correction/);

  el('cancelCalibration').listeners.click();
  assert.equal(el('lensCorrectionEnabled').checked, false);
  assert.deepEqual(options(), {});
  console.log('Lens correction UI passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
