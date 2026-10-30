// Run with: node --test tests/test_pose_completion_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  const toggle = {checked: true, disabled: false};
  const status = {};
  const calls = [];
  const ctx = {save() {}, restore() {}, setLineDash: v => calls.push(['dash', v]),
    beginPath() {}, moveTo() {}, lineTo() {},
    stroke() { calls.push(['line', this.strokeStyle]); },
    strokeRect() { calls.push(['joint', this.strokeStyle]); },
    fillText(text) { calls.push(['label', text]); }};
  const context = vm.createContext({document: {getElementById: id => id === 'togPoseEstimates' ? toggle : status},
    selectedPlayerId: 'P1', ctx});
  vm.runInContext(readFileSync('viewer/static/pose_completion.js', 'utf8'), context);
  const ann = {f: 5, pose_estimates: [{source: 'temporal_pose_completion_v1', player_id: 'P1',
    observed_pose_index: null, estimated_joints: [5, 7],
    keypoints: Array.from({length: 17}, () => [10, 20])}]};
  context.ann = ann;
  return {context, calls, toggle, status, ann, draw: () =>
    vm.runInContext('drawPoseEstimates(ctx, ann, 5, x => x, y => y, 1)', context)};
}

test('derived skeleton uses dashed cyan and separate estimated label', () => {
  const {calls, draw} = fixture();
  draw();
  assert.ok(calls.some(c => c[0] === 'dash' && c[1].length === 2));
  assert.ok(calls.some(c => c[0] === 'joint' && c[1] === '#00b9e8'));
  assert.deepEqual(calls.find(c => c[0] === 'label'), ['label', 'P1 · estimated']);
});

test('disabled, stale or unknown-schema estimates never draw; other players gray', () => {
  for (const kind of ['disabled', 'unchecked', 'stale', 'schema']) {
    const {toggle, ann, calls, draw} = fixture();
    if (kind === 'disabled') toggle.disabled = true;
    if (kind === 'unchecked') toggle.checked = false;
    if (kind === 'stale') ann.f = 4;
    if (kind === 'schema') ann.pose_estimates[0].source = 'unknown';
    draw();
    assert.ok(!calls.some(c => c[0] === 'joint'));
  }
  const {ann, calls, draw} = fixture();
  ann.pose_estimates[0].player_id = 'P2';
  draw();
  assert.ok(calls.some(c => c[0] === 'joint' && c[1].includes('150,170,190')));
});

test('legacy results disable toggle; completed results explain source and count limits', () => {
  const {context, toggle, status} = fixture();
  vm.runInContext('initPoseEstimates({})', context);
  assert.equal(toggle.disabled, true);
  vm.runInContext(`initPoseEstimates({pose_completion: {schema:'temporal_pose_completion_v1',
    counts:{missing_person_frames:3,partial_person_frames:5}}})`, context);
  assert.equal(toggle.disabled, false);
  assert.match(status.textContent, /3 missing \+ 5 partial/);
  assert.match(status.textContent, /contact counts still use observed poses/);
});
