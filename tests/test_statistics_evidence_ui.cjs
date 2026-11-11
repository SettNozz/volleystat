const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture({failure = false} = {}) {
  class Element {
    children = []; attributes = {}; listeners = {}; dataset = {};
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    setAttribute(key, value) { this.attributes[key] = value; }
    hasAttribute(key) { return key in this.attributes; }
    addEventListener(key, listener) { this.listeners[key] = listener; }
    querySelectorAll() { return this.children.flatMap(item => item.children); }
  }
  const ids = ['evidence-data', 'evidence-video', 'evidence-contacts', 'now', 'explanation',
    'watch-rally', 'previous', 'next', 'open-contact', 'category', 'metric', 'review-status'];
  const nodes = Object.fromEntries(ids.map(id => [id, new Element()]));
  const contacts = [
    {frame: 60, video_frame: 62, player_id: 'P1', type: 'serve', outcome: 'error',
      author_source: 'human', reason: 'derived_outcome', point_reviewed: true, start_frame: 15, end_frame: 90},
    {frame: 120, video_frame: 120, player_id: 'P1', type: 'serve', outcome: 'in_play',
      author_source: 'automatic', reason: 'derived_outcome', start_frame: null, end_frame: null},
  ];
  nodes['evidence-data'].textContent = JSON.stringify({contacts, fps: 30, job: 'test', kind: 'serve', metric: 'rate',
    source: {stats_sha256: 's', annotations_sha256: 'a'}, api: '/api/statistics/test', viewer: '/results/test',
    modes: {serve: ['Serves', [['total', 'Attempts'], ['rate', 'Error %']]]}});
  const video = nodes['evidence-video'];
  video.pause = () => { video.paused = true; };
  video.play = () => { video.paused = false; return Promise.resolve(); };
  const submit = new Element(), select = {value: 'b'};
  const form = {dataset: {attempt: 'service_attempts_fsm:0'}, querySelector: selector => selector === 'button' ? submit : select};
  const calls = [];
  const context = vm.createContext({document: {getElementById: id => nodes[id], createElement: () => new Element(),
    querySelectorAll: selector => selector === '.point-review' ? [form] : []},
    location: {reload: () => calls.push('reload')},
    fetch: async (url, options) => {
      calls.push([url, JSON.parse(options.body)]);
      return {ok: !failure, json: async () => failure ? {error: 'Analysis changed'} : {}};
    },
  });
  vm.runInContext(readFileSync('viewer/static/statistics_evidence.js', 'utf8'), context);
  return {nodes, video, form, select, submit, calls};
}

test('metric evidence preserves numerator/denominator and opens the matching contact frame', () => {
  const {nodes, video} = fixture();
  const buttons = nodes['evidence-contacts'].querySelectorAll();
  assert.match(buttons[0].children[0].textContent, /Human-confirmed author.*Numerator \+ denominator/);
  assert.match(buttons[1].children[0].textContent, /Automatic attribution.*Denominator only/);
  buttons[0].onclick();
  assert.equal(nodes['open-contact'].href, '/results/test?frame=60');
  assert.match(nodes.explanation.textContent, /contact outcome remains derived/);
  assert.equal(video.paused, false);
  nodes.next.onclick();
  assert.equal(nodes['watch-rally'].disabled, true);
  assert.equal(nodes.next.disabled, true);
});

test('rally playback uses matched bounds and stops at the end', () => {
  const {nodes, video} = fixture();
  nodes['watch-rally'].onclick();
  assert.equal(video.currentTime, .5);
  assert.equal(video.paused, false);
  video.currentTime = 3.1;
  video.listeners.timeupdate();
  assert.equal(video.paused, true);
});

test('point review sends source hashes and source-scoped target before refreshing', async () => {
  const {form, calls} = fixture();
  await form.onsubmit({preventDefault() {}});
  assert.deepEqual(calls[0], ['/api/statistics/test', {source: {stats_sha256: 's', annotations_sha256: 'a'},
    attempt_ref: 'service_attempts_fsm:0', reset: false, winner_side: 'b'}]);
  assert.equal(calls[1], 'reload');
});

test('stale source errors remain visible without refreshing or disabling retries', async () => {
  const {form, calls, nodes, submit} = fixture({failure: true});
  await form.onsubmit({preventDefault() {}});
  assert.equal(nodes['review-status'].textContent, 'Analysis changed');
  assert.equal(submit.disabled, false);
  assert.equal(calls.length, 1);
});
