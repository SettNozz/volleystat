// Run with: node --test tests/test_viewer_highlights_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  class Element {
    children = [];
    attributes = {};
    dataset = {};
    value = '';
    checked = false;
    disabled = false;
    classes = new Set();
    classList = {contains: name => this.classes.has(name)};
    offsetTop = 0;
    clientHeight = 100;
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener() {}
    closest() { return this.interactive || false; }
  }
  const nodes = new Map();
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, new Element());
    return nodes.get(id);
  };
  element('page-video').classes.add('active');
  element('highlightBefore').value = '3';
  element('highlightAfter').value = '2';
  const listeners = {};
  const vid = {
    currentTime: 0, duration: 100, paused: true, seeking: false,
    pause() { this.paused = true; },
    addEventListener(name, handler) { listeners[name] = handler; },
  };
  let toggles = 0;
  const context = vm.createContext({
    JOB_ID: 'job', FPS: 30, URLSearchParams, selectedPlayerId: null, vid,
    localStorage: {getItem: () => null, setItem() {}},
    location: {search: ''}, setTrajectoryOverlay: {clip: {start: 1, stop: 2}},
    document: {getElementById: element, createElement: () => new Element(), querySelector: () => null},
    formatTime: seconds => String(seconds), redraw() {},
    togglePlay: () => { toggles++; vid.paused = !vid.paused; },
    seekAndPlay: frame => { vid.currentTime = frame / 30; vid.paused = false; },
  });
  const run = source => vm.runInContext(source, context);
  run(readFileSync('viewer/static/viewer_highlights.js', 'utf8'));
  run(`playerHighlights.moments = [
    {frame: 30, time: 1, type: 'attack', category: 'attack', label_source: 'human'},
    {frame: 300, time: 10, type: 'dig', category: 'dig_pass', label_source: 'derived', uncertain: true},
  ]; renderPlayerHighlights();`);
  return {context, element, vid, listeners, run, toggles: () => toggles, Element};
}

test('single highlight uses shared video and pauses at its tail; manual seek cancels its clip', () => {
  const {element, vid, run, listeners, context} = fixture();
  run('initPlayerHighlights(); playPlayerHighlight(0)');
  assert.equal(vid.currentTime, 0);
  assert.equal(context.setTrajectoryOverlay.clip, null);
  assert.equal(element('highlightList').children[0].attributes['aria-current'], 'true');
  vid.currentTime = 3;
  run('enforcePlayerHighlightClip()');
  assert.equal(vid.paused, true);
  run('playPlayerHighlight(1)');
  assert.equal(vid.currentTime, 7);
  vid.currentTime = 50;
  listeners.seeking();
  assert.equal(run('playerHighlights.clip'), null);
  run('enforcePlayerHighlightClip()');
  assert.equal(vid.paused, false);
});

test('play all advances in source order, retains pause/resume, stops after last moment', () => {
  const {element, vid, run} = fixture();
  run('togglePlayerHighlights()');
  assert.equal(run('playerHighlights.continuous'), true);
  vid.paused = true;
  vid.currentTime = 3;
  run('enforcePlayerHighlightClip()');
  assert.equal(run('playerHighlights.current'), 0);
  vid.paused = false;
  run('enforcePlayerHighlightClip()');
  assert.equal(run('playerHighlights.current'), 1);
  assert.equal(vid.currentTime, 7);
  assert.equal(element('highlightNext').disabled, true);
  vid.currentTime = 12;
  run('enforcePlayerHighlightClip()');
  assert.equal(vid.paused, true);
  assert.equal(run('playerHighlights.continuous'), false);
});

test('category filter and suggestions stay explicit; windows clamp to video bounds', () => {
  const {element, vid, run} = fixture();
  assert.match(element('highlightList').children[1].children[1].children[2].textContent, /Uncertain/);
  run("playerHighlights.filter = 'attack'; renderPlayerHighlights()");
  assert.equal(element('highlightList').children.length, 1);
  element('highlightBefore').value = '999';
  element('highlightAfter').value = '999';
  vid.duration = 2;
  run('playPlayerHighlight(0)');
  assert.equal(run('playerHighlights.clip.start'), 0);
  assert.equal(run('playerHighlights.clip.end'), 2);
});

test('natural video end completes highlight sequence even after the browser marks video paused', () => {
  const {vid, run, listeners} = fixture();
  run('initPlayerHighlights(); playPlayerHighlight(1, true)');
  vid.currentTime = 12;
  vid.ended = true;
  vid.paused = true;
  listeners.ended();
  assert.equal(run('playerHighlights.clip'), null);
  assert.equal(run('playerHighlights.continuous'), false);
});

test('space toggles only visible viewer and preserves interactive controls, repeats and modifiers', () => {
  const {element, run, context, toggles, Element} = fixture();
  const event = overrides => ({key: ' ', code: 'Space', target: new Element(),
    preventDefault() { this.prevented = true; }, ...overrides});
  context.event = event();
  run('handleViewerPlaybackKey(event)');
  assert.equal(toggles(), 1);
  assert.equal(context.event.prevented, true);
  context.event = event({repeat: true});
  run('handleViewerPlaybackKey(event)');
  assert.equal(toggles(), 1);
  assert.equal(context.event.prevented, true);
  for (const overrides of [{ctrlKey: true}, {altKey: true}, {metaKey: true},
    {isComposing: true}, {defaultPrevented: true},
    {target: Object.assign(new Element(), {interactive: true})},
    {target: Object.assign(new Element(), {isContentEditable: true})}]) {
    context.event = event(overrides);
    run('handleViewerPlaybackKey(event)');
    assert.equal(context.event.prevented, undefined);
  }
  context.document.querySelector = () => ({open: true});
  context.event = event();
  run('handleViewerPlaybackKey(event)');
  assert.equal(context.event.prevented, undefined);
  assert.equal(toggles(), 1);
  context.document.querySelector = () => null;
  element('page-video').classes.clear();
  context.event = event();
  run('handleViewerPlaybackKey(event)');
  assert.equal(context.event.prevented, undefined);
  assert.equal(toggles(), 1);
});

test('newer highlight requests win; failed requests clear stale clips and rows', async () => {
  const {context, element, vid, run} = fixture();
  const pending = [];
  context.fetch = url => new Promise(resolve => pending.push({url, resolve}));
  run('playPlayerHighlight(0)');
  const first = run("loadPlayerHighlights('P1')");
  element('highlightSuggestions').checked = true;
  const second = run("loadPlayerHighlights('P2')");
  assert.equal(vid.paused, true);
  assert.equal(element('highlightPlayAll').disabled, true);
  assert.ok(pending[1].url.includes('suggestions=1'));
  pending[1].resolve({ok: true, json: async () => ({player: 'P2', roster: ['P1', 'P2'],
    moments: [{frame: 60, time: 2, type: 'attack', category: 'attack'}], categories: {attack: 'Attacks'}})});
  await second;
  pending[0].resolve({ok: false, json: async () => ({error: 'Old failure'})});
  await first;
  assert.equal(run('playerHighlights.player'), 'P2');
  assert.equal(element('highlightList').children.length, 1);
  const third = run('loadPlayerHighlights()');
  pending[2].resolve({ok: false, json: async () => ({error: 'Analysis changed'})});
  await third;
  assert.equal(element('highlightStatus').textContent, 'Analysis changed');
  assert.equal(element('highlightList').children.length, 0);
  assert.equal(element('highlightPlayAll').disabled, true);
});

test('workspace buttons switch one shared panel and keep Video Analysis selected', () => {
  function node(dataset = {}) {
    const classes = new Set();
    return {dataset, hidden: false, attributes: {}, classes,
      classList: {add: name => classes.add(name), remove: name => classes.delete(name),
        toggle: (name, active) => active ? classes.add(name) : classes.delete(name)},
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; }};
  }
  const views = ['video', 'trajectories', 'highlights'];
  const panels = views.map(workspacePanel => node({workspacePanel}));
  const buttons = views.map(workspaceView => node({workspaceView}));
  const nav = ['video', 'charts', 'statistics'].map(page => node({page}));
  const pages = {'page-video': node(), 'page-charts': node()};
  let loads = 0, cancellations = 0;
  const context = vm.createContext({
    document: {getElementById: id => pages[id], querySelectorAll: selector => ({
      '.page': Object.values(pages), '.nav-item': nav,
      '[data-workspace-panel]': panels, '[data-workspace-view]': buttons,
    })[selector]},
    selectedPlayerId: 'P1', activePlayerContact: {}, setTrajectoryOverlay: {clip: {}},
    cancelPlayerHighlight: () => cancellations++, loadPlayerHighlights: () => loads++,
    toggleVideoSettings() {}, renderCharts() {}, resizeCanvas() {},
    location: {href: 'http://localhost/results/job'}, URL, JOB_ID: 'job', history: {replaceState() {}},
  });
  const source = readFileSync('viewer/static/viewer_page.js', 'utf8');
  vm.runInContext(source.slice(source.indexOf('function showPage('), source.indexOf('/* ── Shared chart')), context);
  for (const [index, view] of views.entries()) {
    vm.runInContext(`showPage('${view}')`, context);
    assert.deepEqual(panels.map(panel => panel.hidden), views.map((_, i) => i !== index));
    assert.deepEqual(buttons.map(button => button.attributes['aria-pressed']), views.map((_, i) => String(i === index)));
    assert.ok(pages['page-video'].classes.has('active'));
    assert.ok(nav[0].classes.has('active'));
  }
  assert.equal(loads, 1);
  assert.equal(cancellations, 2);
  assert.equal(context.selectedPlayerId, null);
  vm.runInContext("showPage('charts')", context);
  assert.ok(pages['page-charts'].classes.has('active'));
  assert.ok(!pages['page-video'].classes.has('active'));
  assert.ok(nav[1].classes.has('active'));
});
