// Run with: node --test tests/test_viewer_events_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture(events, reducedMotion = false) {
  class Element {
    children = [];
    hidden = false;
    dataset = {};
    attributes = {};
    style = {setProperty() {}};
    classes = new Set();
    classList = {toggle: (name, value) => value ? this.classes.add(name) : this.classes.delete(name)};
    lastChild = {};
    animations = 0;
    scrollTop = 0;
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    removeAttribute(name) { delete this.attributes[name]; }
    getBoundingClientRect() { return {top: 0, bottom: 100}; }
    getAnimations() { return []; }
    animate() { this.animations++; }
  }
  const nodes = new Map();
  const element = id => {
    if (!nodes.has(id)) nodes.set(id, new Element());
    return nodes.get(id);
  };
  element('togEvents').checked = true;
  const seeks = [];
  const context = vm.createContext({
    EVENTS: events, FPS: 30, RALLIES: [], currentFrameIdx: 0,
    eventTeams: events.map(event => event.team), vid: {paused: true, seeking: false},
    document: {getElementById: element, createElement: () => new Element()},
    window: {matchMedia: () => ({matches: reducedMotion})},
    evColor: () => '#3b82f6', formatTime: seconds => seconds.toFixed(1),
    seekAndPlay: frame => seeks.push(frame),
  });
  const run = source => vm.runInContext(source, context);
  run(readFileSync('viewer/static/viewer_events.js', 'utf8'));
  run('renderEventList()');
  return {context, element, run, seeks, rows: () => run('contactFeed.rows.map(item => item.row)')};
}

test('history follows authoritative frames in order and keeps original contact indexes', () => {
  const events = [
    {frame: 90, time: 0, type: 'attack', team: null},
    {frame: 30, time: 99, type: 'serve', team: 'team_b', inferred: true},
    {frame: 60, time: 2, type: 'reception', team: null},
  ];
  const {element, run, rows, seeks} = fixture(events);
  assert.ok(rows().every(row => row.hidden));
  run('updateActiveEvent(29)');
  assert.ok(element('videoEventBadge').hidden);
  run('updateActiveEvent(30)');
  assert.equal(rows()[0].id, 'ev1');
  assert.equal(rows()[0].hidden, false);
  assert.equal(rows()[1].hidden, true);
  assert.equal(element('videoEventMeta').textContent, 'Near side · Inferred');
  run('updateActiveEvent(60)');
  assert.equal(rows()[1].hidden, false);
  assert.equal(element('videoEventMeta').textContent, 'Unattributed');
  assert.equal(rows()[1].attributes['aria-current'], 'true');
  rows()[1].onclick();
  assert.deepEqual(seeks, [60]);
  run('updateActiveEvent(0)');
  assert.ok(rows().every(row => row.hidden));
  assert.ok(element('videoEventBadge').hidden);
  assert.equal(events[0].frame, 90);
});

test('full log shows future contacts without making them active or displaying their badge', () => {
  const {element, run, rows} = fixture([{frame: 300, type: 'attack'}]);
  run('toggleEventLog()');
  assert.equal(rows()[0].hidden, false);
  assert.ok(rows()[0].classes.has('future'));
  assert.equal(rows()[0].attributes['aria-current'], undefined);
  assert.ok(element('videoEventBadge').hidden);
  run('toggleEventLog()');
  assert.equal(rows()[0].hidden, true);
});

test('playback animates new contacts; seeks and reduced motion do not animate', () => {
  for (const reducedMotion of [false, true]) {
    const {context, element, run, rows} = fixture([
      {frame: 30, type: 'serve'}, {frame: 300, type: 'attack'},
    ], reducedMotion);
    context.vid.paused = false;
    run('updateActiveEvent(29); updateActiveEvent(30)');
    assert.equal(rows()[0].animations, reducedMotion ? 0 : 1);
    assert.equal(element('videoEventBadge').animations, reducedMotion ? 0 : 1);
    context.vid.seeking = true;
    run('updateActiveEvent(300)');
    assert.equal(rows()[1].animations, 0);
    context.vid.seeking = false;
    element('togEvents').checked = false;
    run('updateActiveEvent(300)');
    assert.ok(element('videoEventBadge').hidden);
    assert.equal(rows()[1].hidden, false);
  }
});

test('legacy timestamps, simultaneous contacts and empty results remain usable', () => {
  const {element, run, rows} = fixture([
    {time: 1, type: 'serve'}, {frame: 30, type: 'reception'}, {type: 'invalid'},
  ]);
  run('updateActiveEvent(30)');
  assert.equal(rows().length, 2);
  assert.ok(rows().every(row => !row.hidden));
  assert.equal(rows()[1].attributes['aria-current'], 'true');
  run('updateActiveEvent(105)');
  assert.ok(element('videoEventBadge').hidden);
  const empty = fixture([]);
  empty.run('toggleEventLog()');
  assert.equal(empty.element('eventFeedEmpty').textContent, 'No contact events in this result.');
});
