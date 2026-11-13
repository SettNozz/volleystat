// Run with: node --test tests/test_team_names_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture(saved = {}, blocked = false) {
  class Element {
    constructor(side, text, job = 'match') {
      this.dataset = {teamSide: side, teamJob: job}; this.textContent = text;
      this.listeners = {}; this.style = {}; this.attributes = {};
    }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
  }
  class Input extends Element {
    constructor(side, text, job) { super(side, text, job); this.value = this.defaultValue = text; }
  }
  const near = new Input('b', 'Near team'), far = new Input('a', 'Far team');
  const option = new Element('b', 'Near side'), otherJob = new Element('b', 'Near side', 'other');
  const unknown = new Element('', 'Unattributed');
  const fields = [near, far, option, otherJob, unknown];
  const storage = new Map(Object.entries(saved));
  const listeners = {}, events = [];
  const context = vm.createContext({HTMLInputElement: Input,
    document: {querySelector: () => fields[0], querySelectorAll: () => fields},
    localStorage: {
      getItem: key => { if (blocked) throw Error('unavailable'); return storage.get(key) ?? null; },
      setItem: (key, value) => { if (blocked) throw Error('unavailable'); storage.set(key, value); },
    },
    CustomEvent: class { constructor(type) { this.type = type; } },
    addEventListener: (name, fn) => { listeners[name] = fn; },
    dispatchEvent: event => { events.push(event.type); },
  });
  context.window = context;
  vm.runInContext(readFileSync('viewer/static/team_names.js', 'utf8'), context);
  return {context, near, far, option, unknown, otherJob, storage, listeners, events};
}

test('typing updates every same-job label and dynamic lookup immediately', () => {
  const f = fixture();
  f.near.value = ' Falcons ';
  f.near.listeners.input();
  assert.equal(f.option.textContent, 'Falcons');
  assert.equal(f.near.value, ' Falcons '); // Editing never moves the caret or trims mid-input.
  assert.equal(f.context.TeamNames.label('team_b', 'Near'), 'Falcons');
  assert.equal(f.storage.get('["court-team-name","match","b"]'), 'Falcons');
  assert.equal(f.otherJob.textContent, 'Near side');
  assert.equal(f.far.value, 'Far team');
  assert.equal(f.unknown.textContent, 'Unattributed');
  assert.deepEqual(f.events, ['team-names-changed']);
});

test('reload and cross-tab updates preserve side aliases and reset original fallbacks', () => {
  const f = fixture({'["court-team-name","match","b"]': 'Falcons'});
  assert.equal(f.near.value, 'Falcons');
  assert.equal(f.option.textContent, 'Falcons');
  f.listeners.storage({key: '["court-team-name","match","b"]', newValue: 'Bears'});
  assert.equal(f.near.value, 'Bears');
  assert.equal(f.context.TeamNames.label('b', 'Near'), 'Bears');
  f.near.value = '';
  f.near.listeners.input();
  assert.equal(f.option.textContent, 'Near side');
  f.near.listeners.blur();
  assert.equal(f.near.value, 'Near team');
});

test('unavailable storage still permits live names; HTML templates escape names', () => {
  const f = fixture({}, true);
  f.near.value = '<img src=x onerror=alert(1)>';
  f.near.listeners.input();
  assert.equal(f.option.textContent, '<img src=x onerror=alert(1)>');
  assert.equal(f.context.TeamNames.html('b', 'Near'), '&lt;img src=x onerror=alert(1)&gt;');
  assert.equal(f.context.TeamNames.label(null, 'Unattributed'), 'Unattributed');
});

test('chart labels use names while unknown events retain their missing side', () => {
  const f = fixture({'["court-team-name","match","b"]': 'Falcons'});
  vm.runInContext(readFileSync('viewer/static/viewer_charts.js', 'utf8'), f.context);
  assert.equal(f.context.teamLabel('team_b'), 'Falcons');
  assert.equal(f.context.teamLabel(null), '—');
  assert.equal(f.context.teamLabel('team_a'), 'A');
  assert.equal(f.context.rallyContactsLabel({contacts_team_a: 2, contacts_team_b: 3}), 'A:2 Falcons:3');
});
