// Run with: node --test tests/test_player_profiles_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  class Element {
    children = []; dataset = {}; attributes = {}; listeners = {}; open = false;
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name] || null; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
    querySelector(selector) {
      const key = selector.slice(6, -1).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const visit = node => key in node.dataset ? node : node.children.map(visit).find(Boolean);
      return visit(this);
    }
  }
  const nodes = Object.fromEntries(['roster-picker', 'roster-form', 'roster-editor', 'photo-picker',
    'photo-preview', 'photo-status', 'save-photo', 'next-photo', 'prev-photo', 'photo-title'].map(id => [id, new Element()]));
  const storage = new Map();
  const context = vm.createContext({
    document: {getElementById: id => nodes[id] || null, querySelectorAll: () => [], querySelector: () => null,
      createElement: () => new Element(), addEventListener() {}},
    localStorage: {getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value)},
    window: {dispatchEvent() {}, addEventListener() {}}, CustomEvent: class {},
  });
  const run = source => vm.runInContext(source, context);
  run(readFileSync('viewer/static/player_profiles.js', 'utf8'));
  const config = {jobId: 'job', sourceVersion: 'ann-v1', roster: ['P1', 'P7'], portraits: {P1: 3, P7: 1}, positions: {setter: 'Setter'}};
  context.config = config;
  return {nodes, storage, context, run};
}

test('profiles keep existing browser keys; photos remain bound to annotation source', () => {
  const {storage, run} = fixture();
  storage.set(JSON.stringify(['player-profiles', 'job']), JSON.stringify({P1: {name: 'Ann', number: '7', position: 'setter'}}));
  storage.set(JSON.stringify(['player-photo', 'job', 'ann-v1', 'P1']), '2');
  run('PlayerProfiles.configure(config)');
  assert.equal(run("PlayerProfiles.label('P1')"), 'Ann');
  assert.equal(run("PlayerProfiles.photos.P1"), 2);
  run("PlayerProfiles.configure({...config, sourceVersion: 'ann-v2'})");
  assert.equal(run("PlayerProfiles.label('P1')"), 'Ann');
  assert.equal(run('PlayerProfiles.photos.P1'), 0);
  assert.equal(run("PlayerProfiles.label('P9')"), 'P9');
});

test('photo changes require Edit players and cancel discards draft choices', () => {
  const {nodes, storage, run} = fixture();
  run("PlayerProfiles.configure(config); PlayerProfiles.openPhoto('P1')");
  assert.equal(nodes['photo-picker'].open, false);
  run("PlayerProfiles.openEditor(); PlayerProfiles.openPhoto('P1'); PlayerProfiles.stepPhoto(1)");
  assert.equal(nodes['roster-picker'].open, true);
  assert.equal(nodes['photo-picker'].open, true);
  nodes['save-photo'].onclick();
  assert.equal(run('PlayerProfiles.draftPhotos.P1'), 1);
  assert.equal(storage.size, 0);
  nodes['roster-picker'].close();
  assert.equal(run('PlayerProfiles.draftPhotos'), null);
  assert.equal(run('PlayerProfiles.photos.P1'), 0);
});

test('save commits names, jersey, position and draft photo together for statistics and highlights', () => {
  const {nodes, storage, run} = fixture();
  run("PlayerProfiles.configure(config); PlayerProfiles.openEditor(); PlayerProfiles.draftPhotos.P1 = 2");
  const row = nodes['roster-editor'].children[0];
  row.querySelector('[data-player-name]').value = ' Ann ';
  row.querySelector('[data-player-number]').value = ' 7 ';
  row.querySelector('[data-player-pos]').value = 'setter';
  nodes['roster-form'].listeners.submit({submitter: {value: 'save'}});
  const profile = JSON.parse(storage.get(JSON.stringify(['player-profiles', 'job']))).P1;
  assert.deepEqual(profile, {name: 'Ann', number: '7', position: 'setter'});
  assert.equal(storage.get(JSON.stringify(['player-photo', 'job', 'ann-v1', 'P1'])), '2');
  assert.equal(run("PlayerProfiles.label('P1')"), 'Ann');
});

test('lineup follows inclusive saved intervals; unknown intervals retain roster membership', () => {
  const {run} = fixture();
  run(readFileSync('viewer/static/viewer_roster.js', 'utf8'));
  const lineup = JSON.stringify({P1: {intervals: [[0, 29], [90, 120]]}, P7: {intervals: [[30, 89]]}});
  const at = frame => JSON.parse(run(`JSON.stringify(rosterPlayersAtFrame(['P1','P2','P7'], ${lineup}, ${frame}))`));
  assert.deepEqual(at(29), {playing: ['P1', 'P2'], bench: ['P7']});
  assert.deepEqual(at(30), {playing: ['P2', 'P7'], bench: ['P1']});
  assert.deepEqual(at(90), {playing: ['P1', 'P2'], bench: ['P7']});
});
