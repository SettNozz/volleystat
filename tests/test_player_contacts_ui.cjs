// Run with: node --test tests/test_player_contacts_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  class Element {
    constructor() { this.children = []; this.attributes = {}; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    dataset = {};
  }
  const nodes = Object.fromEntries(['playerChips', 'playerContactSummary', 'playerContactStatus',
    'playerContactList', 'togPlayers'].map(id => [id, new Element()]));
  const calls = [];
  const context = vm.createContext({
    document: {getElementById: id => nodes[id], createElement: () => new Element(), addEventListener() {}},
    formatTime: t => `${t}s`, seekAndPlay: f => calls.push(['seek', f]),
    redraw: () => calls.push(['redraw']), JOB_ID: 'demo',
    fetch: async (_, request) => {
      const body = JSON.parse(request.body);
      calls.push(['review', body]);
      return {ok: true, json: async () => ({...data, contacts: [], players: data.players})};
    },
  });
  const data = {
    source: {stats_sha256: 's', annotations_sha256: 'a'}, side: 'b', roster_ids: ['P1', 'P7'],
    totals: {assigned: 1, unresolved: 1},
    players: [{player_id: 'P1', total: 1, reviewed: 0, counts: {reception: 1, dig_pass: 0, attack: 0, block: 0, serve: 0}}],
    contacts: [
      {event_index: 0, frame: 30, time: 1, type: 'reception', player_id: 'P1', status: 'assigned', reason: 'body_proximity', label_source: 'derived'},
      {event_index: 1, frame: 90, time: 3, type: 'block', player_id: null, status: 'unresolved', reason: 'ambiguous_players', label_source: 'derived'},
    ],
  };
  context.data = data;
  vm.runInContext(readFileSync('viewer/static/player_contacts.js', 'utf8'), context);
  vm.runInContext('playerContactReport = data; renderPlayerContacts()', context);
  return {context, nodes, calls};
}

test('selection filters list, highlights only selected skeleton, preserves P7', () => {
  const {context, nodes, calls} = fixture();
  assert.equal(nodes.playerChips.children.length, 3);
  assert.equal(nodes.playerChips.children[2].textContent, 'P7');
  assert.equal(nodes.playerContactList.children.length, 2);
  nodes.playerChips.children[1].onclick();
  assert.equal(nodes.togPlayers.checked, true);
  assert.equal(nodes.playerContactList.children.length, 1);
  assert.match(nodes.playerContactSummary.textContent, /Reception 1/);
  const style = value => vm.runInContext(`playerPoseStyle(${JSON.stringify(value)})`, context);
  assert.equal(style({team: 'b', player_id: 'P1'}).width, 3);
  assert.equal(style({team: 'b', player_id: 'P7'}).width, 1.5);
  assert.equal(style({team: 'a', player_id: 'P1'}).width, 1.5);
  assert.equal(style({team: 'b', player_id: null}).width, 1.5);
  nodes.playerContactList.children[0].children[0].onclick();
  assert.deepEqual(calls.at(-1), ['seek', 30]);
  nodes.playerChips.children[0].onclick();
  assert.equal(nodes.playerContactList.children.length, 2);
});

test('correction posts source-bound event index and refreshes rows', async () => {
  const {nodes, calls} = fixture();
  const select = nodes.playerContactList.children[1].children[1];
  select.value = 'P7';
  await select.onchange();
  const body = calls.find(c => c[0] === 'review')[1];
  assert.equal(body.event_index, 1);
  assert.equal(body.player_id, 'P7');
  assert.deepEqual(body.source, {stats_sha256: 's', annotations_sha256: 'a'});
  assert.match(nodes.playerContactStatus.textContent, /Attribution saved/);
});
