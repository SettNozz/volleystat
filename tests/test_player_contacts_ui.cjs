// Run with: node --test tests/test_player_contacts_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  class Element {
    constructor() { this.children = []; this.attributes = {}; this.style = {}; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    dataset = {};
  }
  const nodes = Object.fromEntries(['playerChips', 'playerContactSummary', 'playerContactStatus',
    'playerContactList', 'togPlayers', 'playerContactMethodBar', 'playerContactMethod', 'playerContactMethodQuality'].map(id => [id, new Element()]));
  const calls = [];
  const context = vm.createContext({
    getComputedStyle: () => ({getPropertyValue: () => ''}),
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
  assert.equal(style({team: 'b', player_id: 'P1'}).width, 3.5);
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

test('selected roster member remains highlighted after a court-side switch', () => {
  const {context} = fixture();
  vm.runInContext("playerContactReport.team_id = 'observed_team'; selectedPlayerId = 'P1'", context);
  const style = vm.runInContext("playerPoseStyle({team:'a',team_id:'observed_team',player_id:'P1'})", context);
  assert.equal(style.width, 3.5);
});

test('team colour remains stable when the same roster crosses court sides', () => {
  const {context} = fixture();
  vm.runInContext("playerContactReport.team_id = 'observed_team'", context);
  const style = value => vm.runInContext(`playerPoseStyle(${JSON.stringify(value)})`, context);
  const before = style({team: 'b', team_id: 'observed_team', player_id: 'P1'});
  const after = style({team: 'a', team_id: 'observed_team', player_id: 'P1'});
  assert.equal(before.color, after.color);
  assert.notEqual(after.color, style({team: 'b', team_id: 'opponent_team'}).color);
});

test('model evidence highlights one observed pose despite a conflicting old roster ID or side', () => {
  const {context} = fixture();
  vm.runInContext(`selectedPlayerId = 'P1'; activePlayerContact = {
    player_id: 'P1', label_source: 'derived', candidates: [{frame: 30, pose_index: 1, raw_track_id: 99}]
  }`, context);
  assert.equal(vm.runInContext(`playerPoseStyle({id: 99, team: 'a', player_id: 'P7'}, 1, 30).width`, context), 3.5);
  assert.equal(vm.runInContext(`playerPoseStyle({id: 7, team: 'b', player_id: 'P1'}, 0, 30).width`, context), 1.5);
});

test('single promoted model remains visible and fallback never advertises the new model ran', () => {
  const {context, nodes} = fixture();
  vm.runInContext(`data.method_id = 'new_model'; data.attribution = {status: 'completed'}; renderPlayerContactMethods(data)`, context);
  assert.equal(nodes.playerContactMethodBar.style.display, '');
  assert.equal(nodes.playerContactMethod.disabled, true);
  assert.match(nodes.playerContactMethod.children[0].textContent, /soft HGB/);
  assert.match(nodes.playerContactMethodQuality.textContent, /unverified/);
  vm.runInContext(`data.attribution = {status: 'roster_unavailable'}; renderPlayerContactMethods(data)`, context);
  assert.match(nodes.playerContactMethodQuality.textContent, /geometric proposals/);
});
