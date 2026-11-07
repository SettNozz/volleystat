const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  class Element {
    children = []; attributes = {}; dataset = {}; style = {};
    append(...values) { this.children.push(...values); }
    appendChild(value) { this.children.push(value); }
    replaceChildren(...values) { this.children = values; }
    setAttribute(name, value) { this.attributes[name] = value; }
  }
  const nodes = Object.fromEntries(['playerChips', 'playerContactSummary', 'playerContactStatus', 'playerContactList',
    'togPlayers', 'playerContactMethodBar', 'playerContactMethod', 'playerContactMethodQuality',
    'showPlayerContactCandidates'].map(id => [id, new Element()]));
  nodes.showPlayerContactCandidates.checked = true;
  const calls = [];
  const data = {source: {stats_sha256: 's', annotations_sha256: 'a'}, side: 'b', roster_ids: ['P1', 'P2'],
    method_id: 'learned', methods: {learned: {label: 'Learned side', team_lovo: {accuracy: .84}}, control: {label: 'Control'}},
    players: [{player_id: 'P1', total: 1, reviewed: 0, counts: {reception: 1, dig_pass: 0, attack: 0, block: 0, serve: 0}},
      {player_id: 'P2', total: 0, reviewed: 0, counts: {reception: 0, dig_pass: 0, attack: 0, block: 0, serve: 0}}],
    totals: {assigned: 1, unresolved: 1}, contacts: [
      {event_index: 0, frame: 30, time: 1, contact_frame: 32, contact_time: 32/30, type: 'reception', player_id: 'P1', status: 'assigned', label_source: 'derived', reason: 'learned'},
      {event_index: 1, frame: 60, time: 2, type: 'pass', player_id: null, proposed_player_id: 'P1', status: 'unresolved', label_source: 'derived', reason: 'ambiguous_players'},
    ]};
  const context = vm.createContext({document: {getElementById: id => nodes[id], createElement: () => new Element(), addEventListener() {}},
    formatTime: t => `${t.toFixed(2)}s`, seekAndPlay: f => calls.push(['seek', f]), redraw() {}, JOB_ID: 'clip',
    fetch: async (url, request) => {
      calls.push(['fetch', url, request]);
      return {ok: true, json: async () => ({...data, method_id: url.includes('method=control') ? 'control' : 'learned'})};
    }, data});
  vm.runInContext(readFileSync('viewer/static/player_contacts.js', 'utf8'), context);
  vm.runInContext('playerContactReport=data; renderPlayerContacts(); selectPlayer("P1")', context);
  return {context, data, nodes, calls};
}

test('uncertain candidates are separate from counts and jump uses proposed observed frame', () => {
  const {nodes, calls} = fixture();
  assert.equal(nodes.playerContactList.children.length, 2);
  assert.match(nodes.playerContactSummary.textContent, /1 assigned.*1 uncertain suggestions/);
  assert.match(nodes.playerContactList.children[1].className, /candidate/);
  nodes.playerContactList.children[0].children[0].onclick();
  assert.deepEqual(calls.at(-1), ['seek', 32]);
  nodes.showPlayerContactCandidates.checked = false;
});

test('method selector fetches chosen report and exposes team-only quality', async () => {
  const {context, nodes, calls} = fixture();
  assert.equal(nodes.playerContactMethodBar.style.display, '');
  assert.equal(nodes.playerContactMethod.children.length, 2);
  assert.match(nodes.playerContactMethodQuality.textContent, /84.0% accuracy.*Individual authors remain unverified/);
  await vm.runInContext('changePlayerContactMethod("control")', context);
  assert.ok(calls.some(c => c[0] === 'fetch' && c[1].endsWith('?method=control')));
  assert.equal(nodes.playerContactMethod.value, 'control');
});

test('human correction suppresses old conflicting model suggestion', () => {
  const {context, data, nodes} = fixture();
  data.contacts[1].player_id = 'P2';
  data.contacts[1].label_source = 'human';
  data.contacts[1].verified = true;
  vm.runInContext('renderPlayerContacts()', context);
  assert.equal(nodes.playerContactList.children.length, 1);
  assert.doesNotMatch(nodes.playerContactSummary.textContent, /uncertain suggestions/);
});

test('proposed actor focus is bound to exact source frame and respects human override', () => {
  const {context, data, nodes} = fixture();
  data.contacts[0].candidates = [{frame:32, pose_index:0, raw_track_id:7, player_id:'P1', side:'b', bbox:[1,2,3,4]}];
  nodes.playerContactList.children[0].children[0].onclick();
  const style = (frame, index=0) => vm.runInContext(`playerPoseStyle({team:'b',id:7,player_id:null},${index},${frame})`, context);
  assert.equal(style(32).width, 3);
  assert.equal(style(31).width, 1.5);
  assert.equal(style(32, 1).width, 1.5);
  data.contacts[0].label_source = 'human';
  assert.equal(style(32).width, 1.5);
});

test('report replacement clears stale candidate focus after method or human changes', () => {
  const {context, data, nodes} = fixture();
  data.contacts[0].candidates = [{frame:32, pose_index:0, raw_track_id:7, side:'b'}];
  nodes.playerContactList.children[0].children[0].onclick();
  const style = () => vm.runInContext("playerPoseStyle({team:'b',id:7,player_id:null},0,32)", context);
  assert.equal(style().width, 3);
  vm.runInContext("playerContactReport={...data, contacts:data.contacts.map(c => ({...c,label_source:'human'}))}; renderPlayerContacts()", context);
  assert.equal(style().width, 1.5);
  vm.runInContext("playerContactReport={...data, contacts:data.contacts.map(c => ({...c,candidates:[]}))}; renderPlayerContacts()", context);
  assert.equal(style().width, 1.5);
});
