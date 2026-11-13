// Run with: node --test tests/test_set_trajectories_ui.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function load() {
  const context = vm.createContext({
    rallyStartFrame: rally => rally.start_frame,
    lifecycleFrame: (rally, key, fallback) => fallback,
    distanceToSegment(point, a, b) {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const t = Math.max(0, Math.min(1,
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)));
      return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
    },
  });
  vm.runInContext(readFileSync('viewer/static/viewer_set_trajectories.js', 'utf8'), context);
  return context;
}

const EVENTS = [
  {frame: 10, type: 'serve'},
  {frame: 40, type: 'reception'},
  {frame: 60, type: 'setting_pass'},
  {frame: 90, type: 'attack'},
  {frame: 200, type: 'pass'},
  {frame: 230, type: 'attack'},
  {frame: 300, type: 'setting_pass'},
  {frame: 330, type: 'attack'},
];
const TEAMS = ['team_a', 'team_b', 'team_b', 'team_b', 'team_b', 'team_b', 'team_b', 'team_a'];
const RALLIES = [{start_frame: 0, end_frame: 400}];

test('pairs keep the source event indexes of the set and the attack', () => {
  const pairs = load().setAttackPairs(EVENTS, TEAMS, RALLIES);
  // The last pair crosses court sides, so it is not a set-to-attack pair.
  assert.deepEqual(JSON.parse(JSON.stringify(pairs)), [
    {start: 60, end: 90, setIndex: 2, attackIndex: 3, team: 'team_b'},
    {start: 200, end: 230, setIndex: 4, attackIndex: 5, team: 'team_b'},
  ]);
});

test('setters come only from assigned contacts that match the saved event', () => {
  const report = {contacts: [
    {event_index: 2, frame: 60, type: 'setting_pass', status: 'assigned', player_id: 'P3'},
    {event_index: 4, frame: 200, type: 'pass', status: 'unresolved', player_id: null, proposed_player_id: 'P5'},
    {event_index: 6, frame: 301, type: 'setting_pass', status: 'assigned', player_id: 'P3'},
    {event_index: 7, frame: 330, type: 'attack', status: 'other_side', player_id: null},
  ]};
  const context = load();
  assert.deepEqual(Array.from(context.setAttackSetterIds(report, EVENTS), ([index, id]) => `${index}:${id}`), ['2:P3']);
  assert.equal(context.setAttackSetterIds(null, EVENTS).size, 0);
});

test('the main setter is the most frequent identified one and a tie has none', () => {
  const context = load();
  const pairs = ['P3', 'P3', 'P5', null, 'P3'].map(setter => ({setter}));
  const {ranked, main} = context.setAttackMainSetter(pairs);
  assert.equal(main, 'P3');
  assert.deepEqual(JSON.parse(JSON.stringify(ranked)), [['P3', 3], ['P5', 1]]);
  assert.equal(context.setAttackMainSetter([{setter: 'P3'}, {setter: 'P5'}]).main, null);
  assert.equal(context.setAttackMainSetter([{setter: null}]).main, null);
});

test('selecting a set replaces an unrelated player focus with its saved setter', () => {
  const context = load();
  const toggle = {};
  context.document = {getElementById: id => id === 'setTrajectoryList' ? {children: []} : toggle};
  context.selectedPlayerId = 'P1';
  context.activePlayerContact = {event_index: 999};
  context.playerContactReport = {contacts: [{event_index: 2, player_id: 'P5'}]};
  context.FPS = 30;
  context.seekAndPlay = () => {};
  context.redraw = () => {};
  context.selectSetTrajectory({start: 60, end: 90, setIndex: 2, setter: 'P5'});
  assert.equal(context.selectedPlayerId, 'P5');
  assert.equal(context.activePlayerContact.event_index, 2);
  assert.equal(toggle.checked, true);
  context.selectSetTrajectory(null);
  assert.equal(context.selectedPlayerId, null);
  assert.equal(context.activePlayerContact, null);
});

test('filters combine court side, attack zone and setter', () => {
  const {setTrajectoryMatches} = load();
  const pair = {team: 'team_b', zone: 'left', setter: 'P5'};
  assert.equal(setTrajectoryMatches(pair, {team: 'team_b', zone: 'left', setter: 'P5'}, 'P3'), true);
  assert.equal(setTrajectoryMatches(pair, {zone: 'right'}, 'P3'), false);
  assert.equal(setTrajectoryMatches(pair, {team: 'team_a'}, 'P3'), false);
  assert.equal(setTrajectoryMatches(pair, {setter: 'other'}, 'P3'), true);
  assert.equal(setTrajectoryMatches(pair, {setter: 'other'}, 'P5'), false);
  // Without a main setter nobody is an exception, and unknown is never "other".
  assert.equal(setTrajectoryMatches(pair, {setter: 'other'}, null), false);
  assert.equal(setTrajectoryMatches({...pair, setter: null}, {setter: 'other'}, 'P3'), false);
  assert.equal(setTrajectoryMatches({...pair, setter: null}, {setter: 'unknown'}, 'P3'), true);
});

test('a click picks the nearest path inside the tolerance', () => {
  const {nearestSetTrajectory} = load();
  const low = {start: 1, attackPoint: [100, 100], runs: [{points: [[0, 100], [100, 100]]}]};
  const high = {start: 2, attackPoint: [100, 40], runs: [{points: [[0, 40], [50, 20]]}, {points: [[60, 22], [100, 40]]}]};
  assert.equal(nearestSetTrajectory([low, high], [50, 94], 10), low);
  assert.equal(nearestSetTrajectory([low, high], [80, 36], 10), high);
  assert.equal(nearestSetTrajectory([low, high], [50, 65], 10), null);
  // The attack dot is clickable even when the path has a single point there.
  assert.equal(nearestSetTrajectory([{start: 3, attackPoint: [10, 10], runs: []}], [12, 12], 10).start, 3);
});
