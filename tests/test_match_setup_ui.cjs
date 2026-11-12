const assert = require('node:assert/strict');
const {test} = require('node:test');
const setup = require('../viewer/static/match_setup.js');

function config() {
  return {team_identity_map: {team_a: 'red', team_b: 'blue'}, teams: {red: {name: 'Red'}, blue: {name: 'Blue'}},
    segments: [{start_frame: 0, end_frame: 99, match_phase: 'non_game'}],
    initial_score: {red: 0, blue: 0}, roster: [{player_id: 'P1', team_id: null, confirmed: false}],
    rotations: [{start_frame: 100, end_frame: 199, team_id: 'blue', rotation: 2}], custom: {keep: true}};
}

test('interval drafts retain inclusive frame boundaries and reject overlap without mutation', () => {
  const original = config(), before = JSON.stringify(original);
  const interval = setup.intervalFromFields({start: '100', end: '199', type: 'set', set: 'set_1', far: 'blue', near: 'red'}, 300);
  const next = setup.putInterval(original, interval, null, 300);
  assert.equal(next.segments.length, 2);
  assert.deepEqual(next.segments[1].team_identity_map, {team_a: 'blue', team_b: 'red'});
  assert.deepEqual(next.rotations, original.rotations);
  assert.deepEqual(next.custom, original.custom);
  assert.throws(() => setup.putInterval(original, {...interval, start_frame: 99}, null, 300), /overlap/);
  assert.equal(JSON.stringify(original), before);
});

test('changing selected interval and removing it preserve the other draft intervals', () => {
  const original = config();
  const changed = setup.putInterval(original, {start_frame: 0, end_frame: 89, match_phase: 'non_game'}, 0, 300);
  assert.equal(changed.segments[0].end_frame, 89);
  assert.deepEqual(setup.removeInterval(changed, 0).segments, []);
  assert.equal(original.segments[0].end_frame, 99);
});

test('frame inputs reject empty, fractional, reversed or out-of-video values', () => {
  for (const [start, end] of [['', '9'], ['0.5', '9'], ['10', '9'], ['0', '300'], ['-1', '20']]) {
    assert.throws(() => setup.frameRange(start, end, 300));
  }
  assert.deepEqual(setup.frameRange('0', '299', 300), [0, 299]);
  assert.throws(() => setup.integer(null, 'Frame'));
  assert.throws(() => setup.integer(false, 'Frame'));
});

test('warm-up removes set/switch fields while match intervals require a set and distinct sides', () => {
  const previous = {start_frame: 0, end_frame: 9, set_id: 'set_1', team_identity_map: {team_a: 'red', team_b: 'blue'}, note: 'keep'};
  assert.deepEqual(setup.intervalFromFields({start: 0, end: 9, type: 'non_game'}, 300, previous),
    {start_frame: 0, end_frame: 9, match_phase: 'non_game', note: 'keep'});
  assert.throws(() => setup.intervalFromFields({start: 0, end: 9, type: 'set', set: '  '}, 300), /set ID/);
  assert.throws(() => setup.intervalFromFields({start: 0, end: 9, type: 'set', set: 'set_1', far: 'red', near: 'red'}, 300), /different teams/);
  assert.equal(setup.intervalFromFields({start: 0, end: 9, type: 'set', set: 'set_1', far: '', near: ''}, 300).team_identity_map, undefined);
  assert.throws(() => setup.intervalFromFields({start: 0, end: 9, type: 'unknown', set: 'set_1'}, 300), /Choose a set/);
});

test('team/score/roster edits keep unknowns explicit and preserve rotation metadata', () => {
  const original = config();
  const fields = {teams: [{id: 'red', name: '  <Red>  ', score: '3'}, {id: 'blue', name: 'Blue', score: '2'}],
    far: 'blue', near: 'red', roster: [{player_id: 'P1', team_id: '', confirmed: false}]};
  const changed = setup.editContextFields(original, fields);
  assert.equal(changed.teams.red.name, '<Red>');
  assert.deepEqual(changed.initial_score, {red: 3, blue: 2});
  assert.deepEqual(changed.roster, [{player_id: 'P1', team_id: null, confirmed: false}]);
  assert.deepEqual(changed.rotations, original.rotations);
  assert.throws(() => setup.editContextFields(original, {...fields, roster: [{player_id: 'P1', team_id: '', confirmed: true}]}), /Choose a team/);
  assert.throws(() => setup.editContextFields(original, {...fields, teams: [{id: 'red', name: 'Red', score: '-1'}]}), /whole number/);
});

test('save binds source and revision; stale 409 never overwrites or silently retries', async () => {
  const calls = [], original = config();
  const payload = {config: original, source: {stats_sha256: 's', annotations_sha256: 'a'}, revision: 4, roster_ids: ['P1']};
  let reject = true;
  const controller = setup.createController(async (url, options) => {
    calls.push([url, options]);
    if (!options) return {ok: true, json: async () => payload};
    return reject ? {ok: false, status: 409, json: async () => ({error: 'Setup changed in another tab'})}
      : {ok: true, json: async () => ({...payload, revision: 5, config: JSON.parse(options.body).config})};
  }, '/api/match_context/job');
  await controller.load();
  const draft = {...original, initial_score: {red: 4, blue: 0}}; controller.setDraft(draft);
  await assert.rejects(controller.save(draft), /another tab/);
  assert.equal(controller.state.revision, 4); assert.equal(controller.state.stale, true);
  assert.deepEqual(controller.state.config.initial_score, {red: 4, blue: 0});
  const count = calls.length;
  await assert.rejects(controller.save(draft), /Reload/);
  assert.equal(calls.length, count);
  const body = JSON.parse(calls[1][1].body);
  assert.deepEqual(body.source, payload.source); assert.equal(body.revision, 4);
  assert.deepEqual(body.config.rotations, original.rotations);
  reject = false; await controller.load(); controller.setDraft(draft); await controller.save(draft);
  assert.equal(controller.state.revision, 5); assert.equal(controller.state.dirty, false); assert.equal(controller.state.stale, false);
});
