const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function load() {
  const context = vm.createContext({document: {addEventListener() {}}});
  vm.runInContext(readFileSync('viewer/static/tactics.js', 'utf8'), context);
  return context;
}

test('pair filters combine set, stable team, setter, lane, reception quality and manual rotation', () => {
  const {tacticsMatches} = load();
  const pair = {set_id: 'set_2', team_id: 'home', setter: 'P3', zone: 'left', reception_quality: 'controlled', rotation: 4};
  const filters = {set: 'set_2', team: 'home', setter: 'P3', zone: 'left', quality: 'controlled', rotation: '4'};
  assert.equal(tacticsMatches(pair, filters), true);
  for (const key of Object.keys(filters)) assert.equal(tacticsMatches(pair, {...filters, [key]: 'different'}), false);
});

test('unknown identities and unlabelled rotations only match explicit unknown filters', () => {
  const {tacticsMatches} = load();
  const pair = {set_id: null, team_id: null, setter: null, zone: 'unknown', reception_quality: 'unknown', rotation: null};
  assert.equal(tacticsMatches(pair, {setter: 'unknown', team: 'unknown', rotation: 'unknown'}), true);
  assert.equal(tacticsMatches(pair, {setter: 'P1'}), false);
  assert.equal(tacticsMatches(pair, {rotation: '1'}), false);
});

test('rotation evidence ignores pair-only filters without changing its set/team/rotation scope', () => {
  const {tacticsMatches} = load();
  const rally = {team_id: 'home', set_id: 'set_1', rotation: 3, role: 'receive', won: true};
  assert.equal(tacticsMatches(rally, {team: 'home', setter: 'P3', zone: 'right', quality: 'perfect'}, false), true);
  assert.equal(tacticsMatches(rally, {team: 'away', setter: 'P3'}, false), false);
  assert.equal(tacticsMatches(rally, {rotation: '4'}, false), false);
});

test('side-out and break-point rates retain unknown winners outside classified denominators', () => {
  const {tacticsRotationSummary} = load();
  const rows = [['receive', true], ['receive', false], ['receive', null], ['serve', true], ['serve', null], ['unknown', true]]
    .map(([role, won]) => ({team_id: 'home', set_id: 'set_1', rotation: 1, role, won}));
  const [group] = tacticsRotationSummary(rows);
  assert.equal(group.side_out.wins, 1);
  assert.equal(group.side_out.classified, 2);
  assert.equal(group.side_out.unknown, 1);
  assert.equal(group.break_point.classified, 1);
  assert.equal(group.break_point.unknown, 1);
  assert.equal(group.unknown_roles, 1);
});

test('same rotation number never combines different teams or sets', () => {
  const {tacticsRotationSummary} = load();
  const rows = [{team_id: 'home', set_id: 'set_1'}, {team_id: 'away', set_id: 'set_1'}, {team_id: 'home', set_id: 'set_2'}]
    .map(value => ({...value, rotation: 1, role: 'receive', won: true}));
  assert.equal(tacticsRotationSummary(rows).length, 3);
});
