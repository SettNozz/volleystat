const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

function fixture() {
  const nodes = Object.fromEntries(['nearScore', 'farScore', 'unknownScore', 'a', 'b'].map(id => [id, {}]));
  const context = vm.createContext({document: {getElementById: id => nodes[id],
    querySelector: selector => nodes[selector.includes('"a"') ? 'a' : 'b']}});
  vm.runInContext(readFileSync('viewer/static/serve_score.js', 'utf8'), context);
  context.workspace = {
    config: {teams: {red: {name: 'Red'}, blue: {name: 'Blue'}}, initial_score: {red: 3, blue: 4},
      team_identity_map: {team_a: 'blue', team_b: 'red'}, segments: [
        {start_frame: 0, end_frame: 99, set_id: 's1'},
        {start_frame: 100, end_frame: 199, set_id: 's2', team_identity_map: {team_a: 'red', team_b: 'blue'}},
      ]}, first_set: 's1', points: [
        {frame: 30, set_id: 's1', status: 'confirmed', winner_team_id: 'red'},
        {frame: 60, set_id: 's1', status: 'unresolved', winner_team_id: null},
        {frame: 130, set_id: 's2', status: 'confirmed', winner_team_id: 'blue'},
      ],
  };
  return {nodes, context};
}

test('opening score and team names follow stable identities in the first set', () => {
  const {nodes, context} = fixture();
  vm.runInContext('renderWorkspaceScore(workspace, 20)', context);
  assert.equal(nodes.nearScore.value, 3);
  assert.equal(nodes.farScore.value, 4);
  assert.equal(nodes.b.value, 'Red');
  assert.equal(nodes.a.value, 'Blue');
});

test('unresolved winners stay separate and points appear only at their evidence frame', () => {
  const {nodes, context} = fixture();
  vm.runInContext('renderWorkspaceScore(workspace, 50)', context);
  assert.equal(nodes.nearScore.value, 4);
  assert.equal(nodes.farScore.value, 4);
  assert.equal(nodes.unknownScore.hidden, true);
  vm.runInContext('renderWorkspaceScore(workspace, 70)', context);
  assert.equal(nodes.unknownScore.textContent, '+?1');
  assert.equal(nodes.nearScore.value, 4);
});

test('side switch changes displayed names while each new set starts its own score', () => {
  const {nodes, context} = fixture();
  vm.runInContext('renderWorkspaceScore(workspace, 140)', context);
  assert.equal(nodes.nearScore.value, 1);
  assert.equal(nodes.farScore.value, 0);
  assert.equal(nodes.b.value, 'Blue');
  assert.equal(nodes.a.value, 'Red');
});

test('removed and non-game serves do not create inferred score transitions', () => {
  const {context} = fixture();
  const points = vm.runInContext(`buildServeScore([
    {frame:10,type:'serve'}, {frame:20,type:'serve',review_excluded:true},
    {frame:30,type:'serve',excluded_from_match_stats:true}, {frame:40,type:'serve'}
  ], [], ['team_b','team_a','team_a','team_b'])`, context);
  assert.equal(points.length, 1);
  assert.equal(points[0].near, 1);
  assert.equal(points[0].far, 0);
});
