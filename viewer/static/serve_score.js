function buildServeScore(events, attempts, eventTeams) {
  const serves = new Map();
  events.forEach((event, index) => {
    if (event.review_excluded || event.excluded_from_match_stats) return;
    if (event.type === 'serve' && Number.isInteger(event.frame) && event.frame >= 0) {
      serves.set(event.frame, eventTeams[index]);
    }
  });
  const attemptsByServe = new Map();
  for (const attempt of attempts) {
    if (attempt.excluded_from_match_stats) continue;
    const start = attempt.lifecycle?.serve_frame ?? attempt.serve_reception?.serve?.frame ?? attempt.start_frame;
    if (!Number.isInteger(start) || start < 0) continue;
    if (!serves.has(start) && !attempt.lifecycle?.serve_confirmed && !attempt.serve_reception?.serve) continue;
    attemptsByServe.set(start, attempt);
    const team = attempt.lifecycle?.serving_team ?? attempt.serve_reception?.serve?.team;
    if (!serves.get(start)) serves.set(start, team);
  }

  const ordered = [...serves].sort(([a], [b]) => a - b).filter(([frame], index, sorted) => {
    const attempt = attemptsByServe.get(frame);
    const next = sorted[index + 1]?.[0];
    // A second serve peak within the same short, non-playable attempt can
    // anchor the actual rally. Keep the playable anchor in that case.
    return !(attempt?.lifecycle?.status === 'serve_not_played' &&
      next - frame <= 3 && next <= attempt.end_frame &&
      attemptsByServe.get(next)?.lifecycle?.is_playable);
  });
  let near = 0;
  let far = 0;
  let unknown = 0;
  return ordered.slice(1).map(([serve, team], index) => {
    const previous = ordered[index][0];
    const end = attemptsByServe.get(previous)?.end_frame;
    if (team === 'team_b') near++;
    else if (team === 'team_a') far++;
    else unknown++;
    return { frame: Number.isInteger(end) && end > previous && end < serve ? end : serve,
      near, far, unknown };
  });
}

function serveScoreAt(points, frame) {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (points[mid].frame <= frame) low = mid + 1;
    else high = mid;
  }
  return points[low - 1] || { near: 0, far: 0, unknown: 0 };
}

function renderWorkspaceScore(workspace, frame) {
  const config = workspace.config;
  const segment = config.segments.find(item => item.start_frame <= frame && frame <= item.end_frame) || {};
  const mapping = {...config.team_identity_map, ...(segment.team_identity_map || {})};
  const set = segment.set_id ?? workspace.first_set;
  const score = Object.fromEntries(Object.keys(config.teams).map(team => [team,
    set === workspace.first_set ? config.initial_score[team] || 0 : 0]));
  let unknown = 0;
  for (const point of workspace.points) {
    if (point.set_id !== set || !Number.isFinite(point.frame) || point.frame > frame) continue;
    if (point.status === 'confirmed' && point.winner_team_id in score) score[point.winner_team_id]++;
    else unknown++;
  }
  document.getElementById('nearScore').value = score[mapping.team_b] || 0;
  document.getElementById('farScore').value = score[mapping.team_a] || 0;
  const missing = document.getElementById('unknownScore');
  missing.hidden = !unknown; missing.textContent = unknown ? `+?${unknown}` : '';
  for (const [side, team] of [['b', mapping.team_b], ['a', mapping.team_a]]) {
    const input = document.querySelector(`.scoreboard input[data-team-side="${side}"]`);
    if (input) { input.value = config.teams[team].name; input.readOnly = true; input.title = 'Edit team names in Match setup'; }
  }
}
