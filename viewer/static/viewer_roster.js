// Saved roster metadata only; lineup follows recorded substitution intervals.
let viewerRoster = null;
let viewerRosterKey = null;

async function loadViewerRoster() {
  try {
    const response = await fetch('/api/player_roster/' + encodeURIComponent(JOB_ID));
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Players unavailable');
    viewerRoster = data;
    PlayerProfiles.configure({jobId: JOB_ID, sourceVersion: data.source.annotations_sha256,
      roster: data.roster, portraits: data.portraits, positions: data.positions});
    document.getElementById('edit-players').disabled = !data.roster.length;
    renderViewerLineup(currentFrameIdx);
  } catch (error) {
    document.getElementById('rosterStatus').textContent = error.message;
  }
}

function rosterPlayersAtFrame(roster, lineup, frame) {
  const playing = [], bench = [];
  for (const id of roster) {
    const intervals = lineup[id]?.intervals;
    (intervals && !intervals.some(([start, end]) => start <= frame && frame <= end) ? bench : playing).push(id);
  }
  return {playing, bench};
}

function renderViewerLineup(frame) {
  if (!viewerRoster) return;
  const {playing, bench} = rosterPlayersAtFrame(viewerRoster.roster, viewerRoster.lineup, frame);
  const key = JSON.stringify([playing, bench, selectedPlayerId]);
  if (key === viewerRosterKey) return;
  viewerRosterKey = key;
  const render = (target, ids) => {
    const list = document.getElementById(target);
    list.replaceChildren();
    for (const id of ids) {
      const row = document.createElement('button');
      row.type = 'button'; row.className = 'roster-player';
      row.setAttribute('aria-pressed', String(selectedPlayerId === id));
      row.append(PlayerProfiles.identity(id));
      row.onclick = () => { selectPlayer(selectedPlayerId === id ? null : id); renderViewerLineup(frame); };
      list.append(row);
    }
  };
  render('teamRosterList', playing);
  render('rosterBenchList', bench);
  document.getElementById('rosterBench').hidden = !bench.length;
  document.getElementById('teamRosterTitle').textContent = Object.keys(viewerRoster.lineup).length ? 'Team lineup' : 'Team roster';
  document.getElementById('rosterStatus').textContent = viewerRoster.roster.length
    ? 'Add names and positions with Edit players. Select a player to follow them on video.' : 'No tracked player roster in this result.';
  PlayerProfiles.refresh();
}

document.addEventListener('DOMContentLoaded', loadViewerRoster);
window.addEventListener('player-profiles-changed', () => {
  // Profile targets update in place; only lineup membership rebuilds cards.
  if (typeof renderHighlightIdentity === 'function') renderHighlightIdentity();
});
