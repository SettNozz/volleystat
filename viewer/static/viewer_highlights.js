// Highlight clips share the main viewer's video, controls and overlays.
const playerHighlights = {
  player: null, moments: [], shown: [], categories: {}, filter: 'all',
  current: -1, continuous: false, clip: null, request: 0, loading: false,
};

function highlightPlayerLabel(id) {
  if (typeof PlayerProfiles !== 'undefined') return PlayerProfiles.label(id);
  try {
    const profiles = JSON.parse(localStorage.getItem(JSON.stringify(['player-profiles', JOB_ID]))) || {};
    const profile = profiles[id] || {};
    return profile.name || (profile.number ? '#' + profile.number : id);
  } catch { return id; }
}

function renderHighlightIdentity() {
  const box = document.getElementById('highlightIdentity');
  if (!box || typeof PlayerProfiles === 'undefined') return;
  box.replaceChildren(...(playerHighlights.player ? [PlayerProfiles.identity(playerHighlights.player, true)] : []));
}

function initPlayerHighlights() {
  const params = new URLSearchParams(location.search);
  playerHighlights.player = params.get('player');
  playerHighlights.filter = params.get('type') || 'all';
  document.getElementById('highlightSuggestions').checked = params.get('suggestions') === '1';
  for (const [id, key] of [['highlightBefore', 'before'], ['highlightAfter', 'after']]) {
    const input = document.getElementById(id);
    try {
      const saved = localStorage.getItem(JSON.stringify(['highlight-window', key]));
      if (saved !== null && Number.isFinite(Number(saved))) input.value = Math.max(0, Math.min(15, Number(saved)));
    } catch {}
    input.addEventListener('change', () => {
      input.value = highlightWindowSeconds(id);
      try { localStorage.setItem(JSON.stringify(['highlight-window', key]), input.value); } catch {}
      cancelPlayerHighlight();
    });
  }
  vid.addEventListener('play', updatePlayerHighlightControls);
  vid.addEventListener('pause', updatePlayerHighlightControls);
  vid.addEventListener('ended', enforcePlayerHighlightClip);
  vid.addEventListener('seeking', () => {
    const clip = playerHighlights.clip;
    if (clip && (vid.currentTime < clip.start - 0.1 || vid.currentTime > clip.end + 0.1)) cancelPlayerHighlight(false);
  });
}

async function loadPlayerHighlights(player = playerHighlights.player) {
  const state = playerHighlights;
  const requestId = ++state.request;
  cancelPlayerHighlight();
  state.loading = true;
  state.moments = [];
  state.shown = [];
  state.current = -1;
  document.getElementById('highlightList').replaceChildren();
  document.getElementById('highlightTypes').replaceChildren();
  updatePlayerHighlightControls();
  document.getElementById('highlightStatus').textContent = 'Loading player highlights…';
  const params = new URLSearchParams();
  if (player) params.set('player', player);
  if (document.getElementById('highlightSuggestions').checked) params.set('suggestions', '1');
  try {
    const response = await fetch('/api/player_highlights/' + encodeURIComponent(JOB_ID) + '?' + params);
    const data = await response.json();
    if (requestId !== state.request) return;
    if (!response.ok) throw new Error(data.error || 'Highlights unavailable');
    state.player = data.player;
    state.moments = data.moments;
    state.categories = data.categories;
    state.loading = false;
    if (state.filter !== 'all' && !(state.filter in state.categories)) state.filter = 'all';
    const select = document.getElementById('highlightPlayer');
    select.replaceChildren();
    for (const id of data.roster) {
      const option = document.createElement('option');
      option.value = id;
      option.dataset.playerOption = id;
      option.textContent = highlightPlayerLabel(id);
      select.append(option);
    }
    select.value = state.player || '';
    select.disabled = !data.roster.length;
    selectedPlayerId = state.player;
    renderHighlightIdentity();
    if (state.player) document.getElementById('togPlayers').checked = true;
    renderPlayerHighlightTypes();
    renderPlayerHighlights();
    redraw();
  } catch (error) {
    if (requestId !== state.request) return;
    state.loading = false;
    updatePlayerHighlightControls();
    document.getElementById('highlightStatus').textContent = error.message;
  }
}

function renderPlayerHighlightTypes() {
  const state = playerHighlights;
  const box = document.getElementById('highlightTypes');
  box.replaceChildren();
  for (const [id, label] of [['all', 'All'], ...Object.entries(state.categories)]) {
    const count = state.moments.filter(moment => id === 'all' || moment.category === id).length;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip' + (state.filter === id ? ' on' : '');
    button.textContent = label + ' · ' + count;
    button.setAttribute('aria-pressed', String(state.filter === id));
    button.onclick = () => {
      cancelPlayerHighlight();
      state.filter = id;
      renderPlayerHighlightTypes();
      renderPlayerHighlights();
    };
    box.append(button);
  }
}

function renderPlayerHighlights() {
  const state = playerHighlights;
  state.shown = state.moments.filter(moment => state.filter === 'all' || moment.category === state.filter);
  state.current = -1;
  const list = document.getElementById('highlightList');
  list.replaceChildren();
  state.shown.forEach((moment, index) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'highlight-row';
    const image = document.createElement('img');
    image.loading = 'lazy';
    image.alt = '';
    image.src = '/api/frame/' + encodeURIComponent(JOB_ID) + '/' + moment.frame;
    image.onerror = () => { image.hidden = true; };
    const text = document.createElement('span');
    const title = document.createElement('strong');
    title.textContent = moment.type.replace(/_/g, ' ');
    const meta = document.createElement('small');
    meta.textContent = formatTime(moment.time) + ' · ' + (moment.label_source === 'human' ? 'human review' : 'model proposal');
    text.append(title, meta);
    if (moment.uncertain) {
      const note = document.createElement('small');
      note.className = 'highlight-uncertain';
      note.textContent = 'Uncertain suggestion';
      text.append(note);
    }
    row.append(image, text);
    row.onclick = () => playPlayerHighlight(index);
    list.append(row);
  });
  updatePlayerHighlightControls();
}

function highlightWindowSeconds(id) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? Math.max(0, Math.min(15, value)) : 0;
}

function cancelPlayerHighlight(pause = true) {
  const state = playerHighlights;
  const hadClip = Boolean(state.clip);
  state.clip = null;
  state.continuous = false;
  if (pause && hadClip) vid.pause();
  updatePlayerHighlightControls();
}

function playPlayerHighlight(index, continuous = false) {
  const state = playerHighlights;
  if (state.loading || index < 0 || index >= state.shown.length) return;
  const moment = state.shown[index];
  cancelPlayerHighlight(false);
  setTrajectoryOverlay.clip = null;
  const duration = Number.isFinite(vid.duration) ? vid.duration : Infinity;
  const start = Math.max(0, Math.min(duration, moment.time - highlightWindowSeconds('highlightBefore')));
  const end = Math.min(duration, moment.time + highlightWindowSeconds('highlightAfter'));
  seekAndPlay(Math.round(start * FPS));
  state.current = index;
  state.continuous = continuous;
  state.clip = {start, end};
  updatePlayerHighlightControls();
  const list = document.getElementById('highlightList');
  const row = list.children[index];
  if (row) list.scrollTop = Math.max(0, row.offsetTop - list.offsetTop - list.clientHeight / 2);
}

function togglePlayerHighlights() {
  const state = playerHighlights;
  if (state.continuous && !vid.paused) { cancelPlayerHighlight(); return; }
  playPlayerHighlight(state.current >= 0 && state.current < state.shown.length - 1 ? state.current : 0, true);
}

function enforcePlayerHighlightClip() {
  const state = playerHighlights;
  if (!state.clip || (vid.paused && !vid.ended) || vid.seeking || vid.currentTime < state.clip.end) return;
  if (state.continuous && state.current < state.shown.length - 1) playPlayerHighlight(state.current + 1, true);
  else cancelPlayerHighlight();
}

function updatePlayerHighlightControls() {
  const state = playerHighlights;
  document.getElementById('highlightPrev').disabled = state.loading || state.current <= 0;
  document.getElementById('highlightNext').disabled = state.loading || state.current >= state.shown.length - 1;
  const playAll = document.getElementById('highlightPlayAll');
  playAll.disabled = state.loading || !state.shown.length;
  playAll.textContent = state.continuous && !vid.paused ? '❚❚ Pause' : '▶ Play all';
  for (const [index, row] of Array.from(document.getElementById('highlightList').children).entries()) {
    row.setAttribute('aria-current', String(index === state.current));
  }
  if (state.loading) return;
  document.getElementById('highlightStatus').textContent = state.current >= 0
    ? `${state.current + 1} / ${state.shown.length} · ${state.shown[state.current].type.replace(/_/g, ' ')}`
    : state.shown.length ? `${state.shown.length} moments` : 'No matching moments for this player.';
}

function handleViewerPlaybackKey(event) {
  if (event.defaultPrevented || event.isComposing || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.code !== 'Space' && event.key !== ' ') return;
  if (!document.getElementById('page-video').classList.contains('active')) return;
  if (document.querySelector('dialog[open]')) return;
  // Preserve native keyboard activation and text entry in interactive controls.
  if (event.target.closest('input, textarea, select, button, a, summary, [role="button"], [role="slider"]') || event.target.isContentEditable) return;
  event.preventDefault();
  if (!event.repeat) togglePlay();
}
