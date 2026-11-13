// Results page main script: state, video page, annotations load, court editing,
// playback controls, listeners and init.
// Needs the page boot constants (JOB_ID, FPS, ...) and is loaded after
// viewer_charts.js, viewer_highlights.js and viewer_overlay.js.

function isPlayableRally(r) {
  const lc = r && r.lifecycle;
  if (!lc) return !LIFECYCLE.available;
  return lc.status === 'playable_rally' && lc.is_playable === true;
}

function lifecycleFrame(item, key, fallback = null) {
  const lc = (item && item.lifecycle) || {};
  const value = lc[key];
  return Number.isFinite(value) ? value : fallback;
}

function rallyStartFrame(r) {
  return lifecycleFrame(r, 'start_frame', r.start_frame);
}

function isSuccessfulOutcome(outcome) {
  return outcome === 'success' || outcome === 'playable_rally';
}

function isFailedOutcome(outcome) {
  return outcome === 'failed' || outcome === 'reception_failed' || outcome === 'serve_fault';
}

// Canonical results are metadata-gated. Legacy results retain their original
// behaviour because they have no lifecycle classification to consume.
const RALLIES = RAW_RALLIES.filter(isPlayableRally);
let eventTeams = EVENTS.map(event => event.team);
const scoreAttempts = SERVICE_ATTEMPTS.length ? SERVICE_ATTEMPTS : RALLIES;
let scorePoints = buildServeScore(EVENTS, scoreAttempts, eventTeams);

const EV_COLORS = {
  serve: '#eab308', attack: '#ef4444', block: '#f97316',
  reception: '#10b981', reception_pass: '#10b981', dig: '#14b8a6',
  setting_pass: '#3b82f6', set: '#3b82f6', pass: '#8b5cf6', ace: '#facc15',
  serve_out: '#9ca3af', out: '#9ca3af', ball_out: '#9ca3af', goal_or_out: '#9ca3af',
};
// Event types without a fixed colour still get a stable, distinct one.
const EV_FALLBACK = ['#ec4899', '#06b6d4', '#84cc16', '#a855f7', '#f43f5e', '#0ea5e9', '#d946ef', '#22c55e'];
function evColor(type) {
  if (EV_COLORS[type]) return EV_COLORS[type];
  let h = 0;
  for (const ch of String(type)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return EV_FALLBACK[h % EV_FALLBACK.length];
}
const GREEN = 'oklch(62% 0.15 150)', RED = 'oklch(60% 0.18 25)', INK = 'oklch(30% 0.03 260)';

const vid = document.getElementById('vid');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const courtEditor = {
  active: false, dragIndex: null, originalCorners: null,
  setups: [], setupId: null,
  // Lines, not corners, is the default: on a wide lens the corners are often
  // off-frame while every line is visible.
  mode: 'lines', anchors: null, drag: null,
};

let annotations = null;
let currentFrameIdx = 0;

/* ── Page navigation ─────────────────────────────────── */
function toggleVideoSettings(open, restoreFocus = true) {
  const menu = document.getElementById('videoSettingsMenu');
  const button = document.getElementById('videoSettingsButton');
  const wasOpen = !menu.hidden;
  const show = open === undefined ? menu.hidden : open;
  menu.hidden = !show;
  button.setAttribute('aria-expanded', String(show));
  if (show) menu.querySelector('.settings-close').focus();
  else if (wasOpen && restoreFocus) button.focus();
}

function showPage(name) {
  const workspace = ['video', 'trajectories', 'highlights'].includes(name);
  if (name !== 'highlights') cancelPlayerHighlight();
  if (name === 'trajectories') {
    selectedPlayerId = null;
    activePlayerContact = null;
  }
  if (name !== 'trajectories') setTrajectoryOverlay.clip = null;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(workspace ? 'page-video' : 'page-' + name).classList.add('active');
  document.querySelectorAll('[data-workspace-panel]').forEach(panel => {
    panel.hidden = panel.dataset.workspacePanel !== name;
  });
  document.querySelectorAll('.nav-item').forEach(n => {
    const active = n.dataset.page === (workspace ? 'video' : name);
    n.classList.toggle('active', active);
    if (active) n.setAttribute('aria-current', 'page');
    else n.removeAttribute('aria-current');
  });
  document.querySelectorAll('[data-workspace-view]').forEach(button => {
    const active = button.dataset.workspaceView === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  toggleVideoSettings(false, false);
  if (name === 'charts') renderCharts();
  if (name === 'highlights') loadPlayerHighlights();
  if (workspace) resizeCanvas();
  const url = new URL(location.href);
  url.pathname = '/results/' + encodeURIComponent(JOB_ID);
  url.searchParams.set('view', name);
  history.replaceState(null, '', url);
}

/* ── Shared chart/table filters ──────────────────────── */
function buildChips(containerId, items, activeSet, labelFn, onToggle, colorFn = null) {
  const box = document.getElementById(containerId);
  box.innerHTML = '';
  for (const it of items) {
    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'chip' + (activeSet.has(it) ? ' on' : '');
    if (colorFn) {
      const dot = document.createElement('i');
      dot.className = 'chip-dot';
      dot.style.background = colorFn(it);
      dot.setAttribute('aria-hidden', 'true');
      el.append(dot);
    }
    el.append(labelFn ? labelFn(it) : it);
    el.setAttribute('aria-pressed', String(activeSet.has(it)));
    el.onclick = () => {
      onToggle(it);
      el.classList.toggle('on', activeSet.has(it));
      el.setAttribute('aria-pressed', String(activeSet.has(it)));
    };
    box.appendChild(el);
  }
}

function toggleSet(set, v) { set.has(v) ? set.delete(v) : set.add(v); }

function updateContactTeams(report) {
  eventTeams = EVENTS.map(event => event.team);
  if (report.side === 'a' || report.side === 'b') {
    const opposite = report.side === 'a' ? 'b' : 'a';
    for (const contact of report.contacts) {
      const index = contact.event_index;
      const event = EVENTS[index];
      if (!event || event.frame !== contact.frame || event.type !== contact.type) continue;
      if (contact.label_source !== 'human' && event.team) continue;
      if (contact.status === 'assigned') eventTeams[index] = 'team_' + report.side;
      else if (contact.status === 'other_side') eventTeams[index] = 'team_' + opposite;
      else if (contact.label_source === 'human') eventTeams[index] = null;
    }
  }
  scorePoints = buildServeScore(EVENTS, scoreAttempts, eventTeams);
  refreshAttackHeights();
  renderEventList();
  redraw();
  refreshSetTrajectories();
}

function renderServeScore() {
  const score = serveScoreAt(scorePoints, currentFrameIdx);
  const near = document.getElementById('nearScore');
  const far = document.getElementById('farScore');
  const unknown = document.getElementById('unknownScore');
  if (near.value !== String(score.near)) near.value = score.near;
  if (far.value !== String(score.far)) far.value = score.far;
  unknown.hidden = !score.unknown;
  const missing = `+?${score.unknown}`;
  if (unknown.textContent !== missing) unknown.textContent = missing;
}

/* ── Annotations load ────────────────────────────────── */
fetch('/api/annotations/' + JOB_ID)
  .then(r => {
    if (!r.ok) throw new Error('Annotations unavailable');
    return r.json();
  })
  .then(data => {
    annotations = {};
    for (const f of data.frames) annotations[f.f] = f;
    window._annData = data;
    initPoseEstimates(data, Boolean(playerContactReport && playerContactReport.parameters && playerContactReport.parameters.estimates));
    courtEditor.originalCorners = Array.isArray(data.court_corners)
      ? JSON.parse(JSON.stringify(data.court_corners)) : null;
    const slider = document.getElementById('netYSlider');
    const fw = data.frame_width || 1280;
    const fh = data.frame_height || 720;
    slider.max = fh;
    slider.value = data.net_y;
    document.getElementById('netYVal').textContent = Math.round(data.net_y);
    const xLSlider = document.getElementById('netXLSlider');
    const xRSlider = document.getElementById('netXRSlider');
    const postLeftX = !data.net_tape_endpoint_y && data.net_floor_left ? data.net_floor_left[0] : data.net_x_left;
    const postRightX = !data.net_tape_endpoint_y && data.net_floor_right ? data.net_floor_right[0] : data.net_x_right;
    xLSlider.max = fw; xLSlider.value = postLeftX ?? Math.round(fw * 0.29);
    xRSlider.max = fw; xRSlider.value = postRightX ?? Math.round(fw * 0.66);
    document.getElementById('netXLVal').textContent = Math.round(xLSlider.value);
    document.getElementById('netXRVal').textContent = Math.round(xRSlider.value);
    renderAttackHeatmapSummary(data.attack_heatmap);
    refreshAttackHeights();
    refreshSetTrajectories();
    startRenderLoop();
    redraw();
    if (document.getElementById('page-charts').classList.contains('active')) renderCharts(true);
    loadCourtState();
  })
  .catch(error => {
    document.getElementById('setTrajectoryStatus').textContent = 'Ball annotations unavailable.';
    console.error(error);
  });

function updateNetY(val) {
  if (!window._annData) return;
  const d = window._annData;
  const newY = parseFloat(val);
  const tapeH = (d.net_tape_bottom != null && d.net_tape_top != null)
    ? (d.net_tape_bottom - d.net_tape_top) : 50;
  if (d.net_tape_endpoint_y) {
    const shift = newY - d.net_tape_top;
    d.net_tape_endpoint_y = d.net_tape_endpoint_y.map(y => y + shift);
  }
  d.net_y = newY;
  d.net_tape_top = newY;
  d.net_tape_bottom = newY + tapeH;
  document.getElementById('netYVal').textContent = Math.round(newY);
  redraw();
  if (document.getElementById('page-charts').classList.contains('active')) renderCharts(true);
}

function updateNetX() {
  if (!window._annData) return;
  const l = parseFloat(document.getElementById('netXLSlider').value);
  const r = parseFloat(document.getElementById('netXRSlider').value);
  window._annData.net_x_left = l;
  window._annData.net_x_right = r;
  if (window._annData.net_floor_left) window._annData.net_floor_left[0] = l;
  if (window._annData.net_floor_right) window._annData.net_floor_right[0] = r;
  document.getElementById('netXLVal').textContent = Math.round(l);
  document.getElementById('netXRVal').textContent = Math.round(r);
  redraw();
  if (document.getElementById('page-charts').classList.contains('active')) renderCharts(true);
}

function defaultCourtCorners() {
  const d = window._annData || {};
  const w = Number(d.frame_width || 1280), h = Number(d.frame_height || 720);
  return [[w * .20, h * .40], [w * .80, h * .40], [w * .97, h * .94], [w * .03, h * .94]];
}

function toggleCourtEdit() {
  if (!window._annData) return;
  const d = window._annData;
  if (!Array.isArray(d.court_corners) || d.court_corners.length !== 4) {
    d.court_corners = defaultCourtCorners();
    if (!courtEditor.originalCorners) courtEditor.originalCorners = JSON.parse(JSON.stringify(d.court_corners));
  }
  courtEditor.active = !courtEditor.active;
  courtEditor.dragIndex = null;
  courtEditor.drag = null;
  canvas.classList.toggle('court-editing', courtEditor.active);
  document.getElementById('courtEditBtn').textContent = courtEditor.active ? 'Done editing' : 'Edit court';
  document.getElementById('courtSaveBtn').disabled = !courtEditor.active;
  document.getElementById('courtResetBtn').disabled = !courtEditor.active;
  document.getElementById('courtSetupSaveBtn').disabled = !courtEditor.active;
  document.getElementById('courtModeBtn').disabled = !courtEditor.active;
  if (courtEditor.active) {
    setCourtEditMode(courtEditor.mode);
    return;
  }
  setCourtHint('Drag the court lines into place. Dashed guides show the centre line and both 3 m attack lines — align those, the far baseline is usually hidden.');
  redraw();
}

function setCourtHint(text) {
  document.getElementById('courtEditHint').textContent = text;
}

function resetCourtCorrection() {
  if (!window._annData || !courtEditor.originalCorners) return;
  window._annData.court_corners = JSON.parse(JSON.stringify(courtEditor.originalCorners));
  syncEdgeAnchors();
  redraw();
}

async function saveCourtCorrection(setupName) {
  const d = window._annData;
  if (!d || !Array.isArray(d.court_corners) || d.court_corners.length !== 4) return;
  if (!courtCornersAreConvex(d.court_corners)) {
    setCourtHint('Those four points cross over each other. Order them A, B (near baseline) then C, D (far baseline).');
    return;
  }
  const button = document.getElementById('courtSaveBtn');
  button.disabled = true;
  button.textContent = 'Saving…';
  setCourtHint('Saving and rebuilding the attack map…');
  try {
    const response = await fetch('/api/court/' + JOB_ID, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        court_corners: d.court_corners,
        net_y: d.net_y,
        net_x_left: d.net_x_left,
        net_x_right: d.net_x_right,
        save_as_setup: setupName || undefined,
        court_setup_id: setupName ? (courtEditor.setupId || undefined) : undefined,
      }),
    });
    const saved = await response.json();
    if (!response.ok) throw new Error(saved.error || 'Could not save court correction');
    d.court_corners = saved.court_corners;
    for (const key of ['net_y', 'net_x_left', 'net_x_right', 'net_tape_top',
      'net_tape_bottom', 'net_tape_endpoint_y', 'net_floor_left', 'net_floor_right']) {
      if (key in saved) d[key] = saved[key];
    }
    redraw();
    if (saved.attack_heatmap) d.attack_heatmap = saved.attack_heatmap;
    courtEditor.originalCorners = JSON.parse(JSON.stringify(saved.court_corners));
    courtEditor.setupId = saved.court_setup_id || null;
    renderCourtSetups(saved.setups || [], courtEditor.setupId);
    renderAttackHeatmapSummary(d.attack_heatmap);
    refreshAttackHeights();
    refreshSetTrajectories();
    if (document.getElementById('page-charts').classList.contains('active')) renderCharts(true);
    const total = (d.attack_heatmap || {}).total_attacks;
    setCourtHint(`Saved${saved.court_setup_name ? ' as “' + saved.court_setup_name + '”' : ''}. Attack map rebuilt from ${total} attacks.`);
  } catch (error) {
    setCourtHint('Save failed: ' + error.message);
  } finally {
    button.textContent = 'Save';
    button.disabled = !courtEditor.active;
  }
}

/* ── Court setups (one camera placement, reusable across games) ───── */
function renderCourtSetups(setups, selectedId) {
  courtEditor.setups = setups || [];
  const select = document.getElementById('courtSetupSelect');
  select.innerHTML = '<option value="">— this game only —</option>';
  for (const setup of courtEditor.setups) {
    const option = document.createElement('option');
    option.value = setup.id;
    option.textContent = setup.name;
    select.appendChild(option);
  }
  select.value = selectedId || '';
  courtEditor.setupId = selectedId || null;
  document.getElementById('courtSetupDeleteBtn').disabled = !select.value;
}

function applyCourtSetup(setupId) {
  const d = window._annData;
  document.getElementById('courtSetupDeleteBtn').disabled = !setupId;
  courtEditor.setupId = setupId || null;
  if (!setupId || !d) return;
  const setup = (courtEditor.setups || []).find(item => item.id === setupId);
  if (!setup) return;
  // A setup is stored in the pixels of the game it was drawn on; this result
  // may be a different resolution.
  const sx = Number(d.frame_width || 1280) / Number(setup.frame_width || 1280);
  const sy = Number(d.frame_height || 720) / Number(setup.frame_height || 720);
  d.court_corners = setup.court_corners.map(([x, y]) => [x * sx, y * sy]);
  syncEdgeAnchors();
  setCourtHint(`Loaded “${setup.name}”. Check it against the lines, then Save to rebuild the attack map.`);
  redraw();
}

async function saveCourtSetup() {
  const suggested = (courtEditor.setups || []).find(
    item => item.id === courtEditor.setupId,
  );
  const name = prompt(
    'Name this camera setup (reusable for other games shot from the same spot):',
    suggested ? suggested.name : '',
  );
  if (!name) return;
  await saveCourtCorrection(name.trim());
}

async function deleteCourtSetup() {
  const setupId = document.getElementById('courtSetupSelect').value;
  if (!setupId) return;
  const setup = (courtEditor.setups || []).find(item => item.id === setupId);
  if (!confirm(`Delete court setup “${setup ? setup.name : setupId}”? Results already saved with it keep their corners.`)) return;
  const response = await fetch('/api/court_setups/' + setupId, {method: 'DELETE'});
  if (!response.ok) {
    setCourtHint('Could not delete that setup.');
    return;
  }
  const list = await (await fetch('/api/court_setups')).json();
  renderCourtSetups(list.setups || [], null);
}

async function loadCourtState() {
  try {
    const state = await (await fetch('/api/court/' + JOB_ID)).json();
    if (state.error) return;
    renderCourtSetups(state.setups || [], state.court_setup_id || null);
    const d = window._annData;
    if (d && !Array.isArray(d.court_corners) && Array.isArray(state.court_corners)) {
      // Results analysed before corners were stored fall back to the shared
      // calibration, so the editor starts from the geometry actually used.
      d.court_corners = state.court_corners;
      refreshAttackHeights();
      redraw();
    }
    if (state.heatmap_version !== state.current_heatmap_version) {
      await recomputeAttackMap(state.heatmap_version);
    }
  } catch (error) {
    /* the court panel is optional; the rest of the viewer still works */
  }
}

async function recomputeAttackMap(oldVersion) {
  setCourtHint(oldVersion
    ? 'This result predates measured shot directions — rebuilding the attack map…'
    : 'Rebuilding the attack map…');
  try {
    const response = await fetch('/api/court/' + JOB_ID + '/recompute', {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'recompute failed');
    window._annData.attack_heatmap = data.attack_heatmap;
    renderAttackHeatmapSummary(data.attack_heatmap);
    if (document.getElementById('page-charts').classList.contains('active')) renderCharts(true);
    setCourtHint(`Attack map rebuilt: ${data.attack_heatmap.total_attacks} attacks. Correct the court if the zones look wrong.`);
  } catch (error) {
    setCourtHint('Attack map could not be rebuilt: ' + error.message);
  }
}

function eventImagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  const d = window._annData || {};
  const w = Number(d.frame_width || vid.videoWidth || 1280);
  const h = Number(d.frame_height || vid.videoHeight || 720);
  return [
    (event.clientX - rect.left) * w / Math.max(rect.width, 1),
    (event.clientY - rect.top) * h / Math.max(rect.height, 1),
  ];
}

/* ── Line editing ───────────────────────────────────────────────────
   A wide lens puts court corners outside the frame — the near baseline of
   this footage meets the left sideline at a point the camera never saw. The
   lines themselves are always visible, so the court is edited as four lines
   and the corners are read off as their intersections. */
const EDGE_NAMES = ['near baseline', 'right sideline', 'far baseline', 'left sideline'];

function syncEdgeAnchors() {
  const d = window._annData;
  if (!d || !Array.isArray(d.court_corners)) return;
  courtEditor.anchors = courtEdgeAnchors(
    d.court_corners,
    Number(d.frame_width || 1280),
    Number(d.frame_height || 720),
  );
}

function applyEdgeAnchors() {
  const d = window._annData;
  if (!d || !courtEditor.anchors) return;
  d.court_corners = cornersFromEdgeAnchors(courtEditor.anchors, d.court_corners);
}

function setCourtEditMode(mode) {
  courtEditor.mode = mode;
  courtEditor.drag = null;
  if (mode === 'lines') syncEdgeAnchors();
  document.getElementById('courtModeBtn').textContent =
    mode === 'lines' ? 'Editing: lines' : 'Editing: corners';
  setCourtHint(mode === 'lines'
    ? 'Drag a line to slide it, or its round handles to pivot it. Corners are the intersections — they may sit outside the frame.'
    : 'Drag A/B/C/D. Only works for corners the camera actually sees; switch to line mode for the rest.');
  redraw();
}

function toggleCourtEditMode() {
  setCourtEditMode(courtEditor.mode === 'lines' ? 'corners' : 'lines');
}

function distanceToSegment(point, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-9) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  let t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSquared;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

canvas.addEventListener('pointerdown', event => {
  if (!courtEditor.active || !window._annData) return;
  const point = eventImagePoint(event);
  const scale = Number(window._annData.frame_width || 1280) / 1280;
  const grab = 26 * Math.max(1, scale);

  if (courtEditor.mode === 'lines') {
    if (!courtEditor.anchors) syncEdgeAnchors();
    let best = grab, found = null;
    courtEditor.anchors.forEach((anchor, edge) => {
      anchor.forEach((handle, index) => {
        const distance = Math.hypot(handle[0] - point[0], handle[1] - point[1]);
        if (distance <= best) { best = distance; found = {edge, handle: index}; }
      });
    });
    if (!found) {
      let bestLine = grab * 0.8;
      courtEditor.anchors.forEach((anchor, edge) => {
        const distance = distanceToSegment(point, anchor[0], anchor[1]);
        if (distance <= bestLine) { bestLine = distance; found = {edge, handle: null}; }
      });
    }
    if (found) {
      courtEditor.drag = {...found, last: point};
      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    }
    return;
  }

  const corners = window._annData.court_corners || [];
  let selected = -1, best = grab;
  corners.forEach((corner, index) => {
    const distance = Math.hypot(corner[0] - point[0], corner[1] - point[1]);
    if (distance <= best) { best = distance; selected = index; }
  });
  if (selected >= 0) {
    courtEditor.dragIndex = selected;
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  }
});

canvas.addEventListener('pointermove', event => {
  if (!courtEditor.active || !window._annData) return;
  const [x, y] = eventImagePoint(event);
  const d = window._annData;

  if (courtEditor.mode === 'lines') {
    const drag = courtEditor.drag;
    if (!drag) return;
    const anchor = courtEditor.anchors[drag.edge];
    if (drag.handle === null) {
      const dx = x - drag.last[0], dy = y - drag.last[1];
      anchor[0] = [anchor[0][0] + dx, anchor[0][1] + dy];
      anchor[1] = [anchor[1][0] + dx, anchor[1][1] + dy];
      drag.last = [x, y];
    } else {
      anchor[drag.handle] = [x, y];
    }
    applyEdgeAnchors();
    redraw();
    return;
  }

  if (courtEditor.dragIndex === null) return;
  // Corner mode stays inside the frame; a corner you cannot see is a corner
  // you cannot aim, and line mode is the tool for those.
  d.court_corners[courtEditor.dragIndex] = [
    Math.max(0, Math.min(d.frame_width, x)),
    Math.max(0, Math.min(d.frame_height, y)),
  ];
  redraw();
});

canvas.addEventListener('pointerup', event => {
  if ((courtEditor.dragIndex !== null || courtEditor.drag) && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  courtEditor.dragIndex = null;
  courtEditor.drag = null;
  if (courtEditor.mode === 'lines' && courtEditor.active) syncEdgeAnchors();
});

/* ── Set-to-attack line picking ─────────────────────────────────────
   The canvas takes pointer input for these only while the overlay is shown
   (see drawSetTrajectories); court editing keeps priority. */
canvas.addEventListener('click', event => {
  if (courtEditor.active || !setTrajectoriesShown()) return;
  selectSetTrajectory(setTrajectoryAtEvent(event));
});

canvas.addEventListener('pointermove', event => {
  if (courtEditor.active) return;
  hoverSetTrajectory(setTrajectoryAtEvent(event));
});

canvas.addEventListener('pointerleave', () => hoverSetTrajectory(null));

function attackZoneCounts(heatmap) {
  const counts = {left: 0, center: 0, right: 0};
  for (const zone of ((heatmap && heatmap.zones) || [])) {
    const id = String(zone.id || '').toLowerCase();
    if (id === 'left' || id.endsWith('_left')) counts.left += zone.count || 0;
    else if (id === 'center' || id === 'middle' || id.endsWith('_middle')) counts.center += zone.count || 0;
    else if (id === 'right' || id.endsWith('_right')) counts.right += zone.count || 0;
  }
  return counts;
}

function renderAttackHeatmapSummary(heatmap) {
  if (!heatmap) return;
  document.getElementById('attackHeatmapPanel').hidden = false;
  const counts = attackZoneCounts(heatmap);
  const total = heatmap.total_attacks != null
    ? heatmap.total_attacks : counts.left + counts.center + counts.right;
  document.getElementById('attackHeatmapTotal').textContent = total;
  const rows = {};
  for (const zone of (heatmap.zones || [])) {
    rows[String(zone.id || '').toLowerCase()] = zone;
  }
  const zones = document.getElementById('attackHeatmapZones');
  zones.innerHTML = '';
  for (const id of ['left', 'center', 'right']) {
    const zone = rows[id] || {};
    const item = document.createElement('div');
    item.style.cssText = 'background:var(--head-bg);border-radius:5px;padding:5px 8px;font-size:11.5px;';
    const head = document.createElement('div');
    head.style.cssText = 'display:flex;justify-content:space-between;gap:6px;';
    const name = document.createElement('span');
    name.textContent = id;
    name.style.color = 'var(--muted)';
    const count = document.createElement('strong');
    count.textContent = counts[id];
    count.style.color = 'var(--primary)';
    head.append(name, count);
    item.appendChild(head);
    if (zone.front != null || zone.back != null) {
      const split = document.createElement('div');
      split.style.cssText = 'font-size:10px;color:var(--muted);margin-top:2px';
      split.textContent = `${zone.front || 0} front · ${zone.back || 0} back`;
      item.appendChild(split);
    }
    zones.appendChild(item);
  }
  const directions = heatmap.directions;
  let note = document.getElementById('attackHeatmapDirections');
  if (directions) {
    if (!note) {
      note = document.createElement('div');
      note.id = 'attackHeatmapDirections';
      note.style.cssText = 'font-size:11px;color:var(--muted);margin-top:6px';
      zones.parentElement.appendChild(note);
    }
    note.textContent = `${directions.line || 0} down the line · ${directions.diagonal || 0} diagonal`
      + (directions.unknown ? ` · ${directions.unknown} not measured` : '');
  }
}

let _syncOffset = 0;
function changeSyncOffset(d) {
  _syncOffset = Math.max(-5, Math.min(5, _syncOffset + d));
  document.getElementById('syncOffsetVal').textContent = _syncOffset;
  redraw();
}

/* ── Canvas sizing / render loop ─────────────────────── */
function resizeCanvas() {
  const vr = vid.getBoundingClientRect();
  if (!vr.width) return;
  document.getElementById('videoSettingsMenu').style.maxHeight = Math.max(100, vr.height - 78) + 'px';
  const vw = vid.videoWidth || (window._annData && window._annData.frame_width) || 1280;
  const vh = vid.videoHeight || (window._annData && window._annData.frame_height) || 720;
  const scale = Math.min(vr.width / vw, vr.height / vh);
  canvas.width = vw * scale;
  canvas.height = vh * scale;
  canvas.style.width = canvas.width + 'px';
  canvas.style.height = canvas.height + 'px';
  redraw();
}

vid.addEventListener('loadedmetadata', () => {
  resizeCanvas(); startRenderLoop();
  const params = new URLSearchParams(location.search);
  const entryFrame = Number(params.get('frame'));
  if (params.has('frame') && Number.isFinite(entryFrame) && entryFrame >= 0) {
    seekAndPlay(Math.min(entryFrame, Math.max(0, TOTAL_FRAMES - 1)), false);
  }
});
window.addEventListener('resize', resizeCanvas);

vid.addEventListener('timeupdate', () => {
  enforceSetTrajectoryClip();
  enforcePlayerHighlightClip();
  updateTimeline();
  updateActiveEvent(Math.max(0, Math.round(vid.currentTime * FPS) + _syncOffset));
});

// rAF and rVFC are both suspended while the tab is in the background, and
// neither fires for a seek performed while paused. Sync the overlay straight
// from the seek so it never shows a stale frame.
vid.addEventListener('seeked', () => {
  currentFrameIdx = Math.max(0, Math.round(vid.currentTime * FPS) + _syncOffset);
  _lastDrawnFrame = currentFrameIdx;
  updateTimeline();
  redraw();
});

let _rafId = null;
let _lastDrawnFrame = -1;
const SYNC_OFFSET_FRAMES = 0;

function _drawAtMediaTime(mediaTime) {
  const fi = Math.max(0, Math.round(vid.currentTime * FPS) + _syncOffset);
  if (fi !== _lastDrawnFrame) {
    currentFrameIdx = fi;
    _lastDrawnFrame = fi;
    redraw();
  }
}

function startRenderLoop() {
  // rVFC gives exact per-frame sync, but some embedded browsers expose the
  // method and never invoke it — which leaves the overlay frozen on frame 0.
  // The rAF loop always runs so the overlay can never stall; rVFC refines it
  // to the presented frame where it genuinely works. Both write the same
  // fields, so whichever fires first wins and the other is a no-op.
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    function rVFCLoop(now, meta) {
      _drawAtMediaTime(meta.mediaTime);
      vid.requestVideoFrameCallback(rVFCLoop);
    }
    vid.requestVideoFrameCallback(rVFCLoop);
  }
  if (_rafId) return;
  function rafLoop() {
    enforceSetTrajectoryClip();
    enforcePlayerHighlightClip();
    const fi =Math.max(0, Math.round(vid.currentTime * FPS) - SYNC_OFFSET_FRAMES + _syncOffset);
    if (fi !== _lastDrawnFrame) {
      currentFrameIdx = fi;
      _lastDrawnFrame = fi;
      redraw();
    }
    _rafId = requestAnimationFrame(rafLoop);
  }
  _rafId = requestAnimationFrame(rafLoop);
}

vid.addEventListener('play', () => {
  document.getElementById('playBtn').textContent = '⏸ Pause';
  updateActiveEvent();
});
vid.addEventListener('pause', () => {
  document.getElementById('playBtn').textContent = '▶ Play';
  updateActiveEvent();
});

function togglePlay() { vid.paused ? vid.play() : vid.pause(); }
function toggleMute() {
  vid.muted = !vid.muted;
  document.getElementById('muteBtn').textContent = vid.muted ? '🔇' : '🔊';
}
function setSpeed(s, btn) {
  vid.playbackRate = s;
  document.querySelectorAll('.btn-row .ctrl').forEach(b => {
    if (b.textContent.endsWith('×')) b.classList.remove('active');
  });
  if (btn) btn.classList.add('active');
}
function stepFrame(d) {
  cancelPlayerHighlight(false);
  vid.pause();
  vid.currentTime = Math.max(0, vid.currentTime + d / FPS);
}
function seekToFrame(f) { vid.currentTime = f / FPS; }
function seekAndPlay(f, autoPlay = true) {
  cancelPlayerHighlight(false);
  vid.currentTime = f / FPS;
  if (autoPlay) vid.play(); else vid.pause();
}
function seekVideo(value) {
  if (!Number.isFinite(vid.duration)) return;
  cancelPlayerHighlight(false);
  vid.currentTime = Number(value) / 1000 * vid.duration;
}

function updateTimeline() {
  renderViewerLineup(Math.max(0, Math.round(vid.currentTime * FPS)));
  const pct = vid.duration ? (vid.currentTime / vid.duration * 100) : 0;
  const seek = document.getElementById('videoSeek');
  seek.value = Math.round(pct * 10);
  seek.setAttribute('aria-valuetext', formatTime(vid.currentTime) + ' of ' + formatTime(vid.duration || 0));
  const cur = formatTime(vid.currentTime);
  const tot = formatTime(vid.duration || 0);
  document.getElementById('timeDisplay').textContent = cur + ' / ' + tot;
  const badge = document.getElementById('videoBadge');
  badge.textContent = `${VIDEO_STEM} · f${currentFrameIdx} · ${cur}`;
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60).toString().padStart(2, '0');
  return m + ':' + ss;
}

vid.addEventListener('play',  () => { document.title = `▶ CourtSense — ${VIDEO_STEM}`; });
vid.addEventListener('pause', () => { document.title = `⏸ f${currentFrameIdx} — CourtSense`; });

/* ── Init ────────────────────────────────────────────── */
document.querySelector('.video-shell').append(document.getElementById('videoSettings'));
document.addEventListener('pointerdown', event => {
  if (!document.getElementById('videoSettings').contains(event.target)) toggleVideoSettings(false, false);
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') toggleVideoSettings(false);
  handleViewerPlaybackKey(event);
});
document.getElementById('videoSettings').addEventListener('focusout', event => {
  // Clicking a non-focusable label briefly moves focus to the document body
  // before its checkbox receives the click. Keep the menu open during that gap.
  if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget)) toggleVideoSettings(false, false);
});
initChartFilters();
renderEventList();
window.addEventListener('player-profiles-changed', renderEventList);
window.addEventListener('team-names-changed', () => {
  initChartFilters();
  if (chartsBuilt) renderCharts(true);
  renderSetTrajectories();
  renderPlayerContacts();
  redraw();
});
initPlayerHighlights();
showPage(INITIAL_PAGE);
