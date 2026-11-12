// Full-match set-to-attack overlays from saved contacts and active ball coordinates.
const SET_CONTACTS = new Set(['set', 'setting_pass', 'pass']);
const COURT_SIDES = new Set(['team_a', 'team_b']);
const SET_ATTACK_ZONES = {
  left: {label: 'Left', color: '#38bdf8'},
  center: {label: 'Center', color: '#facc15'},
  right: {label: 'Right', color: '#f472b6'},
  unknown: {label: 'Unknown zone', color: '#a1a1aa'},
};
const SET_TRAJECTORY_LEAD_SECONDS = 0.5;
const SET_TRAJECTORY_TAIL_SECONDS = 2;
const setTrajectoryOverlay = {
  pairs: [], visible: [], paths: null,
  selected: null, hovered: null, zone: 'all',
  // Frame range of the clip started from a pair; playback pauses at `stop`.
  clip: null,
};

function setAttackPairs(events, teams, rallies) {
  const contacts = events.map((event, index) => ({event, index, team: teams[index]}))
    .filter(({event}) => Number.isInteger(event.frame) && !event.review_excluded && !event.excluded_from_match_stats)
    .sort((a, b) => a.event.frame - b.event.frame);
  const ranges = rallies.map(rally => [
    rallyStartFrame(rally), lifecycleFrame(rally, 'end_frame', rally.end_frame),
  ]).filter(range => range.every(Number.isFinite));
  const pairs = [];
  for (let i = 1; i < contacts.length; i++) {
    const set = contacts[i - 1];
    const attack = contacts[i];
    const start = set.event.frame;
    const end = attack.event.frame;
    if (!SET_CONTACTS.has(set.event.type) || attack.event.type !== 'attack' || end <= start) continue;
    if (!ranges.some(([first, last]) => first <= start && end <= last)) continue;
    const setTeam = COURT_SIDES.has(set.team) ? set.team : null;
    const attackTeam = COURT_SIDES.has(attack.team) ? attack.team : null;
    if (setTeam && attackTeam && setTeam !== attackTeam) continue;
    pairs.push({
      start, end, setIndex: set.index, attackIndex: attack.index,
      team: setTeam === attackTeam ? setTeam : null,
    });
  }
  return pairs;
}

// Roster IDs come only from contacts the player-contact report assigned.
// Unresolved contacts stay unidentified; a setter is never inferred here.
function setAttackSetterIds(report, events) {
  const setters = new Map();
  for (const contact of report?.contacts || []) {
    const event = events[contact.event_index];
    if (!event || event.frame !== contact.frame || event.type !== contact.type) continue;
    if (contact.status === 'assigned' && contact.player_id) {
      setters.set(contact.event_index, contact.player_id);
    }
  }
  return setters;
}

// The usual setter is the roster ID with the most identified sets. A tie has
// no usual setter, so no pair is flagged as an exception.
function setAttackMainSetter(pairs) {
  const counts = new Map();
  for (const {setter} of pairs) {
    if (setter) counts.set(setter, (counts.get(setter) || 0) + 1);
  }
  const ranked = [...counts].sort((a, b) =>
    b[1] - a[1] || a[0].localeCompare(b[0], undefined, {numeric: true}));
  const main = ranked.length && (ranked.length === 1 || ranked[0][1] > ranked[1][1])
    ? ranked[0][0] : null;
  return {ranked, main};
}

function setTrajectoryMatches(pair, {team = 'all', zone = 'all', setter = 'all'}, main) {
  if (team !== 'all' && pair.team !== team) return false;
  if (zone !== 'all' && pair.zone !== zone) return false;
  if (setter === 'all') return true;
  if (setter === 'unknown') return !pair.setter;
  if (setter === 'other') return Boolean(main && pair.setter && pair.setter !== main);
  return pair.setter === setter;
}

function nearestSetTrajectory(pairs, point, tolerance) {
  let nearest = null;
  let best = tolerance;
  for (const pair of pairs) {
    let distance = pair.attackPoint
      ? Math.hypot(point[0] - pair.attackPoint[0], point[1] - pair.attackPoint[1]) : Infinity;
    for (const {points} of pair.runs) {
      for (let i = 1; i < points.length; i++) {
        distance = Math.min(distance, distanceToSegment(point, points[i - 1], points[i]));
      }
    }
    if (distance <= best) {
      best = distance;
      nearest = pair;
    }
  }
  return nearest;
}

function setAttackBallPoint(ann) {
  const point = ann?.ball;
  return !ann?.ball_out_of_frame && Array.isArray(point) &&
    Number.isFinite(point[0]) && Number.isFinite(point[1]) ? point : null;
}

function setAttackArcBounds(frames, start, end, fps, frameWidth) {
  const pointAt = frame => setAttackBallPoint(frames[frame]);
  const scale = (frameWidth || 1280) / 1280;
  const edgeWindow = Math.max(2, Math.round(fps * 0.24));
  const legFrames = Math.max(2, Math.round(fps * 0.10));
  let apex = null;
  for (let frame = start; frame <= end; frame++) {
    const point = pointAt(frame);
    if (point && (apex === null || point[1] < pointAt(apex)[1])) apex = frame;
  }
  if (apex === null || apex <= start || apex >= end) return {start, end};

  // A set marked slightly early can still contain the last descending
  // frames of the incoming ball. Start at the bottom of that short leg.
  const first = pointAt(start);
  if (first) {
    let bottom = start;
    for (let frame = start + 1; frame <= Math.min(start + edgeWindow, apex - 1); frame++) {
      const point = pointAt(frame);
      if (!point || frames[frame].ball_new) break;
      if (point[1] > pointAt(bottom)[1]) bottom = frame;
    }
    if (pointAt(bottom)[1] - first[1] >= 4 * scale) start = bottom;
  }

  // Look for a break in velocity on the descending leg near the saved
  // attack. Require intact observations on both sides of the candidate.
  const velocity = (from, to) => {
    const p = pointAt(from);
    if (!p) return null;
    for (let frame = from + 1; frame <= to; frame++) {
      if (!pointAt(frame) || frames[frame].ball_new) return null;
    }
    const q = pointAt(to);
    return [(q[0] - p[0]) / (to - from), (q[1] - p[1]) / (to - from)];
  };
  const breaksFlight = (before, after, degrees, speedRatio) => {
    const speed = Math.hypot(...before);
    const nextSpeed = Math.hypot(...after);
    if (speed >= 8 * scale && nextSpeed * speedRatio < speed) return true;
    if (Math.min(speed, nextSpeed) < 4 * scale) return false;
    const cosine = (before[0] * after[0] + before[1] * after[1]) / (speed * nextSpeed);
    return cosine < Math.cos(degrees * Math.PI / 180) ||
      Math.max(speed / nextSpeed, nextSpeed / speed) > speedRatio;
  };
  for (let frame = Math.max(apex + legFrames + 1, end - edgeWindow); frame < end; frame++) {
    const before = velocity(frame - legFrames, frame);
    const next = velocity(frame, frame + 1);
    if (!before || !next || before[1] <= 0) continue;
    if (frame + 1 === end) {
      if (breaksFlight(before, next, 50, 1.9)) return {start, end: frame};
      continue;
    }
    const after = velocity(frame, frame + 2);
    if (after && breaksFlight(before, next, 35, 1.65) && breaksFlight(before, after, 35, 1.65)) {
      return {start, end: frame};
    }
  }
  return {start, end};
}

function setAttackBallPath(frames, start, end) {
  const runs = [];
  let previous = null;
  let run = null;
  let edgeCount = 0;
  for (let frame = start; frame <= end; frame++) {
    const ann = frames[frame];
    const point = setAttackBallPoint(ann);
    if (!point) {
      previous = run = null;
      continue;
    }
    const derived = Boolean(ann.ball_interp || (ann.ball_source && ann.ball_source !== 'detected'));
    if (previous && !ann.ball_new) {
      const derivedEdge = previous.derived || derived;
      if (!run || run.derived !== derivedEdge) {
        run = {derived: derivedEdge, points: [previous.point]};
        runs.push(run);
      }
      run.points.push(point);
      edgeCount++;
    } else {
      run = null;
    }
    previous = {point, derived};
  }
  return {runs, complete: edgeCount === end - start};
}

function setAttackZone(point, netEnds) {
  const [left, right] = netEnds;
  if (!point || !Number.isFinite(left) || !Number.isFinite(right) || left === right) return 'unknown';
  const fraction = (point[0] - Math.min(left, right)) / Math.abs(right - left);
  if (fraction < 1 / 3) return 'left';
  if (fraction > 2 / 3) return 'right';
  return 'center';
}

function refreshSetTrajectories() {
  if (!annotations) return;
  const data = window._annData;
  const netEnds = [
    data.net_floor_left?.[0] ?? data.net_x_left,
    data.net_floor_right?.[0] ?? data.net_x_right,
  ];
  const setters = setAttackSetterIds(playerContactReport, EVENTS);
  setTrajectoryOverlay.pairs = setAttackPairs(EVENTS, eventTeams, RALLIES).map(pair => {
    const arc = setAttackArcBounds(annotations, pair.start, pair.end, FPS, data.frame_width);
    const {runs, complete} = setAttackBallPath(annotations, arc.start, arc.end);
    const attackPoint = setAttackBallPoint(annotations[arc.end]);
    const zone = setAttackZone(attackPoint, netEnds);
    const paths = {observed: new Path2D(), derived: new Path2D(), ends: attackPoint ? [attackPoint] : []};
    for (const {derived, points} of runs) {
      const path = derived ? paths.derived : paths.observed;
      path.moveTo(...points[0]);
      for (let i = 1; i < points.length; i++) path.lineTo(...points[i]);
    }
    return {
      ...pair, arc, zone, paths, runs, attackPoint, complete, hasPath: runs.length > 0,
      setter: setters.get(pair.setIndex) || null,
    };
  });
  renderSetTrajectories();
}

function renderSetTrajectorySetters(select, pairs, ranked, main) {
  const others = main ? pairs.filter(pair => pair.setter && pair.setter !== main).length : 0;
  const unknown = pairs.filter(pair => !pair.setter).length;
  const options = [['all', `All setters (${pairs.length})`]];
  if (others) options.push(['other', `Not the main setter (${others})`]);
  for (const [id, count] of ranked) {
    options.push([id, `${id}${id === main ? ' · main setter' : ''} (${count})`]);
  }
  if (unknown) options.push(['unknown', `Unidentified (${unknown})`]);
  const value = options.some(([id]) => id === select.value) ? select.value : 'all';
  select.replaceChildren(...options.map(([id, label]) => {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = label;
    return option;
  }));
  select.value = value;
}

function renderSetTrajectoryZones(box, pairs) {
  const state = setTrajectoryOverlay;
  const counts = {all: pairs.length};
  for (const pair of pairs) counts[pair.zone] = (counts[pair.zone] || 0) + 1;
  const zones = ['all', 'left', 'center', 'right'];
  if (counts.unknown || state.zone === 'unknown') zones.push('unknown');
  box.replaceChildren(...zones.map(zone => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip' + (state.zone === zone ? ' on' : '');
    chip.setAttribute('aria-pressed', String(state.zone === zone));
    if (zone !== 'all') {
      const swatch = document.createElement('span');
      swatch.className = 'set-trajectory-swatch';
      swatch.style.backgroundColor = SET_ATTACK_ZONES[zone].color;
      swatch.setAttribute('aria-hidden', 'true');
      chip.append(swatch);
    }
    chip.append(`${zone === 'all' ? 'All' : SET_ATTACK_ZONES[zone].label} ${counts[zone] || 0}`);
    chip.onclick = () => {
      state.zone = zone;
      renderSetTrajectories();
    };
    return chip;
  }));
}

function renderSetTrajectories() {
  if (!annotations) return;
  const state = setTrajectoryOverlay;
  const team = document.getElementById('setTrajectorySide').value;
  const setterSelect = document.getElementById('setTrajectorySetter');
  const sidePairs = state.pairs.filter(pair => setTrajectoryMatches(pair, {team}));
  const {ranked, main} = setAttackMainSetter(sidePairs);
  renderSetTrajectorySetters(setterSelect, sidePairs, ranked, main);
  const setter = setterSelect.value;
  const setterPairs = sidePairs.filter(pair => setTrajectoryMatches(pair, {setter}, main));
  renderSetTrajectoryZones(document.getElementById('setTrajectoryZones'), setterPairs);
  const pairs = setterPairs.filter(pair => setTrajectoryMatches(pair, {zone: state.zone}));
  const paths = new Map();
  const list = document.getElementById('setTrajectoryList');
  const overlayToggle = document.getElementById('togSetTrajectories');
  const timeLabel = frame => {
    const tenths = Math.floor(frame / FPS * 10);
    return `${formatTime(tenths / 10)}.${tenths % 10}`;
  };
  const selected = pairs.find(pair => pair.start === state.selected);
  state.selected = selected?.start ?? null;
  state.hovered = null;
  state.visible = pairs;
  const rows = [];
  let drawn = 0;
  for (const pair of pairs) {
    const {label, color} = SET_ATTACK_ZONES[pair.zone];
    if (pair.hasPath) {
      if (!paths.has(pair.zone)) paths.set(pair.zone, {observed: new Path2D(), derived: new Path2D(), ends: []});
      const group = paths.get(pair.zone);
      group.observed.addPath(pair.paths.observed);
      group.derived.addPath(pair.paths.derived);
      group.ends.push(...pair.paths.ends);
      drawn++;
    }
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'ctrl set-trajectory-row';
    row.dataset.start = String(pair.start);
    row.setAttribute('aria-pressed', String(pair === selected));
    row.classList.toggle('active', pair === selected);
    const side = pair.team === 'team_b' ? 'Near' : pair.team === 'team_a' ? 'Far' : 'Unattributed';
    const quality = pair.complete ? '' : pair.hasPath ? ' · partial' : ' · no path';
    const swatch = document.createElement('span');
    swatch.className = 'set-trajectory-swatch';
    swatch.style.backgroundColor = color;
    swatch.setAttribute('aria-hidden', 'true');
    row.append(swatch, `${timeLabel(pair.arc.start)} → ${timeLabel(pair.arc.end)} · ${side} · ${label}${quality}`);
    const setterTag = document.createElement('span');
    const otherSetter = Boolean(main && pair.setter && pair.setter !== main);
    setterTag.className = 'set-trajectory-setter' + (otherSetter ? ' other' : '');
    setterTag.textContent = pair.setter || 'P?';
    setterTag.title = !pair.setter ? 'Setter not identified'
      : otherSetter ? `Set by ${pair.setter}, not the main setter ${main}`
      : `Set by ${pair.setter}`;
    row.append(setterTag);
    row.onclick = () => selectSetTrajectory(pair);
    rows.push(row);
  }
  list.replaceChildren(...rows);
  state.paths = paths;
  overlayToggle.disabled = drawn === 0;
  // The share shows how firmly the side's sets point at one roster ID.
  const identified = ranked.reduce((total, [, count]) => total + count, 0);
  const mainNote = main ? ` · main setter ${main} (${ranked[0][1]} of ${identified} identified sets)` : '';
  document.getElementById('setTrajectoryStatus').textContent = pairs.length
    ? `${pairs.length} pairs · ${drawn} with a drawable path${mainNote}`
    : 'No matching pairs for these filters.';
  redraw();
}

// Selecting a pair plays the set, the attack and the seconds after it, then
// pauses. Passing null clears the selection and leaves playback alone.
function selectSetTrajectory(pair) {
  const state = setTrajectoryOverlay;
  const list = document.getElementById('setTrajectoryList');
  state.selected = pair ? pair.start : null;
  state.clip = null;
  for (const row of list.children) {
    const active = Boolean(pair) && Number(row.dataset.start) === pair.start;
    row.classList.toggle('active', active);
    row.setAttribute('aria-pressed', String(active));
    if (!active) continue;
    // Scroll inside the list only; scrollIntoView would move the whole page.
    const top = row.offsetTop - list.offsetTop;
    if (top < list.scrollTop || top + row.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = top - list.clientHeight / 2 + row.offsetHeight / 2;
    }
  }
  if (pair) {
    if (pair.hasPath) document.getElementById('togSetTrajectories').checked = true;
    const start = Math.max(0, pair.start - Math.round(SET_TRAJECTORY_LEAD_SECONDS * FPS));
    state.clip = {start, stop: pair.end + Math.round(SET_TRAJECTORY_TAIL_SECONDS * FPS)};
    seekAndPlay(start);
  }
  redraw();
}

// Pause at the end of the clip started by selectSetTrajectory. A seek outside
// the clip means the user moved on, so the pending pause is dropped.
function enforceSetTrajectoryClip() {
  const clip = setTrajectoryOverlay.clip;
  if (!clip) return;
  const frame = Math.round(vid.currentTime * FPS);
  if (frame < clip.start - 1 || frame > clip.stop + FPS) {
    setTrajectoryOverlay.clip = null;
  } else if (frame >= clip.stop) {
    setTrajectoryOverlay.clip = null;
    vid.pause();
  }
}

function setTrajectoriesShown() {
  const overlayToggle = document.getElementById('togSetTrajectories');
  return overlayToggle.checked && !overlayToggle.disabled && Boolean(setTrajectoryOverlay.paths);
}

function setTrajectoryAtEvent(event) {
  if (!setTrajectoriesShown()) return null;
  const rect = canvas.getBoundingClientRect();
  const frameWidth = Number(window._annData.frame_width || vid.videoWidth || 1280);
  // Ten CSS pixels around the pointer, expressed in annotation pixels.
  const tolerance = 10 * frameWidth / Math.max(rect.width, 1);
  const drawable = setTrajectoryOverlay.visible.filter(pair => pair.hasPath);
  return nearestSetTrajectory(drawable, eventImagePoint(event), tolerance);
}

function hoverSetTrajectory(pair) {
  const start = pair ? pair.start : null;
  canvas.style.cursor = pair ? 'pointer' : '';
  if (setTrajectoryOverlay.hovered === start) return;
  setTrajectoryOverlay.hovered = start;
  redraw();
}

function drawSetTrajectories() {
  const shown = setTrajectoriesShown();
  // The canvas takes pointer input only while there are lines to click.
  canvas.classList.toggle('trajectory-picking', shown);
  if (!shown) return;
  const data = window._annData;
  const sx = canvas.width / (data.frame_width || vid.videoWidth || 1280);
  const sy = canvas.height / (data.frame_height || vid.videoHeight || 720);
  const state = setTrajectoryOverlay;
  const selected = state.visible.find(pair => pair.start === state.selected);
  const hovered = state.visible.find(pair => pair.start === state.hovered);
  ctx.save();
  ctx.scale(sx, sy);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const stroke = (color, paths, width, opacity) => {
    ctx.fillStyle = ctx.strokeStyle = color;
    ctx.lineWidth = width / sx;
    ctx.globalAlpha = opacity;
    ctx.setLineDash([]);
    ctx.stroke(paths.observed);
    ctx.setLineDash([4 / sx, 4 / sx]);
    ctx.stroke(paths.derived);
    const radius = (width / 2 + 1.6) / sx;
    ctx.beginPath();
    for (const [x, y] of paths.ends) {
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
    }
    ctx.fill();
  };
  // A selected pair pushes every other path into the background.
  const baseOpacity = selected?.hasPath ? 0.2 : 0.8;
  for (const [zone, paths] of state.paths) {
    stroke(SET_ATTACK_ZONES[zone].color, paths, 2, baseOpacity);
  }
  if (hovered?.hasPath && hovered !== selected) {
    stroke(SET_ATTACK_ZONES[hovered.zone].color, hovered.paths, 3, 1);
  }
  if (selected?.hasPath) {
    stroke('#000', selected.paths, 7, 0.45);
    stroke(SET_ATTACK_ZONES[selected.zone].color, selected.paths, 4, 1);
  }
  ctx.restore();
}
