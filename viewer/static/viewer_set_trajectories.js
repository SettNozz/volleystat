// Full-match set-to-attack overlays from saved contacts and active ball coordinates.
const SET_CONTACTS = new Set(['set', 'setting_pass', 'pass']);
const COURT_SIDES = new Set(['team_a', 'team_b']);
const SET_ATTACK_ZONES = {
  left: {label: 'Left', color: '#38bdf8'},
  center: {label: 'Center', color: '#facc15'},
  right: {label: 'Right', color: '#f472b6'},
  unknown: {label: 'Unknown zone', color: '#a1a1aa'},
};
const setTrajectoryOverlay = {pairs: [], selected: null, paths: null};

function setAttackPairs(events, teams, rallies) {
  const contacts = events.map((event, index) => ({event, team: teams[index]}))
    .filter(({event}) => Number.isInteger(event.frame))
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
    pairs.push({start, end, team: setTeam === attackTeam ? setTeam : null});
  }
  return pairs;
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
    return {...pair, arc, zone, paths, complete, hasPath: runs.length > 0};
  });
  renderSetTrajectories();
}

function renderSetTrajectories() {
  if (!annotations) return;
  const team = document.getElementById('setTrajectorySide').value;
  const pairs = setTrajectoryOverlay.pairs.filter(pair => team === 'all' || pair.team === team);
  const paths = new Map();
  const list = document.getElementById('setTrajectoryList');
  const overlayToggle = document.getElementById('togSetTrajectories');
  const timeLabel = frame => {
    const tenths = Math.floor(frame / FPS * 10);
    return `${formatTime(tenths / 10)}.${tenths % 10}`;
  };
  const selected = pairs.find(pair => pair.start === setTrajectoryOverlay.selected);
  setTrajectoryOverlay.selected = selected?.start ?? null;
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
    row.setAttribute('aria-pressed', String(pair === selected));
    row.classList.toggle('active', pair === selected);
    const side = pair.team === 'team_b' ? 'Near' : pair.team === 'team_a' ? 'Far' : 'Unattributed';
    const quality = pair.complete ? '' : pair.hasPath ? ' · partial' : ' · no path';
    const swatch = document.createElement('span');
    swatch.className = 'set-trajectory-swatch';
    swatch.style.backgroundColor = color;
    swatch.setAttribute('aria-hidden', 'true');
    row.append(swatch, `${timeLabel(pair.arc.start)} → ${timeLabel(pair.arc.end)} · ${side} · ${label}${quality}`);
    row.onclick = () => {
      setTrajectoryOverlay.selected = pair.start;
      for (const button of list.children) {
        button.classList.toggle('active', button === row);
        button.setAttribute('aria-pressed', String(button === row));
      }
      if (pair.hasPath) overlayToggle.checked = true;
      seekAndPlay(pair.arc.start, false);
      redraw();
    };
    rows.push(row);
  }
  list.replaceChildren(...rows);
  for (const swatch of document.querySelectorAll('#setTrajectoryLegend [data-zone]')) {
    swatch.style.backgroundColor = SET_ATTACK_ZONES[swatch.dataset.zone].color;
  }
  document.getElementById('setTrajectoryUnknown').hidden = !pairs.some(pair => pair.zone === 'unknown');
  setTrajectoryOverlay.paths = paths;
  overlayToggle.disabled = drawn === 0;
  document.getElementById('setTrajectoryStatus').textContent = pairs.length
    ? `${pairs.length} pairs · ${drawn} with a drawable path`
    : 'No matching pairs for this side.';
  redraw();
}

function drawSetTrajectories() {
  const overlayToggle = document.getElementById('togSetTrajectories');
  if (!overlayToggle.checked || overlayToggle.disabled || !setTrajectoryOverlay.paths) return;
  const data = window._annData;
  const sx = canvas.width / (data.frame_width || vid.videoWidth || 1280);
  const sy = canvas.height / (data.frame_height || vid.videoHeight || 720);
  const selected = setTrajectoryOverlay.pairs.find(pair => pair.start === setTrajectoryOverlay.selected);
  ctx.save();
  ctx.scale(sx, sy);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const stroke = (zone, paths, width, opacity) => {
    ctx.fillStyle = ctx.strokeStyle = SET_ATTACK_ZONES[zone].color;
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
  for (const [zone, paths] of setTrajectoryOverlay.paths) {
    stroke(zone, paths, 2, 0.8);
  }
  if (selected?.hasPath) stroke(selected.zone, selected.paths, 4, 1);
  ctx.restore();
}
