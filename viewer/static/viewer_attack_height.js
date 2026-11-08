// Transfer the net's height to the estimated depth of the attack using the
// court perspective. All points and directions here are in image coordinates.
let attackHeightEvents = [];

function attackCross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function attackDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function attackVertical(across, along, horizon, width, height) {
  const upright = [0, 1, 0];
  if (!(width > 0 && height > 0)) return upright;
  // Assume square pixels and a centered principal point. Fit focal length
  // using the court axes' orthogonality and equal scale in camera space.
  const cx = width / 2, cy = height / 2;
  const [ux, uy, uz] = [across[0] - cx * across[2], across[1] - cy * across[2], across[2]];
  const [vx, vy, vz] = [along[0] - cx * along[2], along[1] - cy * along[2], along[2]];
  const coeff = [2 * uz * vz, uz * uz - vz * vz];
  const target = [-2 * (ux * vx + uy * vy), vx * vx + vy * vy - ux * ux - uy * uy];
  const focalSquared = (coeff[0] * target[0] + coeff[1] * target[1]) / (coeff[0] ** 2 + coeff[1] ** 2);
  const residual = Math.hypot(coeff[0] * focalSquared - target[0], coeff[1] * focalSquared - target[1]);
  const fitScale = ux * ux + uy * uy + vx * vx + vy * vy;
  if (!(focalSquared > (width / 5) ** 2 && focalSquared < (4 * width) ** 2 &&
        residual < 0.15 * fitScale)) return upright;

  // The vertical vanishing point is the pole of the court horizon under K Kᵀ.
  const z = attackDot(horizon, [cx, cy, 1]);
  const y = focalSquared * horizon[1] + cy * z;
  if (Math.abs(y) < 1e-9) return upright;
  return [(focalSquared * horizon[0] + cx * z) / y, 1, z / y];
}

function attackHeightCalibration(data) {
  const h = courtHomography(data.court_corners);
  const netHeight = data.net_height_m;
  if (!h || !Number.isFinite(netHeight) || netHeight <= 0) return null;
  const left = data.net_floor_left, right = data.net_floor_right;
  const floor = left?.slice(0, 2).every(Number.isFinite) &&
    right?.slice(0, 2).every(Number.isFinite) && Math.abs(left[0] - right[0]) >= 5
    ? [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, 1]
    : [h[0][2], h[1][2], 1];
  const [leftY, rightY] = data.net_tape_endpoint_y || [];
  const span = data.net_x_right - data.net_x_left;
  const hasTapeEnds = span > 0 && Number.isFinite(leftY) && Number.isFinite(rightY);
  const slope = hasTapeEnds ? (rightY - leftY) / span : 0;
  const intercept = hasTapeEnds ? leftY - slope * data.net_x_left : data.net_tape_top ?? data.net_y;
  if (!Number.isFinite(intercept)) return null;
  const tapeLine = [-slope, 1, -intercept];
  const netPixels = attackDot(tapeLine, floor);
  if (netPixels < 5) return null;

  const across = h.map(row => row[0]);
  const along = h.map(row => row[1]);
  const horizon = attackCross(across, along);
  const netDepthScale = attackDot(horizon, floor);
  if (!Number.isFinite(netDepthScale) || Math.abs(netDepthScale) < 1e-9) return null;
  const vertical = attackVertical(across, along, horizon, data.frame_width, data.frame_height);
  const verticalToTape = attackDot(tapeLine, vertical);
  if (Math.abs(verticalToTape) < 1e-9) return null;
  const scale = -netPixels / (netHeight * verticalToTape);
  return {netHeight, across, horizon, netDepthScale, up: vertical.map(v => v * scale)};
}

function estimateAttackHeight(calibration, ball, floor) {
  if (!calibration || !ball || !floor) return null;
  const {netHeight, across, horizon, netDepthScale, up} = calibration;
  const floorImage = [floor[0], floor[1], 1];
  const depthRatio = attackDot(horizon, floorImage) / netDepthScale;
  if (depthRatio <= 0) return null;
  // At this floor depth, the line at net height runs along the court's
  // transverse axis. Solve the signed offset of the ball above that line.
  const netLevel = floorImage.map((v, i) => v / depthRatio + netHeight * up[i]);
  const contactLine = attackCross(across, [ball[0], ball[1], 1]);
  const verticalScale = attackDot(contactLine, up);
  if (Math.abs(verticalScale) < 1e-9) return null;
  const offset = -attackDot(contactLine, netLevel) / verticalScale;
  const height = netHeight + offset;
  return height > 0 && height <= 4.5 ? height : null;
}

function nearAttackPose(poses, ball, playerId) {
  const near = poses.filter(pose => pose.team === 'b' || pose.team === 'team_b');
  const assigned = playerId && near.find(pose => pose.player_id === playerId);
  if (assigned) return assigned;
  let best = null;
  let distance = 0.45;
  for (const pose of near) {
    const height = pose.bb?.[3] - pose.bb?.[1];
    if (!(height > 0)) continue;
    for (const wrist of (pose.kp || []).slice(9, 11)) {
      if (!(wrist?.[2] >= 0.3)) continue;
      const score = Math.hypot(wrist[0] - ball[0], wrist[1] - ball[1]) / height;
      if (score < distance) {
        best = pose;
        distance = score;
      }
    }
  }
  return best;
}

function attackFloorPoint(frames, frame, attacker, fps) {
  if (!attacker || (attacker.id == null && !attacker.player_id)) return null;
  const points = [];
  const radius = Math.round(fps * 0.55);
  for (let i = frame - radius; i <= frame + radius; i++) {
    const pose = frames[i]?.poses?.find(p =>
      attacker.id != null ? p.id === attacker.id : p.player_id === attacker.player_id);
    const feet = (pose?.kp || []).slice(15, 17)
      .filter(p => p?.[2] >= 0.3 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (feet.length) points.push(feet.reduce((lower, p) => p[1] > lower[1] ? p : lower));
  }
  // Lower visible ankles in the local window approximate floor contact.
  // The percentile avoids trusting a single downward keypoint outlier.
  if (points.length < 3) return null;
  points.sort((a, b) => a[1] - b[1]);
  return points[Math.floor((points.length - 1) * 0.8)];
}

function refreshAttackHeights() {
  if (!annotations) return;
  const calibration = attackHeightCalibration(window._annData);
  const report = playerContactReport;
  const assignments = new Map((report?.side === 'b' ? report.contacts : [])
    .filter(contact => contact.type === 'attack' && contact.status === 'assigned')
    .map(contact => [contact.event_index, contact]));
  attackHeightEvents = [];
  for (const [index, event] of EVENTS.entries()) {
    if (event.type !== 'attack' || !Number.isInteger(event.frame)) continue;
    const near = eventTeams[index] === 'team_b';
    let height = null;
    const ann = annotations[event.frame];
    const ball = setAttackBallPoint(ann);
    if (near && calibration && ball) {
      const assigned = assignments.get(index);
      const playerId = assigned?.frame === event.frame ? assigned.player_id : null;
      // Keep the report's raw track through takeoff and landing. The frame-level
      // court-side tag can flip when an airborne player's feet cross the divider.
      const candidate = playerId && assigned.candidates?.find(p =>
        p.player_id === playerId && p.raw_track_id != null);
      const attacker = candidate ? {id: candidate.raw_track_id} : nearAttackPose(ann.poses || [], ball, playerId);
      height = estimateAttackHeight(calibration, ball, attackFloorPoint(annotations, event.frame, attacker, FPS));
    }
    attackHeightEvents.push({frame: event.frame, near, height});
  }
  attackHeightEvents.sort((a, b) => a.frame - b.frame);
}

function renderAttackHeight() {
  const badge = document.getElementById('attackHeightBadge');
  if (!document.getElementById('togEvents').checked) {
    badge.hidden = true;
    return;
  }
  for (let i = attackHeightEvents.length - 1; i >= 0; i--) {
    const attack = attackHeightEvents[i];
    const age = currentFrameIdx - attack.frame;
    if (age < 0) continue;
    if (age >= 2 * FPS || !attack.near) break;
    const text = attack.height === null ? 'Attack · Height unavailable' : `Attack · ≈ ${attack.height.toFixed(1)} m`;
    if (badge.textContent !== text) badge.textContent = text;
    badge.hidden = false;
    return;
  }
  badge.hidden = true;
}
