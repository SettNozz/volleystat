/* ── Court geometry (mirrors src/tracking/court_setups.py) ──────────
   Solved in the browser so the guide lines follow the drag without a
   round-trip; the server recomputes the same matrix when saving. */
const COURT_HALF_W = 4.5, COURT_HALF_L = 9, COURT_ATTACK_LINE = 3;
const COURT_WORLD_CORNERS = [
  [-COURT_HALF_W, -COURT_HALF_L], [COURT_HALF_W, -COURT_HALF_L],
  [COURT_HALF_W, COURT_HALF_L], [-COURT_HALF_W, COURT_HALF_L],
];

function solveLinearSystem(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col] / a[col][col];
      for (let k = col; k <= n; k++) a[row][k] -= factor * a[col][k];
    }
  }
  return a.map((row, i) => row[n] / row[i]);
}

function courtHomography(corners) {
  if (!Array.isArray(corners) || corners.length !== 4) return null;
  const rows = [], rhs = [];
  for (let i = 0; i < 4; i++) {
    const [wx, wy] = COURT_WORLD_CORNERS[i];
    const ix = Number(corners[i][0]), iy = Number(corners[i][1]);
    if (!Number.isFinite(ix) || !Number.isFinite(iy)) return null;
    rows.push([wx, wy, 1, 0, 0, 0, -ix * wx, -ix * wy]); rhs.push(ix);
    rows.push([0, 0, 0, wx, wy, 1, -iy * wx, -iy * wy]); rhs.push(iy);
  }
  const solution = solveLinearSystem(rows, rhs);
  if (!solution) return null;
  const h = solution.concat([1]);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
}

function courtGuideLines(corners) {
  const H = courtHomography(corners);
  if (!H) return null;
  const project = (wx, wy) => {
    const w = H[2][0] * wx + H[2][1] * wy + H[2][2];
    if (Math.abs(w) < 1e-9) return null;
    return [
      (H[0][0] * wx + H[0][1] * wy + H[0][2]) / w,
      (H[1][0] * wx + H[1][1] * wy + H[1][2]) / w,
    ];
  };
  const segments = {
    center_line: [project(-COURT_HALF_W, 0), project(COURT_HALF_W, 0)],
    attack_line_near: [project(-COURT_HALF_W, -COURT_ATTACK_LINE), project(COURT_HALF_W, -COURT_ATTACK_LINE)],
    attack_line_far: [project(-COURT_HALF_W, COURT_ATTACK_LINE), project(COURT_HALF_W, COURT_ATTACK_LINE)],
  };
  for (const segment of Object.values(segments)) {
    if (!segment[0] || !segment[1]) return null;
  }
  return segments;
}

function courtCornersAreConvex(corners) {
  if (!Array.isArray(corners) || corners.length !== 4) return false;
  const signs = [];
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = corners[i], [bx, by] = corners[(i + 1) % 4], [cx, cy] = corners[(i + 2) % 4];
    signs.push((bx - ax) * (cy - by) - (by - ay) * (cx - bx));
  }
  return signs.every(s => s > 0) || signs.every(s => s < 0);
}


function clipSegmentToFrame(p, q, width, height, margin = 24) {
  // Liang–Barsky against the frame inset by a margin, so a handle always
  // stays grabbable inside the picture.
  const minX = margin, maxX = width - margin;
  const minY = margin, maxY = height - margin;
  let t0 = 0, t1 = 1;
  const dx = q[0] - p[0], dy = q[1] - p[1];
  const tests = [[-dx, p[0] - minX], [dx, maxX - p[0]], [-dy, p[1] - minY], [dy, maxY - p[1]]];
  for (const [pk, qk] of tests) {
    if (Math.abs(pk) < 1e-9) {
      if (qk < 0) return null;
      continue;
    }
    const r = qk / pk;
    if (pk < 0) { if (r > t1) return null; if (r > t0) t0 = r; }
    else { if (r < t0) return null; if (r < t1) t1 = r; }
  }
  return [
    [p[0] + t0 * dx, p[1] + t0 * dy],
    [p[0] + t1 * dx, p[1] + t1 * dy],
  ];
}

function courtEdgeAnchors(corners, width, height) {
  // Two handles per edge, placed where that edge is actually on screen.
  return corners.map((corner, index) => {
    const next = corners[(index + 1) % 4];
    const visible = clipSegmentToFrame(corner, next, width, height) || [corner, next];
    const lerp = t => [
      visible[0][0] + (visible[1][0] - visible[0][0]) * t,
      visible[0][1] + (visible[1][1] - visible[0][1]) * t,
    ];
    return [lerp(0.25), lerp(0.75)];
  });
}

function lineIntersection(a1, a2, b1, b2) {
  const d1x = a2[0] - a1[0], d1y = a2[1] - a1[1];
  const d2x = b2[0] - b1[0], d2y = b2[1] - b1[1];
  const denominator = d1x * d2y - d1y * d2x;
  if (Math.abs(denominator) < 1e-9) return null;  // parallel edges
  const t = ((b1[0] - a1[0]) * d2y - (b1[1] - a1[1]) * d2x) / denominator;
  return [a1[0] + t * d1x, a1[1] + t * d1y];
}

function cornersFromEdgeAnchors(anchors, fallback) {
  return fallback.map((corner, index) => {
    const previous = anchors[(index + 3) % 4];
    const current = anchors[index];
    const point = lineIntersection(previous[0], previous[1], current[0], current[1]);
    return point && point.every(Number.isFinite) ? point : corner;
  });
}
