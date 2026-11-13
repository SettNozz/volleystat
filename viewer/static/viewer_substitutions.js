// Player substitutions: timeline markers, a video banner while the two players
// cross, IN/OUT highlights on their skeletons and a side-panel list.
// Classic script: declarations only. Reads the SUBSTITUTIONS boot constant
// (player_identity.substitutions from the stats JSON); the page calls
// renderSubstitutionPanel() once and the overlay calls drawSubstitutions().
// Detected lineup changes are automatic and unverified.

const SUBSTITUTION_LEAD_SEC = 2;   // banner shows this long before the crossing
const SUBSTITUTION_HOLD_SEC = 4;   // ... and this long after it
const SUBSTITUTION_IN = '#22c55e';
const SUBSTITUTION_OUT = '#f97316';

function substitutionEvents() {
  const block = (typeof SUBSTITUTIONS === 'object' && SUBSTITUTIONS) || {};
  return Array.isArray(block.events) ? block.events : [];
}

function substitutionsVisible() {
  const toggle = document.getElementById('togSubstitutions');
  return !toggle || toggle.checked;
}

function substitutionSpan(ev) {
  const [low, high] = Array.isArray(ev.window) ? ev.window : [ev.frame, ev.frame];
  return [low - SUBSTITUTION_LEAD_SEC * FPS, high + SUBSTITUTION_HOLD_SEC * FPS];
}

function activeSubstitution(frame) {
  for (const ev of substitutionEvents()) {
    const [start, end] = substitutionSpan(ev);
    if (frame >= start && frame <= end) return ev;
  }
  return null;
}

function substitutionLabel(ev) {
  return `${ev.player_in} in · ${ev.player_out} out`;
}

function setupSubstitutionMarkers(bar) {
  if (!bar) return;
  bar.querySelectorAll('.substitution-marker').forEach(m => m.remove());
  if (!substitutionsVisible()) return;
  for (const ev of substitutionEvents()) {
    const m = document.createElement('div');
    m.className = 'event-marker substitution-marker';
    m.style.left = (ev.frame / TOTAL_FRAMES * 100) + '%';
    m.title = `Substitution: ${substitutionLabel(ev)} @f${ev.frame} (automatic, unverified)`;
    m.onclick = (e) => { e.stopPropagation(); seekAndPlay(Math.max(0, substitutionSpan(ev)[0])); };
    bar.appendChild(m);
  }
}

function onSubstitutionToggle() {
  setupSubstitutionMarkers(document.getElementById('timelineBar'));
  redraw();
}

function renderSubstitutionPanel() {
  const panel = document.getElementById('substitutionPanel');
  if (!panel) return;
  const block = (typeof SUBSTITUTIONS === 'object' && SUBSTITUTIONS) || {};
  const events = substitutionEvents();
  const list = document.getElementById('substitutionList');
  const status = document.getElementById('substitutionStatus');
  document.getElementById('substitutionCount').textContent = events.length ? `(${events.length})` : '';
  list.replaceChildren();
  if (!block.status) {
    status.textContent = 'This result has no substitution analysis; re-run identity to add it.';
    return;
  }
  if (!events.length) {
    status.textContent = block.status === 'disabled' ? 'Substitution detection was disabled for this run.'
      : 'No substitutions detected.';
    return;
  }
  status.textContent = 'Found automatically from who leaves and enters the court between rallies. Unverified.';
  for (const ev of events) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'ctrl substitution-row';
    row.dataset.frame = ev.frame;
    const [low, high] = ev.window || [ev.frame, ev.frame];
    row.innerHTML = `<b>${formatTime(ev.frame / FPS)}</b> ` +
      `<span style="color:${SUBSTITUTION_IN}">▲ ${ev.player_in}</span> ` +
      `<span style="color:${SUBSTITUTION_OUT}">▼ ${ev.player_out}</span>` +
      `<small>f${low}–f${high}</small>`;
    row.onclick = () => seekAndPlay(Math.max(0, substitutionSpan(ev)[0]));
    list.appendChild(row);
  }
}

function highlightActiveSubstitutionRow(ev) {
  document.querySelectorAll('.substitution-row').forEach(row =>
    row.classList.toggle('current', !!ev && Number(row.dataset.frame) === ev.frame));
}

function drawSubstitutions(ctx, ann, frame, px, py, sx) {
  const ev = substitutionsVisible() ? activeSubstitution(frame) : null;
  highlightActiveSubstitutionRow(ev);
  if (!ev) return;
  // Skeleton highlights only on the frame the annotation belongs to.
  if (Number(ann.f) === frame) {
    for (const pose of ann.poses || []) {
      const role = pose.player_id === ev.player_in ? 'IN' : pose.player_id === ev.player_out ? 'OUT' : null;
      if (!role || !pose.bb) continue;
      const color = role === 'IN' ? SUBSTITUTION_IN : SUBSTITUTION_OUT;
      const [x0, y0, x1, y1] = pose.bb;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(px(x0), py(y0), px(x1) - px(x0), py(y1) - py(y0));
      ctx.setLineDash([]);
      ctx.font = `bold ${Math.max(11, Math.round(14 * sx))}px system-ui`;
      ctx.fillStyle = color;
      ctx.fillText(`${role} ${pose.player_id}`, px(x0), py(y1) + Math.max(13, 16 * sx));
    }
  }
  const [start, end] = substitutionSpan(ev);
  const fade = Math.min(1, (frame - start) / FPS, (end - frame) / FPS);
  const text = `⇄ SUBSTITUTION · ${ev.player_in} IN · ${ev.player_out} OUT`;
  const fs = Math.round(26 * Math.max(sx, 0.6));
  ctx.font = `bold ${fs}px system-ui`;
  const width = ctx.measureText(text).width;
  const x = (ctx.canvas.width - width) / 2;
  // Below the FSM state line (y=70) and the event flash (y=115).
  const y = py(115) + 2 * fs;
  ctx.globalAlpha = Math.max(0.25, fade);
  ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
  ctx.fillRect(x - 14, y - fs - 6, width + 28, fs + 18);
  ctx.strokeStyle = SUBSTITUTION_IN;
  ctx.lineWidth = 2;
  ctx.strokeRect(x - 14, y - fs - 6, width + 28, fs + 18);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, x, y);
  ctx.globalAlpha = 1;
}
