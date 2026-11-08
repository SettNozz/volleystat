// Visual Stats page: court maps, attack vector map, contact and S/R tables.
// Moved verbatim from viewer/templates/viewer.html. Classic script: shares
// globals with the page boot constants and viewer_page.js. Declarations
// only; listeners and init live in viewer_page.js.

/* ── Charts page ─────────────────────────────────────── */
const chartState = { team: 'all', rally: 'all' };
let chartsBuilt = false;

function initChartFilters() {
  const teamBox = document.getElementById('chartTeamChips');
  const mkChip = (box, label, val, stateKey) => {
    const el = document.createElement('div');
    el.className = 'chip' + (chartState[stateKey] === val ? ' on' : '');
    el.textContent = label;
    el.dataset.val = val;
    el.onclick = () => {
      chartState[stateKey] = val;
      box.querySelectorAll('.chip').forEach(c => c.classList.toggle('on', c.dataset.val === val));
      renderCharts(true);
    };
    box.appendChild(el);
  };
  mkChip(teamBox, 'All', 'all', 'team');
  mkChip(teamBox, 'Team A', 'team_a', 'team');
  mkChip(teamBox, 'Team B', 'team_b', 'team');
  const rallyBox = document.getElementById('chartRallyChips');
  mkChip(rallyBox, 'All', 'all', 'rally');
  RALLIES.forEach((r, i) => mkChip(rallyBox, String(i + 1), String(i), 'rally'));
}

function ballPosNear(frame, tol = 8) {
  if (!annotations) return null;
  for (let d = 0; d <= tol; d++) {
    for (const fi of [frame + d, frame - d]) {
      const a = annotations[fi];
      if (a && a.ball) return a.ball;
    }
  }
  return null;
}

// The contact spotter emits no team, so ``ev.team`` can be null. Rendering
// that as a side would invent information the pipeline never measured.
function teamLabel(team) { return team === 'team_a' ? 'A' : (team === 'team_b' ? 'B' : '—'); }
function teamClass(team) { return team === 'team_a' ? 'team-a' : (team === 'team_b' ? 'team-b' : 'team-none'); }
// Spotter rallies carry no side split (``contacts_team_a`` is null), so they
// show the contact count instead of an invented A/B division.
function rallyContactsLabel(r) {
  return Number.isFinite(r.contacts_team_a) && Number.isFinite(r.contacts_team_b)
    ? `A:${r.contacts_team_a} B:${r.contacts_team_b}`
    : `${(r.events || []).length} contacts`;
}

function rallyIndexOf(ev) {
  for (let i = 0; i < RALLIES.length; i++) {
    const r = RALLIES[i];
    if (r.events && r.events.some(e => e.frame === ev.frame && e.type === ev.type)) return i;
  }
  return -1;
}

function chartEvVisible(ev) {
  if (chartState.team !== 'all' && ev.team !== chartState.team) return false;
  if (chartState.rally !== 'all' && rallyIndexOf(ev) !== Number(chartState.rally)) return false;
  return true;
}

function renderCourtMap(panelId, evs, opts = {}) {
  const panel = document.getElementById(panelId);
  panel.innerHTML = '';
  const d = window._annData || {};
  const fw = d.frame_width || 1280, fh = d.frame_height || 720;
  if (d.net_y) {
    const line = document.createElement('div');
    line.className = 'court-netline';
    line.style.top = (d.net_y / fh * 100) + '%';
    panel.appendChild(line);
  }
  const svgLines = [];
  const dots = [];
  for (const ev of evs) {
    const pos = ballPosNear(ev.frame);
    if (!pos) continue;
    const x = pos[0] / fw * 100, y = pos[1] / fh * 100;
    let color = INK;
    if (opts.colorByOutcome) {
      const ri = rallyIndexOf(ev);
      const out = ri >= 0 && RALLIES[ri].serve_reception ? RALLIES[ri].serve_reception.outcome : null;
      color = out === 'success' ? GREEN : (out ? RED : INK);
    } else if (opts.colorByTeam) {
      color = ev.team === 'team_a' ? 'oklch(58% 0.19 265)'
            : (ev.team === 'team_b' ? RED : INK);
    }
    if (opts.approachLines) {
      const from = ballPosNear(ev.frame - 8);
      if (from) svgLines.push([from[0] / fw * 100, from[1] / fh * 100, x, y, color]);
    }
    dots.push([x, y, color, ev]);
  }
  if (svgLines.length) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
    for (const [x1, y1, x2, y2, c] of svgLines) {
      const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1);
      l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('stroke', c); l.setAttribute('stroke-width', '0.4');
      l.setAttribute('opacity', '0.7'); l.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.appendChild(l);
    }
    panel.appendChild(svg);
  }
  for (const [x, y, color, ev] of dots) {
    const el = document.createElement('div');
    el.className = 'court-dot';
    el.style.cssText = `left:${x}%;top:${y}%;background:${color};cursor:pointer`;
    el.title = `${ev.type} · ${teamLabel(ev.team)} @ ${ev.time.toFixed(1)}s`;
    el.onclick = () => { showPage('video'); seekAndPlay(ev.frame); };
    panel.appendChild(el);
  }
  if (!dots.length) {
    const empty = document.createElement('div');
    empty.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12px;color:var(--muted)';
    empty.textContent = annotations ? 'No events for this filter' : 'Loading annotations…';
    panel.appendChild(empty);
  }
}

function renderCharts(force) {
  if (chartsBuilt && !force) { /* re-render is cheap; always refresh */ }
  chartsBuilt = true;
  const vis = EVENTS.filter(chartEvVisible);
  renderCourtMap('serveMap', vis.filter(e => e.type === 'serve'), { colorByOutcome: true });
  renderCourtMap('receptionMap', vis.filter(e => ['reception', 'reception_pass', 'dig'].includes(e.type)), { colorByOutcome: true });
  renderAttackVectorMap('attackMap', vis.filter(e => e.type === 'attack'));
  renderContactsTable(vis);
  renderSRTable();
}

/* Attack map geometry: the attacking half is mirrored ABOVE the net, so the
   three orange origin zones sit at the top and every shot arrow crosses the
   net downwards into the defending half. */
const MAP = {
  left: 12, right: 88, netY: 50, topY: 8, bottomY: 92,
  get width() { return this.right - this.left; },
  x(courtX) {
    const clamped = Math.max(-COURT_HALF_W, Math.min(COURT_HALF_W, Number(courtX) || 0));
    return 50 + clamped / COURT_HALF_W * (this.width / 2);
  },
  y(depth) {
    const value = Number(depth) || 0;
    const span = value >= 0 ? this.netY - this.topY : this.bottomY - this.netY;
    const clamped = Math.max(-COURT_HALF_L, Math.min(COURT_HALF_L, value));
    return this.netY - clamped / COURT_HALF_L * span;
  },
};

function renderAttackVectorMap(panelId, attackEvents) {
  const panel = document.getElementById(panelId);
  panel.innerHTML = '';
  const heatmap = (window._annData && window._annData.attack_heatmap) || {};
  const allowedFrames = new Set(attackEvents.map(e => e.frame));
  const hits = (heatmap.hits || []).filter(h => allowedFrames.has(h.frame));
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 100 100');
  // Angles carry the meaning here — a stretched court would turn a line shot
  // into something that reads as a diagonal.
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%';
  const add = (tag, attrs = {}, parent = svg) => {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    parent.appendChild(node);
    return node;
  };
  const label = (text, x, y, fill, size = 3, anchor = 'middle') => {
    const t = add('text', {x, y, fill, 'font-size': size, 'font-family': 'Inter, sans-serif', 'font-weight': 700, 'text-anchor': anchor});
    t.textContent = text;
    return t;
  };

  const attackLineY = MAP.y(COURT_ATTACK_LINE);
  add('rect', {x: MAP.left, y: MAP.topY, width: MAP.width, height: MAP.bottomY - MAP.topY, rx: 2, fill: '#f7fbff', stroke: '#94a3b8', 'stroke-width': .55});
  for (let i = 0; i < 3; i++) {
    const x = MAP.left + i * (MAP.width / 3);
    add('rect', {x, y: MAP.topY, width: MAP.width / 3, height: MAP.netY - MAP.topY, fill: 'rgba(245, 158, 11, 0.13)', stroke: 'rgba(245, 158, 11, 0.55)', 'stroke-width': .35});
    label(['LEFT', 'CENTER', 'RIGHT'][i], x + MAP.width / 6, MAP.topY + 4, '#a16207', 2.7);
  }
  // Front row is between the net and this line; anything behind it is a
  // back-row attack.
  add('line', {x1: MAP.left, y1: attackLineY, x2: MAP.right, y2: attackLineY, stroke: 'rgba(245, 158, 11, 0.75)', 'stroke-width': .5, 'stroke-dasharray': '2 1.5'});
  label('3 m', MAP.left + 2.5, attackLineY - 1, '#a16207', 2.3, 'start');
  add('line', {x1: MAP.left, y1: MAP.y(-COURT_ATTACK_LINE), x2: MAP.right, y2: MAP.y(-COURT_ATTACK_LINE), stroke: '#cbd5e1', 'stroke-width': .45, 'stroke-dasharray': '2 1.5'});
  add('line', {x1: MAP.left, y1: MAP.netY, x2: MAP.right, y2: MAP.netY, stroke: '#f59e0b', 'stroke-width': 1.2});
  label('NET', 94, MAP.netY + 1, '#a16207', 2.8);
  label('ATTACKING SIDE (mirrored above the net)', 50, MAP.topY - 2.5, '#a16207', 2.6);
  label('WHERE THE BALL WENT', 50, MAP.netY + 6, 'rgba(100, 116, 139, 0.65)', 2.6);

  const defs = add('defs');
  for (const [id, color] of [['attackArrowA', '#4f46e5'], ['attackArrowB', '#dc2626']]) {
    const marker = add('marker', {
      id, viewBox: '0 0 10 10', refX: 9, refY: 5,
      markerWidth: 3.4, markerHeight: 3.4, orient: 'auto-start-reverse',
    }, defs);
    add('path', {d: 'M 0 0 L 10 5 L 0 10 z', fill: color}, marker);
  }

  let noTarget = 0;
  for (const hit of hits) {
    const origin = hit.origin_court;
    if (!origin) continue;
    const x = MAP.x(origin[0]);
    const y = MAP.y(Math.abs(Number(origin[1]) || 0));
    const color = hit.team === 'team_a' ? '#4f46e5' : '#dc2626';
    const target = hit.target;
    const direction = hit.direction;
    const openAttack = () => { showPage('video'); seekAndPlay(hit.frame); };

    if (target && target.court) {
      const estimate = target.source === 'net_crossing_estimate';
      const line = add('line', {
        x1: x, y1: y, x2: MAP.x(target.court[0]), y2: MAP.y(target.court[1]),
        stroke: color, 'stroke-width': estimate ? .8 : 1.15,
        'stroke-dasharray': estimate ? '2 1.6' : 'none',
        'marker-end': `url(#${hit.team === 'team_a' ? 'attackArrowA' : 'attackArrowB'})`,
        opacity: estimate ? .6 : .92,
      });
      line.style.cursor = 'pointer';
      line.addEventListener('click', openAttack);
    } else {
      noTarget += 1;
    }

    // Front-row attacks are solid, back-row attacks are rings: the row is as
    // much a part of the shot as the zone.
    const back = hit.row === 'back';
    const point = add('circle', {
      cx: x, cy: y, r: back ? 1.9 : 1.7,
      fill: back ? '#fff' : color, stroke: color, 'stroke-width': back ? .9 : .5,
    });
    const title = document.createElementNS(NS, 'title');
    title.textContent = [
      `${hit.zone || 'zone'} · ${back ? 'back row' : 'front row'}`,
      direction ? `${direction.kind} (${direction.angle_deg}° off the net normal)` : 'direction not measured',
      target ? (target.source === 'defender_contact'
        ? `target: ${target.event_type || 'contact'} by the defence`
        : 'target: net crossing (estimate)') : 'no target found',
      `${Number(hit.time || 0).toFixed(1)}s`,
    ].join(' · ');
    point.appendChild(title);
    point.style.cursor = 'pointer';
    point.addEventListener('click', openAttack);
  }

  panel.appendChild(svg);

  const directions = heatmap.directions || {};
  const legend = document.createElement('div');
  legend.style.cssText = 'position:absolute;right:8px;bottom:6px;max-width:calc(100% - 16px);text-align:right;font-size:10.5px;line-height:1.5;color:var(--muted);background:rgba(255,255,255,.82);border-radius:5px;padding:3px 6px';
  legend.innerHTML = hits.length
    ? `<span style="color:#dc2626">●</span> team B &nbsp; <span style="color:#4f46e5">●</span> team A &nbsp; ○ back row<br>`
      + `${directions.line || 0} line · ${directions.diagonal || 0} diagonal`
      + (noTarget ? ` · ${noTarget} without a measured target` : '')
      + `<br><span style="opacity:.75">dashed arrow = net crossing estimate, no defender touch</span>`
    : '';
  panel.appendChild(legend);

  if (!hits.length) {
    const empty = document.createElement('div');
    empty.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:0 18px;font-size:12px;color:var(--muted)';
    empty.textContent = !annotations ? 'Loading annotations…'
      : ((heatmap.hits || []).length
        ? 'Attacks exist but are hidden by the event/team filter'
        : 'No attack origins. If the court outline on the video is wrong, fix it under Calibration & sync — the map is rebuilt from it.');
    panel.appendChild(empty);
  }
}

function renderContactsTable(vis) {
  const counts = {};
  for (const ev of vis) {
    counts[ev.type] = counts[ev.type] || { team_a: 0, team_b: 0 };
    counts[ev.type][ev.team] = (counts[ev.type][ev.team] || 0) + 1;
  }
  const body = document.getElementById('contactsTable');
  body.innerHTML = '';
  let ta = 0, tb = 0;
  for (const t of Object.keys(counts)) {
    ta += counts[t].team_a; tb += counts[t].team_b;
    body.insertAdjacentHTML('beforeend',
      `<tr><td>${t.replace(/_/g, ' ')}</td><td class="num">${counts[t].team_a}</td><td class="num">${counts[t].team_b}</td></tr>`);
  }
  body.insertAdjacentHTML('beforeend',
    `<tr><td class="strong">Total</td><td class="num strong">${ta}</td><td class="num strong">${tb}</td></tr>`);
}

function renderSRTable() {
  if (LIFECYCLE.available) {
    const body = document.getElementById('srTable');
    body.innerHTML = '';
    const total = LIFECYCLE.service_attempts || 1;
    const rows = [
      ['Service attempts', LIFECYCLE.service_attempts, INK],
      ['Confirmed serves', LIFECYCLE.confirmed_serves, '#eab308'],
      ['Inferred serves', LIFECYCLE.inferred_serves, '#eab308'],
      ['Playable rallies', LIFECYCLE.playable_rallies, GREEN],
      ['Serves not played (out / net / untouched)', LIFECYCLE.serves_not_played, 'var(--muted)'],
      ['Aces / no reception', LIFECYCLE.aces_no_reception, '#facc15'],
      ['Reception failed', LIFECYCLE.reception_failed, RED],
      ['Serve faults', LIFECYCLE.serve_faults, 'var(--muted)'],
    ];
    for (const [label, value, color] of rows) {
      // A counter the producer did not measure is absent, not zero.
      if (value === null || value === undefined) continue;
      body.insertAdjacentHTML('beforeend',
        `<tr><td style="color:${color};font-weight:700">${label}</td>` +
        `<td class="num">${value}</td><td class="num">${(value / total * 100).toFixed(0)}%</td></tr>`);
    }
    return;
  }
  const counts = { success: 0, failed: 0, no_reception: 0 };
  for (const r of RALLIES) {
    const o = r.serve_reception && r.serve_reception.outcome;
    if (o in counts) counts[o]++;
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const body = document.getElementById('srTable');
  body.innerHTML = '';
  const colors = { success: GREEN, failed: RED, no_reception: 'var(--muted)' };
  for (const [k, v] of Object.entries(counts)) {
    body.insertAdjacentHTML('beforeend',
      `<tr><td style="color:${colors[k]};font-weight:700">${k.replace(/_/g, ' ')}</td>` +
      `<td class="num">${v}</td><td class="num">${(v / total * 100).toFixed(0)}%</td></tr>`);
  }
}

