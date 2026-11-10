// Match Stats page: filters, rally presets, stats and service-attempt tables.
// Moved verbatim from viewer/templates/viewer.html. Classic script: shares
// globals with the page boot constants and viewer_page.js. Declarations
// only; listeners and init live in viewer_page.js.

/* ── Match Stats page ────────────────────────────────── */
const tableState = { team: 'all', outcome: 'all', rally: 'all', tab: 'stats' };

function initTableFilters() {
  const mk = (boxId, label, val, key) => {
    const box = document.getElementById(boxId);
    const el = document.createElement('div');
    el.className = 'chip' + (tableState[key] === val ? ' on' : '');
    el.textContent = label;
    el.dataset.val = val;
    el.onclick = () => {
      tableState[key] = val;
      box.querySelectorAll('.chip').forEach(c => c.classList.toggle('on', c.dataset.val === val));
      renderTables();
    };
    box.appendChild(el);
  };
  mk('tableTeamChips', 'All', 'all', 'team');
  mk('tableTeamChips', 'Team A', 'team_a', 'team');
  mk('tableTeamChips', 'Team B', 'team_b', 'team');
  for (const o of ['all', 'success', 'failed', 'no_reception'])
    mk('tableOutcomeChips', o === 'all' ? 'All' : o.replace(/_/g, ' '), o, 'outcome');
}

function resetTableFilters() {
  tableState.team = 'all'; tableState.outcome = 'all'; tableState.rally = 'all';
  document.querySelectorAll('#tableTeamChips .chip').forEach(c => c.classList.toggle('on', c.dataset.val === 'all'));
  document.querySelectorAll('#tableOutcomeChips .chip').forEach(c => c.classList.toggle('on', c.dataset.val === 'all'));
  document.querySelectorAll('#rallyPresets .preset').forEach(p => p.classList.remove('on'));
  renderTables();
}

function setTableTab(t) {
  tableState.tab = t;
  document.getElementById('tabStats').classList.toggle('on', t === 'stats');
  const attemptsTab = document.getElementById('tabAttempts');
  const attemptsPanel = document.getElementById('attemptsTabPanel');
  if (attemptsTab) attemptsTab.classList.toggle('on', t === 'attempts');
  if (attemptsPanel) attemptsPanel.style.display = t === 'attempts' ? '' : 'none';
  document.getElementById('tabScoring').classList.toggle('on', t === 'scoring');
  document.getElementById('statsTabPanel').style.display = t === 'stats' ? '' : 'none';
  document.getElementById('scoringTabPanel').style.display = t === 'scoring' ? '' : 'none';
}

function renderRallyPresets() {
  const box = document.getElementById('rallyPresets');
  box.innerHTML = '';
  RALLIES.forEach((r, i) => {
    const sr = r.serve_reception || {};
    const outcome = sr.outcome || '—';
    const el = document.createElement('div');
    el.className = 'preset' + (tableState.rally === String(i) ? ' on' : '');
    el.innerHTML = `<div class="p-label">Rally ${i + 1}</div>` +
      `<div class="p-score" style="color:${isSuccessfulOutcome(outcome) ? GREEN : isFailedOutcome(outcome) ? RED : 'var(--muted)'}">${r.duration_sec}s · ${outcome}</div>`;
    el.onclick = () => {
      tableState.rally = tableState.rally === String(i) ? 'all' : String(i);
      box.querySelectorAll('.preset').forEach((p, j) => p.classList.toggle('on', tableState.rally === String(j)));
      renderTables();
    };
    box.appendChild(el);
  });
}

function rallyRowVisible(r, i) {
  if (tableState.rally !== 'all' && String(i) !== tableState.rally) return false;
  const o = (r.serve_reception && r.serve_reception.outcome) || '';
  if (tableState.outcome !== 'all' && o !== tableState.outcome) return false;
  return true;
}

function renderTables() {
  renderRallyPresets();
  renderServiceAttempts();
  // Rally table
  const body = document.getElementById('rallyTableBody');
  body.innerHTML = '';
  let sumA = 0, sumB = 0, sumEv = 0, shown = 0;
  RALLIES.forEach((r, i) => {
    if (!rallyRowVisible(r, i)) return;
    shown++;
    sumA += r.contacts_team_a || 0; sumB += r.contacts_team_b || 0; sumEv += (r.events || []).length;
    const sr = r.serve_reception || {};
    const serveTeam = sr.serve ? teamLabel(sr.serve.team) : '—';
    const recTeam = sr.reception ? teamLabel(sr.reception.team) : '—';
    const outcome = sr.outcome || '—';
    const oColor = isSuccessfulOutcome(outcome) ? 'c-green' : (isFailedOutcome(outcome) ? 'c-red' : 'c-muted');
    const startFrame = rallyStartFrame(r);
    body.insertAdjacentHTML('beforeend',
      `<tr class="clickable" onclick="showPage('video');seekAndPlay(${startFrame})">` +
      `<td class="strong">${i + 1}</td><td>${formatTime(startFrame / FPS)}</td><td class="num">${r.duration_sec}s</td>` +
      `<td class="num bl">${r.contacts_team_a ?? '—'}</td><td class="num">${r.contacts_team_b ?? '—'}</td>` +
      `<td class="bl">${serveTeam}</td><td>${recTeam}</td><td class="${oColor}" style="font-weight:700">${outcome.replace(/_/g, ' ')}</td>` +
      `<td class="num bl">${(r.events || []).length}</td></tr>`);
  });
  body.insertAdjacentHTML('beforeend',
    `<tr class="total"><td></td><td>${shown} rallies</td><td></td>` +
    `<td class="num bl">${sumA}</td><td class="num">${sumB}</td>` +
    `<td class="bl"></td><td></td><td></td><td class="num bl">${sumEv}</td></tr>`);

  // Contact log
  const log = document.getElementById('scoringTableBody');
  log.innerHTML = '';
  EVENTS.forEach(ev => {
    if (tableState.team !== 'all' && ev.team !== tableState.team) return;
    const ri = rallyIndexOf(ev);
    if (tableState.rally !== 'all' && ri !== Number(tableState.rally)) return;
    if (tableState.outcome !== 'all') {
      const o = ri >= 0 && RALLIES[ri].serve_reception ? RALLIES[ri].serve_reception.outcome : '';
      if (o !== tableState.outcome) return;
    }
    log.insertAdjacentHTML('beforeend',
      `<tr class="clickable" onclick="showPage('video');seekAndPlay(${ev.frame})">` +
      `<td>${ri >= 0 ? ri + 1 : '—'}</td><td>${ev.time.toFixed(2)}s</td>` +
      `<td style="color:${evColor(ev.type)};font-weight:700">${ev.type.replace(/_/g, ' ')}</td>` +
      `<td>${ev.team === 'team_a' ? 'Team A'
            : (ev.team === 'team_b' ? 'Team B' : '—')}</td>` +
      `<td class="c-muted">${(ev.rally_phase || '—').replace(/_/g, ' ')}</td>` +
      `<td class="c-muted">${(ev.game_state || '—').replace(/_/g, ' ')}</td></tr>`);
  });
}

function renderServiceAttempts() {
  const body = document.getElementById('attemptsTableBody');
  if (!body) return;
  body.innerHTML = '';
  if (!SERVICE_ATTEMPTS.length) {
    body.innerHTML = '<tr><td colspan="8" class="c-muted">No service attempts recorded</td></tr>';
    return;
  }
  SERVICE_ATTEMPTS.forEach((attempt, i) => {
    const lc = attempt.lifecycle || {};
    const serveFrame = lifecycleFrame(attempt, 'serve_frame',
      lifecycleFrame(attempt, 'start_frame', attempt.start_frame));
    const status = lc.status || 'unconfirmed';
    const statusClass = status === 'playable_rally' ? 'c-green'
      : (status === 'reception_failed' || status === 'serve_fault' ? 'c-red' : 'c-muted');
    const frameCell = value => Number.isFinite(value) ? formatTime(value / FPS) : '—';
    const servingTeam = lc.serving_team === 'team_a' ? 'Team A'
      : (lc.serving_team === 'team_b' ? 'Team B' : '—');
    body.insertAdjacentHTML('beforeend',
      `<tr class="${Number.isFinite(serveFrame) ? 'clickable' : ''}"` +
      `${Number.isFinite(serveFrame) ? ` onclick="showPage('video');seekAndPlay(${serveFrame})"` : ''}>` +
      `<td class="strong">${i + 1}</td><td>${frameCell(serveFrame)}</td><td>${servingTeam}</td>` +
      `<td class="${statusClass}" style="font-weight:700">${status.replace(/_/g, ' ')}</td>` +
      `<td>${lc.serve_confirmed === true ? 'Yes' : 'No'}</td>` +
      `<td>${frameCell(lifecycleFrame(attempt, 'net_cross_frame'))}</td>` +
      `<td>${frameCell(lifecycleFrame(attempt, 'reception_frame'))}</td>` +
      `<td>${frameCell(lifecycleFrame(attempt, 'followup_frame'))}</td></tr>`);
  });
}

