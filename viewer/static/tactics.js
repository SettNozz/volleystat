/* Tactical evidence uses saved contact/point provenance; rotations are manual. */
'use strict';

function tacticsMatches(row, filters, pair = true) {
  const values = {set: row.set_id, team: row.team_id, rotation: row.rotation};
  if (pair) Object.assign(values, {setter: row.setter, zone: row.zone, quality: row.reception_quality});
  return Object.entries(values).every(([key, value]) => !filters[key] || filters[key] === 'all' ||
    String(value ?? 'unknown') === filters[key]);
}

function tacticsRotationSummary(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = JSON.stringify([row.team_id, row.set_id, row.rotation]);
    if (!groups.has(key)) groups.set(key, {...row, side_out: {wins: 0, classified: 0, unknown: 0},
      break_point: {wins: 0, classified: 0, unknown: 0}, unknown_roles: 0});
    const group = groups.get(key);
    if (row.role !== 'serve' && row.role !== 'receive') { group.unknown_roles++; continue; }
    const metric = group[row.role === 'receive' ? 'side_out' : 'break_point'];
    if (row.won === null) metric.unknown++;
    else { metric.classified++; metric.wins += Number(row.won); }
  }
  return [...groups.values()];
}

async function initTactics() {
  const boot = JSON.parse(document.getElementById('tactics-data').textContent);
  const byId = id => document.getElementById(id);
  const video = byId('tactics-video');
  const status = byId('tactics-status');
  let data, shown = [], selected = -1, stopAt = null;
  const filterIds = ['set', 'team', 'setter', 'zone', 'quality', 'rotation'];
  const fmt = frame => (frame / boot.fps).toFixed(2) + 's';
  const unknown = value => value ?? 'Unknown';
  function element(tag, text) { const node = document.createElement(tag); node.textContent = text; return node; }
  function row(cells) { const tr = document.createElement('tr'); cells.forEach(text => tr.append(element('td', text))); return tr; }
  function filters() { return Object.fromEntries(filterIds.map(id => [id, byId('tactics-' + id).value])); }
  function play(start, end) {
    video.pause(); stopAt = Number.isFinite(end) ? end / boot.fps : null;
    video.currentTime = Math.max(0, start / boot.fps); video.play().catch(() => {});
  }
  function select(index, full = false, autoplay = true) {
    if (index < 0 || index >= shown.length) return;
    selected = index;
    const pair = shown[index];
    const start = full ? pair.rally_start_frame : Math.max(pair.rally_start_frame, pair.start_frame - .5 * boot.fps);
    const end = full ? pair.rally_end_frame : Math.min(pair.rally_end_frame, pair.end_frame + 2 * boot.fps);
    for (const [i, button] of Array.from(byId('tactics-pairs').querySelectorAll('button')).entries()) button.setAttribute('aria-current', String(i === index));
    byId('tactics-now').textContent = `${index + 1}/${shown.length} · ${pair.setter || 'Unidentified setter'} · ${fmt(pair.start_frame)} → ${fmt(pair.end_frame)} · ${pair.ball_provenance} ball`;
    byId('tactics-prev').disabled = index === 0; byId('tactics-next').disabled = index === shown.length - 1;
    byId('tactics-rally').disabled = false;
    const link = byId('tactics-viewer'); link.hidden = false; link.href = boot.viewer_url + '?frame=' + pair.end_frame;
    if (autoplay) play(start, end);
  }
  function render() {
    const current = filters();
    shown = data.pairs.filter(pair => tacticsMatches(pair, current)); selected = -1;
    const list = byId('tactics-pairs'); list.replaceChildren();
    shown.forEach((pair, index) => {
      const button = element('button', `${fmt(pair.start_frame)} → ${fmt(pair.end_frame)} · ${pair.setter || 'P?'} · ${unknown(pair.team_id)} · ${pair.zone}`);
      button.type = 'button'; button.className = 'tactics-pair';
      button.append(element('small', `Reception: ${pair.reception_quality} · ${pair.rotation ? 'R' + pair.rotation : 'Unlabelled rotation'} · ${pair.setter_source} author · ${pair.ball_provenance} ball`));
      button.onclick = () => select(index); list.append(button);
    });
    if (!shown.length) list.textContent = 'No fully supported set-to-attack pairs match these filters.';
    byId('tactics-prev').disabled = byId('tactics-next').disabled = byId('tactics-rally').disabled = true;
    byId('tactics-viewer').hidden = true;
    byId('tactics-now').textContent = 'Choose a set-to-attack pair.';
    if (shown.length) select(0, false, false);
    const reasons = Object.entries(data.summary.rejection_reasons).map(([key, count]) => `${count} ${key.replace(/_/g, ' ')}`).join('; ');
    byId('tactics-coverage').textContent = `${shown.length} filtered pairs · ${data.summary.accepted_pairs}/${data.summary.candidate_pairs} supported candidates · ${data.summary.unidentified_setters} unidentified setters · ${data.summary.unknown_teams} unknown teams · ${data.summary.points_without_team_scope || 0} points without team scope` + (reasons ? ' · Excluded: ' + reasons : '');
    const distribution = byId('tactics-distribution'); distribution.replaceChildren();
    const groups = new Map(), totals = new Map();
    for (const pair of shown) {
      const key = JSON.stringify([pair.team_id, pair.setter, pair.zone, pair.reception_quality]);
      const setterKey = JSON.stringify([pair.team_id, pair.setter]);
      if (!groups.has(key)) groups.set(key, {pair, count: 0});
      groups.get(key).count++; totals.set(setterKey, (totals.get(setterKey) || 0) + 1);
    }
    for (const {pair, count} of groups.values()) {
      const tr = row([unknown(pair.team_id), pair.setter || 'P?', pair.zone, pair.reception_quality, count,
        (100 * count / totals.get(JSON.stringify([pair.team_id, pair.setter]))).toFixed(1) + '%']);
      const button = element('button', String(count)); button.type = 'button';
      button.setAttribute('aria-label', `Watch ${count} matching pairs for ${pair.setter || 'unknown setter'}, ${pair.zone}, ${pair.reception_quality}`);
      button.onclick = () => {
        byId('tactics-team').value = pair.team_id || 'unknown';
        byId('tactics-setter').value = pair.setter || 'unknown'; byId('tactics-zone').value = pair.zone;
        byId('tactics-quality').value = pair.reception_quality; render(); if (shown.length) select(0);
      };
      tr.children[4].replaceChildren(button); distribution.append(tr);
    }
    if (!groups.size) distribution.append(row(['No supported pairs', '—', '—', '—', 0, '—']));
    const rallies = data.rotation_rallies.filter(rally => tacticsMatches(rally, current, false));
    const rotationTable = byId('tactics-rotations'); rotationTable.replaceChildren();
    const rate = metric => metric.classified ? `${metric.wins}/${metric.classified} · ${(100 * metric.wins / metric.classified).toFixed(1)}%` : '— · 0 classified';
    for (const group of tacticsRotationSummary(rallies)) rotationTable.append(row([group.team_id, unknown(group.set_id), group.rotation ? 'R' + group.rotation : 'Unlabelled',
      rate(group.side_out), group.side_out.unknown, rate(group.break_point), group.break_point.unknown, group.unknown_roles]));
    const rallyList = byId('rotation-rallies'); rallyList.replaceChildren();
    for (const rally of rallies) {
      const result = rally.evidence_issue ? 'Excluded from rate: ' + rally.evidence_issue.replace(/_/g, ' ') : rally.won === null ? 'Point unresolved' : rally.won ? 'Point won' : 'Point lost';
      const button = element('button', `${fmt(rally.start_frame)} · ${rally.team_id} · ${rally.rotation ? 'R' + rally.rotation : 'Unlabelled'} · ${rally.role} · ${result} · ${rally.point_verified ? 'Human-confirmed winner' : 'Automatic point evidence'}`);
      button.type = 'button'; button.className = 'tactics-pair'; button.onclick = () => play(rally.start_frame, rally.end_frame); rallyList.append(button);
    }
    if (!rallies.length) { rotationTable.append(row(['No point evidence', '—', '—', '—', 0, '—', 0, 0])); rallyList.textContent = 'No source-linked point evidence matches these filters.'; }
  }
  function options(id, values, label, includeUnknown = true) {
    const select = byId(id), previous = select.value;
    select.replaceChildren();
    for (const [value, text] of [['all', label], ...values.map(value => [value, value]), ...(includeUnknown ? [['unknown', 'Unknown']] : [])]) {
      const option = element('option', text); option.value = value; select.append(option);
    }
    if (Array.from(select.options).some(option => option.value === previous)) select.value = previous;
  }
  function renderLabels() {
    const list = byId('rotation-labels'); list.replaceChildren();
    (data.config.rotations || []).forEach((label, index) => {
      const item = element('li', `${label.team_id} · R${label.rotation} · ${fmt(label.start_frame)}–${fmt(label.end_frame)}`);
      const remove = element('button', 'Remove label'); remove.type = 'button';
      remove.onclick = async () => { const config = structuredClone(data.config); config.rotations.splice(index, 1); await saveConfig(config); };
      item.append(remove); list.append(item);
    });
    const select = byId('rotation-team'); select.replaceChildren();
    const ids = new Set(data.options.teams);
    for (const value of Object.values(data.config.team_identity_map || {})) if (value) ids.add(value);
    for (const segment of data.config.segments || []) for (const value of Object.values(segment.team_identity_map || {})) if (value) ids.add(value);
    for (const id of ids) { const option = element('option', id); option.value = id; select.append(option); }
    byId('rotation-save').disabled = !ids.size;
    if (!ids.size) byId('rotation-status').textContent = 'Confirm stable team identities in Match setup before labelling rotations.';
  }
  async function load() {
    const response = await fetch(boot.api_url); const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not load tactics');
    data = payload; data.config ||= {};
    options('tactics-set', data.options.sets, 'All sets'); options('tactics-team', data.options.teams, 'All teams');
    options('tactics-setter', data.options.setters, 'All setters');
    byId('tactics-mapping-source').textContent = data.context_source === 'human_workspace'
      ? 'Using user-saved match setup. Team mapping is configuration; it does not verify contact attribution.'
      : 'Using saved/default team mapping. Verify team names and side switches in Match setup before comparing sets.';
    render(); renderLabels(); status.textContent = 'Saved evidence only. Manual rotations do not change model predictions.'; status.dataset.error = 'false';
  }
  async function saveConfig(config) {
    byId('rotation-save').disabled = true;
    const message = byId('rotation-status'); message.textContent = 'Saving rotation labels…';
    try {
      const response = await fetch(boot.context_api_url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({source: data.source, revision: data.revision, config})});
      const payload = await response.json(); if (!response.ok) throw new Error(payload.error || 'Could not save rotation label');
      await load(); message.textContent = 'Rotation labels saved. Tactical counts refreshed.';
    } catch (error) { message.textContent = error.message; } finally { byId('rotation-save').disabled = !byId('rotation-team').options.length; }
  }
  byId('rotation-use-start').onclick = () => { byId('rotation-start').value = video.currentTime.toFixed(3); };
  byId('rotation-use-end').onclick = () => { byId('rotation-end').value = video.currentTime.toFixed(3); };
  byId('rotation-editor').onsubmit = async event => {
    event.preventDefault();
    const start = Math.round(Number(byId('rotation-start').value) * boot.fps), end = Math.round(Number(byId('rotation-end').value) * boot.fps);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || end >= boot.total_frames) { byId('rotation-status').textContent = 'Choose an ordered time range inside the video.'; return; }
    const config = structuredClone(data.config); config.rotations ||= [];
    config.rotations.push({start_frame: start, end_frame: end, team_id: byId('rotation-team').value, rotation: Number(byId('rotation-number').value)});
    await saveConfig(config);
  };
  byId('tactics-filters').onchange = render;
  byId('tactics-filters').onreset = () => { for (const id of filterIds) byId('tactics-' + id).value = 'all'; render(); };
  byId('tactics-prev').onclick = () => select(selected - 1); byId('tactics-next').onclick = () => select(selected + 1);
  byId('tactics-rally').onclick = () => select(selected, true);
  video.addEventListener('timeupdate', () => { if (!video.paused && stopAt !== null && video.currentTime >= stopAt) { video.pause(); stopAt = null; } });
  try { await load(); } catch (error) { status.textContent = error.message; status.dataset.error = 'true'; }
}

document.addEventListener('DOMContentLoaded', initTactics);
