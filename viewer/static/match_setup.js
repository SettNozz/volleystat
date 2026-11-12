/* Manual match context editor. No phase or rotation inference. */
'use strict';
const MatchSetup = (() => {
  const copy = value => JSON.parse(JSON.stringify(value));

  function integer(value, name) {
    if (!['string', 'number'].includes(typeof value)) throw new Error(`${name} must be a whole number.`);
    if (typeof value === 'string' && !/^\d+$/.test(value.trim())) throw new Error(`${name} must be a whole number.`);
    const number = Number(value);
    if (!Number.isSafeInteger(number) || number < 0) throw new Error(`${name} must be a nonnegative whole number.`);
    return number;
  }

  function frameRange(start, end, totalFrames) {
    const first = integer(start, 'Start frame'), last = integer(end, 'End frame');
    if (!Number.isSafeInteger(totalFrames) || totalFrames < 1) throw new Error('Video frame range is unavailable.');
    if (first > last || last >= totalFrames) throw new Error(`Choose frames from 0 to ${totalFrames - 1}, with start before end.`);
    return [first, last];
  }

  function intervalFromFields(fields, totalFrames, previous = {}) {
    if (!['set', 'non_game'].includes(fields.type)) throw new Error('Choose a set or non-game interval.');
    const [start_frame, end_frame] = frameRange(fields.start, fields.end, totalFrames);
    const interval = {...copy(previous), start_frame, end_frame};
    if (fields.type === 'non_game') {
      interval.match_phase = 'non_game';
      delete interval.set_id;
      delete interval.team_identity_map;
    } else {
      const set = String(fields.set || '').trim();
      if (!set || set.length > 64) throw new Error('Enter a set ID of 1–64 characters.');
      interval.set_id = set;
      delete interval.match_phase;
      if (fields.far || fields.near) {
        if (!fields.far || !fields.near || fields.far === fields.near) throw new Error('Choose two different teams for this interval, or inherit both sides.');
        interval.team_identity_map = {team_a: fields.far, team_b: fields.near};
      } else delete interval.team_identity_map;
    }
    return interval;
  }

  function putInterval(config, interval, index, totalFrames) {
    const next = copy(config), segments = [...(next.segments || [])];
    if (index !== null && (!Number.isInteger(index) || index < 0 || index >= segments.length)) throw new Error('Selected interval no longer exists.');
    if (index === null) segments.push(copy(interval)); else segments[index] = copy(interval);
    segments.sort((a, b) => a.start_frame - b.start_frame);
    for (let i = 0; i < segments.length; i++) {
      frameRange(segments[i].start_frame, segments[i].end_frame, totalFrames);
      if (i && segments[i - 1].end_frame >= segments[i].start_frame) throw new Error('Intervals overlap. Adjust the selection before adding it.');
    }
    next.segments = segments;
    return next;
  }

  function removeInterval(config, index) {
    if (!Number.isInteger(index) || index < 0 || index >= (config.segments || []).length) throw new Error('Choose an interval to remove.');
    const next = copy(config); next.segments.splice(index, 1); return next;
  }

  function editContextFields(config, fields) {
    const next = copy(config);
    next.teams = {...(next.teams || {})}; next.initial_score = {...(next.initial_score || {})};
    for (const row of fields.teams) {
      const name = String(row.name || '').trim();
      if (!name || name.length > 64) throw new Error('Each team needs a name of 1–64 characters.');
      next.teams[row.id] = {...(next.teams[row.id] || {}), name};
      next.initial_score[row.id] = integer(row.score, 'Starting points');
    }
    if (!fields.far || !fields.near || fields.far === fields.near) throw new Error('Choose two different teams for the default court sides.');
    next.team_identity_map = {team_a: fields.far, team_b: fields.near};
    next.roster = fields.roster.map(row => {
      if (row.confirmed && !row.team_id) throw new Error(`Choose a team before confirming ${row.player_id}.`);
      const previous = (config.roster || []).find(player => player.player_id === row.player_id) || {};
      return {...copy(previous), player_id: row.player_id, team_id: row.team_id || null, confirmed: row.confirmed === true};
    });
    return next;
  }

  function createController(fetcher, apiUrl) {
    const state = {config: {}, source: null, revision: null, dirty: false, stale: false, loaded: false};
    function accept(data) {
      if (!data || typeof data.config !== 'object' || data.config === null || Array.isArray(data.config)) throw new Error('Saved match setup is unavailable.');
      Object.assign(state, copy(data), {config: copy(data.config), dirty: false, stale: false, loaded: true});
      return state;
    }
    async function read(response) {
      const data = await response.json();
      if (!response.ok) {
        if (response.status === 409) state.stale = true;
        throw new Error(data.error || 'Could not load match setup.');
      }
      return data;
    }
    return {state,
      load: async () => accept(await read(await fetcher(apiUrl))),
      setDraft: config => { state.config = copy(config); state.dirty = true; },
      save: async config => {
        if (!state.loaded) throw new Error('Load the saved setup before saving.');
        if (state.stale) throw new Error('Saved setup changed. Reload saved setup before saving again.');
        const data = await read(await fetcher(apiUrl, {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({source: state.source, revision: state.revision, config: copy(config)})}));
        return accept(data);
      },
    };
  }

  function initialize(document, options) {
    const byId = id => document.getElementById(id);
    const controller = createController(options.fetcher, options.api_url);
    const video = byId('setup-video');
    let selected = null, selectionPending = false, busy = false, playbackEnd = null;
    let teamIds = [], rosterIds = [];
    const maxFrame = () => Math.max(0, Number(controller.state.total_frames || options.total_frames) - 1);
    const fps = () => Number(controller.state.fps || options.fps);
    const currentFrame = () => Math.min(maxFrame(), Math.max(0, Math.round(video.currentTime * fps()) || 0));
    const time = frame => {
      const seconds = Math.max(0, Number(frame)) / fps();
      return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
    };
    function status(message, state = '') { byId('setup-status').textContent = message; byId('setup-status').dataset.state = state; }
    function controls() {
      for (const id of ['team-fields', 'roster-fields', 'interval-fields']) byId(id).disabled = busy || !controller.state.loaded;
      byId('setup-save').disabled = busy || !controller.state.loaded || controller.state.stale || selectionPending;
      byId('setup-download').disabled = busy || !controller.state.loaded || selectionPending;
      byId('setup-reload').disabled = busy;
      byId('setup-draft').textContent = controller.state.stale ? 'Saved setup changed. Reload before saving.' :
        selectionPending ? 'Selection not applied to draft.' : controller.state.dirty ? 'Unsaved changes' : 'Saved setup loaded';
      byId('setup-reload').textContent = controller.state.dirty || selectionPending ? 'Reload saved setup (discard draft)' : 'Reload saved setup';
    }
    function markDirty() { controller.state.dirty = true; controls(); }
    function pendingSelection() { selectionPending = true; status('Apply the selected interval to the draft, or choose New selection to discard this selection.'); controls(); }
    function teamLabel(id) { return controller.state.config.teams?.[id]?.name || id; }
    function fillTeamSelect(select, value, inherit = false) {
      select.replaceChildren();
      const blank = document.createElement('option'); blank.value = ''; blank.textContent = inherit ? 'Inherit default side' : 'Choose team'; select.append(blank);
      for (const id of teamIds) {
        const option = document.createElement('option'); option.value = id; option.textContent = teamLabel(id); select.append(option);
      }
      select.value = value || '';
    }
    function selectionValues(start, end) {
      byId('interval-start').value = start; byId('interval-end').value = end;
      byId('interval-start-range').value = start; byId('interval-end-range').value = end;
      byId('interval-start-time').textContent = time(start); byId('interval-end-time').textContent = time(end);
    }
    function updateIntervalMode() {
      const excluded = byId('interval-type').value === 'non_game';
      for (const id of ['set-name-label', 'set-far-label', 'set-near-label']) byId(id).hidden = excluded;
      byId('swap-interval-sides').disabled = excluded;
    }
    function selectInterval(index) {
      selected = index; selectionPending = false;
      const interval = controller.state.config.segments?.[index];
      selectionValues(interval?.start_frame ?? currentFrame(), interval?.end_frame ?? maxFrame());
      byId('interval-type').value = interval?.match_phase === 'non_game' ? 'non_game' : 'set';
      byId('interval-set').value = interval?.set_id || `set_${(controller.state.config.segments || []).filter(row => row.set_id).length + 1}`;
      fillTeamSelect(byId('interval-far'), interval?.team_identity_map?.team_a, true);
      fillTeamSelect(byId('interval-near'), interval?.team_identity_map?.team_b, true);
      byId('apply-interval').textContent = index === null ? 'Add interval to draft' : 'Update interval in draft';
      byId('remove-interval').disabled = index === null;
      byId('selection-status').textContent = index === null ? 'Choose a selection, then add it to the draft.' : 'Editing this interval. Update it in the draft before saving.';
      updateIntervalMode(); renderIntervals(); controls();
    }
    function renderIntervals() {
      const list = byId('setup-intervals'), track = byId('setup-timeline'); list.replaceChildren(); track.replaceChildren();
      const segments = controller.state.config.segments || [];
      if (!segments.length) { const empty = document.createElement('li'); empty.textContent = 'No intervals yet. Mark a set or exclude warm-up.'; list.append(empty); }
      segments.forEach((interval, index) => {
        const excluded = interval.match_phase === 'non_game';
        const label = excluded ? 'Non-game' : interval.set_id || 'Match play';
        const mapping = interval.team_identity_map;
        const text = `${label} · ${time(interval.start_frame)}–${time(interval.end_frame)}`;
        const item = document.createElement('li'), button = document.createElement('button'); button.type = 'button';
        button.textContent = text; button.setAttribute('aria-pressed', String(index === selected));
        const detail = document.createElement('small'); detail.textContent = `Frames ${interval.start_frame}–${interval.end_frame}` +
          (mapping ? ` · Far: ${teamLabel(mapping.team_a)} · Near: ${teamLabel(mapping.team_b)}` : ' · Default sides');
        button.append(detail); button.onclick = () => selectInterval(index); item.append(button); list.append(item);
        const mark = document.createElement('button'); mark.type = 'button'; mark.className = 'setup-interval' + (excluded ? ' non-game' : '');
        mark.style.left = (100 * interval.start_frame / (maxFrame() + 1)) + '%';
        mark.style.width = (100 * (interval.end_frame - interval.start_frame + 1) / (maxFrame() + 1)) + '%';
        mark.title = text; mark.textContent = label; mark.setAttribute('aria-label', text); mark.setAttribute('aria-pressed', String(index === selected));
        mark.onclick = () => { selectInterval(index); video.currentTime = interval.start_frame / fps(); }; track.append(mark);
      });
    }
    function renderContext() {
      const config = controller.state.config;
      teamIds = [...new Set([...Object.keys(config.teams || {}), ...Object.values(config.team_identity_map || {})])].filter(Boolean);
      const teams = byId('setup-teams'); teams.replaceChildren();
      for (const id of teamIds) {
        const tr = document.createElement('tr'), idCell = document.createElement('th'); idCell.scope = 'row'; idCell.textContent = id;
        const nameCell = document.createElement('td'), name = document.createElement('input'); name.type = 'text'; name.maxLength = 64;
        name.value = config.teams?.[id]?.name || id; name.dataset.teamName = id; name.setAttribute('aria-label', `Name for ${id}`); name.oninput = markDirty; nameCell.append(name);
        const scoreCell = document.createElement('td'), score = document.createElement('input'); score.type = 'number'; score.min = '0'; score.step = '1'; score.value = config.initial_score?.[id] ?? 0;
        score.dataset.teamScore = id; score.setAttribute('aria-label', `Starting points for ${id}`); score.oninput = markDirty; scoreCell.append(score);
        tr.append(idCell, nameCell, scoreCell); teams.append(tr);
      }
      fillTeamSelect(byId('match-far'), config.team_identity_map?.team_a); fillTeamSelect(byId('match-near'), config.team_identity_map?.team_b);
      rosterIds = [...new Set([...(controller.state.roster_ids || []), ...(config.roster || []).map(row => row.player_id)])];
      const roster = byId('setup-roster'); roster.replaceChildren();
      for (const id of rosterIds) {
        const saved = (config.roster || []).find(row => row.player_id === id) || {};
        const row = document.createElement('tr'), label = document.createElement('th'); label.scope = 'row'; label.textContent = id;
        const teamCell = document.createElement('td'), team = document.createElement('select'); team.dataset.rosterTeam = id; team.setAttribute('aria-label', `Team for ${id}`);
        fillTeamSelect(team, saved.team_id); team.options[0].textContent = 'Unassigned'; team.onchange = markDirty; teamCell.append(team);
        const confirmedCell = document.createElement('td'), confirmed = document.createElement('input'); confirmed.type = 'checkbox'; confirmed.dataset.rosterConfirmed = id;
        confirmed.checked = saved.confirmed === true; confirmed.setAttribute('aria-label', `Confirm roster entry ${id}`); confirmed.onchange = markDirty; confirmedCell.append(confirmed);
        row.append(label, teamCell, confirmedCell); roster.append(row);
      }
      if (!rosterIds.length) { const row = document.createElement('tr'), cell = document.createElement('td'); cell.colSpan = 3; cell.textContent = 'No tracked roster in this analysis.'; row.append(cell); roster.append(row); }
      const history = byId('setup-history'); history.replaceChildren();
      for (const record of [...(controller.state.history || [])].reverse().slice(0, 20)) {
        const item = document.createElement('li'); item.textContent = record.saved_at || record.timestamp || `Revision ${record.revision ?? 'saved'}`; history.append(item);
      }
      for (const id of ['setup-seek', 'interval-start-range', 'interval-end-range', 'interval-start', 'interval-end']) byId(id).max = maxFrame();
      selectInterval(null);
    }
    function collectConfig() {
      const names = [...document.querySelectorAll('[data-team-name]')], scores = [...document.querySelectorAll('[data-team-score]')];
      const teams = names.map(name => ({id: name.dataset.teamName, name: name.value, score: scores.find(score => score.dataset.teamScore === name.dataset.teamName).value}));
      const roster = rosterIds.map(id => ({player_id: id,
        team_id: [...document.querySelectorAll('[data-roster-team]')].find(el => el.dataset.rosterTeam === id).value,
        confirmed: [...document.querySelectorAll('[data-roster-confirmed]')].find(el => el.dataset.rosterConfirmed === id).checked}));
      return editContextFields(controller.state.config, {teams, roster, far: byId('match-far').value, near: byId('match-near').value});
    }
    async function load() {
      busy = true; controls(); status('Loading saved match setup…');
      try { await controller.load(); renderContext(); status('Saved match setup loaded.', 'saved'); }
      catch (error) { status(error.message, 'error'); }
      finally { busy = false; controls(); }
    }
    byId('setup-reload').onclick = load;
    byId('setup-save').onclick = async () => {
      try {
        const draft = collectConfig(); controller.setDraft(draft); busy = true; controls(); status('Saving match setup…');
        await controller.save(draft); renderContext(); status('Match setup saved. Statistics use the saved context.', 'saved');
      } catch (error) { status(error.message, 'error'); }
      finally { busy = false; controls(); }
    };
    byId('setup-download').onclick = () => {
      try {
        const config = collectConfig();
        const blob = new Blob([JSON.stringify(config, null, 2) + '\n'], {type: 'application/json'});
        const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'match_config.json'; link.click(); URL.revokeObjectURL(url);
        status('Downloaded current draft as match-config JSON. Save match setup to apply the draft here.');
      } catch (error) { status(error.message, 'error'); }
    };
    for (const id of ['match-far', 'match-near']) byId(id).onchange = markDirty;
    byId('swap-match-sides').onclick = () => { const far = byId('match-far').value; byId('match-far').value = byId('match-near').value; byId('match-near').value = far; markDirty(); };
    byId('swap-interval-sides').onclick = () => {
      const far = byId('interval-far').value || byId('match-far').value;
      const near = byId('interval-near').value || byId('match-near').value;
      byId('interval-far').value = near; byId('interval-near').value = far; pendingSelection();
    };
    for (const [inputId, isStart] of [['interval-start-range', true], ['interval-end-range', false], ['interval-start', true], ['interval-end', false]]) {
      byId(inputId).oninput = event => {
        const start = isStart ? event.target.value : byId('interval-start').value;
        const end = isStart ? byId('interval-end').value : event.target.value;
        selectionValues(start, end); pendingSelection();
      };
    }
    for (const id of ['interval-set', 'interval-far', 'interval-near']) byId(id).onchange = pendingSelection;
    byId('interval-type').onchange = () => { updateIntervalMode(); pendingSelection(); };
    byId('mark-start').onclick = () => { selectionValues(currentFrame(), byId('interval-end').value); pendingSelection(); };
    byId('mark-end').onclick = () => { selectionValues(byId('interval-start').value, currentFrame()); pendingSelection(); };
    byId('new-interval').onclick = () => { selectInterval(null); status('New selection ready. Existing interval drafts remain unchanged.'); };
    byId('apply-interval').onclick = () => {
      try {
        const currentConfig = collectConfig();
        const interval = intervalFromFields({start: byId('interval-start').value, end: byId('interval-end').value,
          type: byId('interval-type').value, set: byId('interval-set').value, far: byId('interval-far').value, near: byId('interval-near').value},
        maxFrame() + 1, selected === null ? {} : currentConfig.segments[selected]);
        controller.setDraft(putInterval(currentConfig, interval, selected, maxFrame() + 1));
        renderContext(); status('Interval added to draft. Save match setup to apply it.');
      } catch (error) { status(error.message, 'error'); }
    };
    byId('remove-interval').onclick = () => {
      try { controller.setDraft(removeInterval(collectConfig(), selected)); renderContext(); status('Interval removed from draft. Save match setup to apply it.'); }
      catch (error) { status(error.message, 'error'); }
    };
    byId('setup-seek').oninput = event => { playbackEnd = null; video.pause(); video.currentTime = Number(event.target.value) / fps(); };
    byId('preview-selection').onclick = () => {
      try {
        const [start, end] = frameRange(byId('interval-start').value, byId('interval-end').value, maxFrame() + 1);
        playbackEnd = (end + 1) / fps(); video.currentTime = start / fps(); video.play().catch(error => status(error.message, 'error'));
      } catch (error) { status(error.message, 'error'); }
    };
    video.addEventListener('timeupdate', () => {
      const frame = currentFrame(); byId('setup-seek').value = frame; byId('setup-position').textContent = time(frame);
      byId('setup-time').textContent = `${time(frame)} · frame ${frame}`;
      if (playbackEnd !== null && video.currentTime >= playbackEnd) { video.pause(); playbackEnd = null; }
    });
    load();
    return {controller, load};
  }
  return {integer, frameRange, intervalFromFields, putInterval, removeInterval, editContextFields, createController, initialize};
})();

if (typeof module !== 'undefined' && module.exports) module.exports = MatchSetup;
if (typeof document !== 'undefined' && document.getElementById('match-setup-data')) {
  MatchSetup.initialize(document, {...JSON.parse(document.getElementById('match-setup-data').textContent), fetcher: (...args) => fetch(...args)});
}
