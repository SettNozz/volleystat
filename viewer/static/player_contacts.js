/* Per-player contacts use a separate derived report; source events stay intact. */
let selectedPlayerId = null;
let playerContactReport = null;
let playerContactRequest = 0;
let activePlayerContact = null;

function playerContactCandidate(frameIndex) {
  if (!activePlayerContact || activePlayerContact.label_source === 'human') return null;
  const candidate = (activePlayerContact.candidates || [])[0];
  if (!candidate || candidate.frame !== frameIndex) return null;
  const actor = activePlayerContact.player_id || activePlayerContact.proposed_player_id;
  if (selectedPlayerId && actor !== selectedPlayerId) return null;
  return candidate;
}

function drawPlayerContactCandidate(ctx, ann, frameIndex, px, py, scale) {
  if (Number(ann.f) !== frameIndex) return;
  const candidate = playerContactCandidate(frameIndex);
  if (!candidate || !candidate.bbox) return;
  const box = candidate.bbox;
  ctx.save();
  ctx.strokeStyle = '#f59e0b';
  ctx.fillStyle = '#f59e0b';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(px(box[0]), py(box[1]), px(box[2]-box[0]), py(box[3]-box[1]));
  ctx.font = `bold ${Math.max(10, Math.round(12*scale))}px system-ui`;
  const actor = activePlayerContact.player_id || activePlayerContact.proposed_player_id;
  ctx.fillText(`${actor || (candidate.side === 'a' ? 'Far player' : 'Unknown player')} · contact proposal`, px(box[0]), py(box[1])-8);
  ctx.restore();
}

function renderPlayerContactMethods(report) {
  const bar = document.getElementById('playerContactMethodBar');
  const select = document.getElementById('playerContactMethod');
  const quality = document.getElementById('playerContactMethodQuality');
  if (!bar || !select || !quality) return;
  const methods = Object.entries(report.methods || {});
  bar.style.display = methods.length > 1 ? '' : 'none';
  select.replaceChildren();
  for (const [id, metadata] of methods) {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = metadata.label || id;
    select.appendChild(option);
  }
  select.value = report.method_id;
  select.disabled = false;
  const method = (report.methods || {})[report.method_id] || {};
  const metrics = method.team_lovo;
  quality.textContent = metrics
    ? `Team-side video holdout: ${(metrics.accuracy*100).toFixed(1)}% accuracy. Individual authors remain unverified.`
    : 'Geometric contact-author proposals; individual authors remain unverified.';
  if (typeof window !== 'undefined' && window._annData && typeof initPoseEstimates === 'function') {
    initPoseEstimates(window._annData, Boolean(report.parameters && report.parameters.estimates));
  }
}

async function changePlayerContactMethod(method) {
  const select = document.getElementById('playerContactMethod');
  if (select) select.disabled = true;
  await loadPlayerContacts(method);
}

function playerPoseStyle(pose, poseIndex, frameIndex) {
  const targetSide = playerContactReport ? playerContactReport.side : 'b';
  const onSide = pose.team === targetSide || pose.team === 'team_' + targetSide;
  const candidate = playerContactCandidate(frameIndex);
  const focused = candidate && candidate.pose_index === poseIndex && candidate.raw_track_id === pose.id;
  const highlighted = focused || (selectedPlayerId ? (onSide && pose.player_id === selectedPlayerId) : onSide);
  return {
    color: highlighted ? 'rgba(255, 220, 0, 0.98)' : 'rgba(150, 170, 190, 0.58)',
    width: highlighted ? 3 : 1.5,
    radius: highlighted ? 3.5 : 2,
  };
}

function selectPlayer(playerId) {
  selectedPlayerId = playerId;
  if (playerId) document.getElementById('togPlayers').checked = true;
  renderPlayerContacts();
  redraw();
}

function renderPlayerContacts() {
  const report = playerContactReport;
  if (!report) return;
  if (activePlayerContact) {
    activePlayerContact = report.contacts.find(c => c.event_index === activePlayerContact.event_index) || null;
  }
  renderPlayerContactMethods(report);
  const chips = document.getElementById('playerChips');
  chips.replaceChildren();
  for (const id of [null, ...report.roster_ids]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip' + (selectedPlayerId === id ? ' on' : '');
    button.textContent = id || 'All';
    button.setAttribute('aria-pressed', String(selectedPlayerId === id));
    button.onclick = () => selectPlayer(id);
    chips.appendChild(button);
  }
  const summary = document.getElementById('playerContactSummary');
  const player = report.players.find(p => p.player_id === selectedPlayerId);
  const totals = report.totals;
  const includeCandidates = document.getElementById('showPlayerContactCandidates');
  const isSuggestion = c => c.label_source !== 'human' && c.player_id !== selectedPlayerId && c.proposed_player_id === selectedPlayerId;
  const candidateCount = selectedPlayerId ? report.contacts.filter(isSuggestion).length : 0;
  summary.textContent = player
    ? `${player.player_id}: ${player.total} assigned · ${player.reviewed} reviewed` +
      (candidateCount ? ` · ${candidateCount} uncertain suggestions` : '') + '. ' +
      `Reception ${player.counts.reception} · Dig/pass/set ${player.counts.dig_pass} · ` +
      `Attack ${player.counts.attack} · Block ${player.counts.block} · Serve ${player.counts.serve}`
    : `${report.contacts.length} contact events · ${totals.assigned || 0} assigned · ` +
      `${totals.unresolved || 0} unresolved · ${totals.other_side || 0} other side`;
  const list = document.getElementById('playerContactList');
  list.replaceChildren();
  const contacts = report.contacts.filter(c => !selectedPlayerId || c.player_id === selectedPlayerId ||
    (includeCandidates && includeCandidates.checked && isSuggestion(c)));
  if (!contacts.length) {
    list.textContent = 'No attributed contacts. All includes unresolved events for review.';
    return;
  }
  for (const contact of contacts) {
    const row = document.createElement('div');
    const uncertain = selectedPlayerId && contact.player_id !== selectedPlayerId;
    row.className = 'player-contact-row' + (uncertain ? ' candidate' : '');
    row.dataset.eventIndex = String(contact.event_index);
    const jump = document.createElement('button');
    jump.type = 'button';
    jump.className = 'ctrl player-contact-jump';
    const derived = contact.label_source !== 'human';
    jump.textContent = `${formatTime(derived && Number.isFinite(contact.contact_time) ? contact.contact_time : contact.time)} · ${contact.type.replace(/_/g, ' ')}`;
    jump.title = 'Pause at proposed touch; use Play to inspect motion';
    jump.onclick = () => {
      activePlayerContact = contact;
      seekAndPlay(derived && Number.isFinite(contact.contact_frame) ? contact.contact_frame : contact.frame, false);
    };
    const provenance = document.createElement('small');
    provenance.textContent = contact.label_source === 'human'
      ? (contact.verified ? 'Human attribution · type still predicted' : 'Human: unresolved')
      : `Proposal · ${contact.reason.replace(/_/g, ' ')}`;
    if (contact.label_source !== 'human' && Number.isFinite(contact.near_side_score)) {
      provenance.textContent += ` · near-side score ${contact.near_side_score.toFixed(2)}`;
    }
    if (uncertain) provenance.textContent = 'Uncertain suggestion; excluded from total. ' + provenance.textContent;
    if (Number.isFinite(contact.actor_share)) provenance.textContent += ` · actor share ${contact.actor_share.toFixed(2)} (weak supervision)`;
    if (contact.label_source !== 'human' && contact.candidates && contact.candidates.length) {
      const candidates = contact.candidates.slice(0, 3).map(c =>
        `${c.player_id || (c.side === 'a' ? 'far player' : 'unknown player')} @ f${c.frame}`).join(', ');
      provenance.textContent += ` · ${candidates}`;
    }
    const select = document.createElement('select');
    select.className = 'ctrl';
    select.setAttribute('aria-label', `Player at frame ${contact.frame}`);
    const choices = [['unknown', 'Unresolved'], ['other', 'Other side'],
      ...report.roster_ids.map(id => [id, id]), ['reset', 'Reset to proposal']];
    for (const [value, label] of choices) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    }
    select.value = contact.player_id || (contact.status === 'other_side' ? 'other' : 'unknown');
    select.onchange = async () => {
      select.disabled = true;
      const value = select.value;
      try {
        const response = await fetch('/api/player_contacts/' + JOB_ID, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({source: report.source, method: report.method_id, event_index: contact.event_index,
            player_id: report.roster_ids.includes(value) ? value : null,
            other_side: value === 'other', reset: value === 'reset'}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Contact review failed');
        if (playerContactReport.method_id !== report.method_id) {
          await loadPlayerContacts(playerContactReport.method_id);
          return;
        }
        playerContactReport = data;
        document.getElementById('playerContactStatus').textContent = 'Attribution saved. Event type remains a model prediction.';
        renderPlayerContacts();
      } catch (error) {
        document.getElementById('playerContactStatus').textContent = error.message;
        select.value = contact.player_id || (contact.status === 'other_side' ? 'other' : 'unknown');
        select.disabled = false;
      }
    };
    row.append(jump, select, provenance);
    list.appendChild(row);
  }
}

async function loadPlayerContacts(method) {
  const requestId = ++playerContactRequest;
  try {
    const response = await fetch('/api/player_contacts/' + JOB_ID + (method ? '?method=' + encodeURIComponent(method) : ''));
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Contacts unavailable');
    if (requestId !== playerContactRequest) return;
    playerContactReport = data;
    renderPlayerContacts();
    document.getElementById('playerContactStatus').textContent = data.roster_ids.length
      ? 'Automatic attribution is unverified. Review via dropdown; unknowns are excluded from player totals.'
      : 'No stable roster IDs in this result. Contacts stay unresolved.';
  } catch (error) {
    if (requestId !== playerContactRequest) return;
    document.getElementById('playerContactStatus').textContent = error.message;
    const select = document.getElementById('playerContactMethod');
    if (select) {
      select.disabled = false;
      if (playerContactReport) select.value = playerContactReport.method_id;
    }
  }
}

document.addEventListener('DOMContentLoaded', () => loadPlayerContacts());
