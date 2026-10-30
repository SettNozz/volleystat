/* Per-player contacts use a separate derived report; source events stay intact. */
let selectedPlayerId = null;
let playerContactReport = null;

function playerPoseStyle(pose) {
  const targetSide = playerContactReport ? playerContactReport.side : 'b';
  const onSide = pose.team === targetSide || pose.team === 'team_' + targetSide;
  const highlighted = selectedPlayerId ? (onSide && pose.player_id === selectedPlayerId) : onSide;
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
  summary.textContent = player
    ? `${player.player_id}: ${player.total} contacts · ${player.reviewed} reviewed. ` +
      `Reception ${player.counts.reception} · Dig/pass/set ${player.counts.dig_pass} · ` +
      `Attack ${player.counts.attack} · Block ${player.counts.block} · Serve ${player.counts.serve}`
    : `${report.contacts.length} contact events · ${totals.assigned || 0} assigned · ` +
      `${totals.unresolved || 0} unresolved · ${totals.other_side || 0} other side`;
  const list = document.getElementById('playerContactList');
  list.replaceChildren();
  const contacts = report.contacts.filter(c => !selectedPlayerId || c.player_id === selectedPlayerId);
  if (!contacts.length) {
    list.textContent = 'No attributed contacts. All includes unresolved events for review.';
    return;
  }
  for (const contact of contacts) {
    const row = document.createElement('div');
    row.className = 'player-contact-row';
    row.dataset.eventIndex = String(contact.event_index);
    const jump = document.createElement('button');
    jump.type = 'button';
    jump.className = 'ctrl player-contact-jump';
    jump.textContent = `${formatTime(contact.time)} · ${contact.type.replace(/_/g, ' ')}`;
    jump.onclick = () => seekAndPlay(contact.frame);
    const provenance = document.createElement('small');
    provenance.textContent = contact.label_source === 'human'
      ? (contact.verified ? 'Human attribution · type still predicted' : 'Human: unresolved')
      : `Proposal · ${contact.reason.replace(/_/g, ' ')}`;
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
          body: JSON.stringify({source: report.source, event_index: contact.event_index,
            player_id: report.roster_ids.includes(value) ? value : null,
            other_side: value === 'other', reset: value === 'reset'}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Contact review failed');
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

async function loadPlayerContacts() {
  try {
    const response = await fetch('/api/player_contacts/' + JOB_ID);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Contacts unavailable');
    playerContactReport = data;
    renderPlayerContacts();
    document.getElementById('playerContactStatus').textContent = data.roster_ids.length
      ? 'Automatic attribution is unverified. Review via dropdown; unknowns are excluded from player totals.'
      : 'No stable roster IDs in this result. Contacts stay unresolved.';
  } catch (error) {
    document.getElementById('playerContactStatus').textContent = error.message;
  }
}

document.addEventListener('DOMContentLoaded', loadPlayerContacts);
