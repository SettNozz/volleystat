/* Evidence playback and source-bound point reviews. No contact inference. */
'use strict';
const evidenceData = JSON.parse(document.getElementById('evidence-data').textContent);
const video = document.getElementById('evidence-video');
const contacts = evidenceData.contacts;
const list = document.getElementById('evidence-contacts');
let selectedContact = -1;
let stopAt = null;

const reasons = {
  derived_outcome: 'Outcome derived from saved rally evidence.',
  no_score_attempt: 'No unique score rally supports an outcome.',
  point_unresolved: 'Point winner remains unresolved.',
  point_conflict: 'Conflicting point evidence; review the point winner.',
  overlapping_event_attempts: 'Contact belongs to multiple contact rally windows.',
  overlapping_score_attempts: 'Contact belongs to multiple score rally windows.',
  serve_anchor_mismatch: 'Contact and score rallies have different serve anchors.',
  contact_side_mismatch: 'Saved event side disagrees with the assigned roster side.',
  multiple_terminal_contacts: 'Multiple contacts share the final frame; terminal attribution is ambiguous.',
  reception_not_matched: 'Contact does not match the saved reception frame.',
  reception_quality_unknown: 'Reception quality remains unresolved.',
  outcome_not_supported: 'This contact is counted as a touch; outcome scoring is available for attacks, serves and serve receptions.',
  serve_outcome_point_mismatch: 'Saved serve outcome contradicts the point winner. Outcome remains unresolved.',
  attack_fault_point_mismatch: 'Saved attack fault contradicts the point winner. Confirming the winner alone does not verify an attack outcome.',
};

function playWindow(start, end) {
  video.pause();
  stopAt = Number.isFinite(end) ? end : null;
  video.currentTime = Math.max(0, start);
  video.play().catch(() => {});
}

function showContact(index, autoplay = true) {
  if (index < 0 || index >= contacts.length) return;
  selectedContact = index;
  const contact = contacts[index];
  for (const [i, button] of Array.from(list.querySelectorAll('button')).entries()) {
    button.setAttribute('aria-current', String(i === index));
  }
  const time = contact.video_frame / evidenceData.fps;
  document.getElementById('now').textContent = `${index + 1} / ${contacts.length} · ${contact.player_id || 'Unassigned'} · ${contact.type} · ${time.toFixed(2)}s`;
  document.getElementById('explanation').textContent = (reasons[contact.reason] || 'Outcome unresolved.') +
    (contact.point_reviewed ? ' Point winner confirmed by human; contact outcome remains derived.' : '') +
    (contact.event_verified ? ' Event human-verified.' : ' Event detection and type unverified.');
  document.getElementById('watch-rally').disabled = !Number.isFinite(contact.start_frame) || !Number.isFinite(contact.end_frame);
  document.getElementById('previous').disabled = index === 0;
  document.getElementById('next').disabled = index === contacts.length - 1;
  const link = document.getElementById('open-contact');
  link.hidden = false;
  link.href = evidenceData.viewer + '?frame=' + contact.frame;
  if (autoplay) playWindow(Math.max(0, time - 2), time + 3);
}

for (const [index, contact] of contacts.entries()) {
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'contact';
  button.textContent = `${(contact.video_frame / evidenceData.fps).toFixed(2)}s · ${contact.player_id || 'Unassigned'} · ${contact.type.replace(/_/g, ' ')} · ${contact.outcome}`;
  const detail = document.createElement('small');
  detail.textContent = (contact.author_source === 'human' ? 'Human-confirmed author' :
    contact.author_source === 'automatic' ? 'Automatic attribution' : 'Unassigned author') +
    (evidenceData.metric === 'rate' ? ' · ' + (contact.outcome === (evidenceData.kind === 'serve' ? 'error' : 'controlled') ? 'Numerator + denominator' : 'Denominator only') : '');
  button.append(detail);
  button.onclick = () => showContact(index);
  item.append(button); list.append(item);
}
if (contacts.length) showContact(0, false);
else {
  list.textContent = 'No contacts counted by this metric.';
  document.getElementById('previous').disabled = true;
  document.getElementById('next').disabled = true;
}
document.getElementById('previous').onclick = () => showContact(selectedContact - 1);
document.getElementById('next').onclick = () => showContact(selectedContact + 1);
document.getElementById('watch-rally').onclick = () => {
  const contact = contacts[selectedContact];
  if (contact && Number.isFinite(contact.start_frame) && Number.isFinite(contact.end_frame))
    playWindow(contact.start_frame / evidenceData.fps, contact.end_frame / evidenceData.fps);
};
for (const button of document.querySelectorAll('[data-rally-start]')) {
  button.onclick = () => playWindow(Number(button.dataset.rallyStart) / evidenceData.fps,
    button.hasAttribute('data-rally-end') ? Number(button.dataset.rallyEnd) / evidenceData.fps : null);
}
video.addEventListener('timeupdate', () => {
  if (!video.paused && stopAt !== null && video.currentTime >= stopAt) { video.pause(); stopAt = null; }
});
document.getElementById('category').onchange = event => {
  const metric = document.getElementById('metric');
  metric.replaceChildren();
  for (const [value, label] of evidenceData.modes[event.target.value]?.[1] || [['total', 'Contacts']]) {
    const option = document.createElement('option'); option.value = value; option.textContent = label; metric.append(option);
  }
};
for (const form of document.querySelectorAll('.point-review')) {
  form.onsubmit = async event => {
    event.preventDefault();
    const status = document.getElementById('review-status');
    const button = form.querySelector('button');
    const value = form.querySelector('select').value;
    button.disabled = true; status.textContent = 'Saving point review…';
    try {
      const response = await fetch(evidenceData.api, {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source: evidenceData.source, revision: evidenceData.revision, attempt_ref: form.dataset.attempt,
          reset: value === 'reset', winner_side: value === 'a' || value === 'b' ? value : null})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not save point review');
      location.reload();
    } catch (error) { status.textContent = error.message; button.disabled = false; }
  };
}
