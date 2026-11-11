/* Developer-only source review; geometry and model outputs are never generated here. */
'use strict';
const reviewBoot = JSON.parse(document.getElementById('developer-review-data').textContent);
const reviewVideo = document.getElementById('review-video');
const reviewNodes = Object.fromEntries(['summary', 'status', 'list', 'current', 'reasons', 'contact-form', 'point-form',
  'frame', 'type', 'side', 'author', 'detection-verified', 'type-verified', 'save', 'remove', 'reset', 'dismiss',
  'point-winner', 'before', 'after', 'filter-status', 'filter-issue', 'sort', 'history', 'prev', 'next']
  .map(id => [id, document.getElementById('review-' + id)]));
let reviewData = null;
let reviewShown = [];
let reviewSelected = null;
let reviewAdding = false;
let reviewStopAt = null;
let reviewBusy = false;

const reviewReasonLabels = {
  unresolved_author: 'Contact author remains unresolved.', low_event_confidence: 'Saved detector confidence is low.',
  same_frame_contacts: 'Multiple contacts share this frame.', contact_ball_unobserved: 'No observed in-frame ball at this contact.',
  unresolved_outcome: 'Outcome lacks supporting evidence.', unreviewed_contact: 'Event detection/type have not been verified.',
  point_winner_unresolved: 'Point winner remains unresolved.', overlapping_event_attempts: 'Contact rally windows overlap.',
  overlapping_score_attempts: 'Score rally windows overlap.', serve_anchor_mismatch: 'Contact and score serve frames disagree.',
  contact_side_mismatch: 'Saved event side conflicts with the assigned roster side.',
  ambiguous_point_link: 'Point/rally identifiers are ambiguous.', point_set_mismatch: 'Point belongs to another set.',
  attack_fault_point_mismatch: 'Saved attack fault conflicts with the point winner.',
  serve_outcome_point_mismatch: 'Saved serve outcome conflicts with the point winner.',
};
const reviewConflicts = new Set(['overlapping_event_attempts', 'overlapping_score_attempts', 'serve_anchor_mismatch',
  'contact_side_mismatch', 'ambiguous_point_link', 'point_set_mismatch', 'point_side_missing',
  'attack_fault_point_mismatch', 'serve_outcome_point_mismatch', 'multiple_terminal_contacts']);

function reviewMessage(message, error = false) {
  reviewNodes.status.textContent = message;
  reviewNodes.status.setAttribute('role', error ? 'alert' : 'status');
}

function reviewAxisVerified(item, axis) {
  const evidence = item?.review_axes?.[axis];
  return evidence?.verified === true && (axis !== 'detection' || evidence.value !== false);
}

function reviewPlay(item = reviewSelected) {
  if (!item) return;
  const before = Math.max(0, Math.min(15, Number(reviewNodes.before.value) || 0));
  const after = Math.max(0, Math.min(15, Number(reviewNodes.after.value) || 0));
  const time = item.frame / reviewBoot.fps;
  reviewVideo.pause(); reviewVideo.currentTime = Math.max(0, time - before);
  reviewStopAt = time + after; reviewVideo.play().catch(() => {});
}

function reviewSelect(item, play = true) {
  reviewAdding = false; reviewSelected = item;
  reviewNodes['contact-form'].hidden = !item || item.kind !== 'contact';
  reviewNodes['point-form'].hidden = !item || item.kind !== 'point';
  reviewNodes.dismiss.disabled = !item || reviewBusy;
  reviewNodes.current.textContent = item ? `${item.frame / reviewBoot.fps}s · f${item.frame} · ${item.type} · ${item.player_id || 'Unassigned'} · ${item.status}` : 'No matching queue items.';
  reviewNodes.reasons.textContent = item ? item.reasons.map(reason => reviewReasonLabels[reason] || reason.replace(/_/g, ' ')).join(' ') : '';
  for (const button of reviewNodes.list.querySelectorAll('button')) button.setAttribute('aria-current', String(button.dataset.ref === item?.ref));
  if (item?.kind === 'contact') {
    reviewNodes.frame.value = item.frame; reviewNodes.type.value = item.type; reviewNodes.side.value = item.side || '';
    reviewNodes.author.value = 'unchanged';
    reviewNodes['detection-verified'].checked = reviewAxisVerified(item, 'detection');
    reviewNodes['type-verified'].checked = reviewAxisVerified(item, 'type');
    reviewNodes.save.textContent = 'Save correction'; reviewNodes.remove.hidden = false; reviewNodes.reset.hidden = false;
  } else if (item?.kind === 'point') reviewNodes['point-winner'].value = item.winner_side || 'unknown';
  const index = reviewShown.indexOf(item);
  reviewNodes.prev.disabled = index <= 0; reviewNodes.next.disabled = index < 0 || index >= reviewShown.length - 1;
  if (play && item) reviewPlay(item);
}

function reviewRender(keepRef = reviewSelected?.ref) {
  const queue = reviewData.queue;
  const status = reviewNodes['filter-status'].value, issue = reviewNodes['filter-issue'].value;
  reviewShown = queue.items.filter(item => (status === 'all' || item.status === status) &&
    (issue === 'all' || (issue === 'point' ? item.kind === 'point' : issue === 'conflict' ? item.reasons.some(reason => reviewConflicts.has(reason)) : item.reasons.includes(issue))));
  if (reviewNodes.sort.value === 'time') reviewShown.sort((a, b) => a.frame - b.frame || a.ref.localeCompare(b.ref));
  reviewNodes.list.replaceChildren();
  for (const item of reviewShown) {
    const li = document.createElement('li'), button = document.createElement('button');
    button.type = 'button'; button.dataset.ref = item.ref;
    button.textContent = `${(item.frame / reviewBoot.fps).toFixed(2)}s · ${item.type.replace(/_/g, ' ')} · ${item.player_id || (item.kind === 'point' ? 'Point winner' : 'Unassigned')}`;
    const detail = document.createElement('small');
    detail.textContent = `Impact ${item.priority} · ${item.status} · ${item.reasons.map(reason => reason.replace(/_/g, ' ')).join(', ')}`;
    button.append(detail); button.onclick = () => reviewSelect(item); li.append(button); reviewNodes.list.append(li);
  }
  if (!reviewShown.length) reviewNodes.list.textContent = 'No matching review items.';
  reviewNodes.summary.textContent = `${queue.summary.total} moments · ${queue.summary.pending || 0} pending · ${queue.summary.reviewed || 0} reviewed events · ${queue.summary.dismissed || 0} dismissed · ${queue.summary.removed || 0} removed`;
  reviewSelect(reviewShown.find(item => item.ref === keepRef) || reviewShown[0] || null, false);
  reviewNodes.history.replaceChildren();
  for (const record of (reviewData.history || []).slice().reverse()) {
    const li = document.createElement('li');
    li.textContent = (record.saved_at || '') + ' · ' + JSON.stringify(record.edit || record.review || record);
    reviewNodes.history.append(li);
  }
}

function reviewOptions() {
  reviewNodes.type.replaceChildren();
  const blank = document.createElement('option'); blank.value = ''; blank.textContent = 'Choose type'; reviewNodes.type.append(blank);
  for (const type of reviewData.queue.contact_types) {
    const option = document.createElement('option'); option.value = type; option.textContent = type.replace(/_/g, ' '); reviewNodes.type.append(option);
  }
  reviewNodes.author.replaceChildren();
  for (const [value, label] of [['unchanged', 'Keep current attribution'], ...reviewData.queue.roster_ids.map(id => [id, id]), ['unresolved', 'Unresolved author'], ['other_side', 'Other court side']]) {
    const option = document.createElement('option'); option.value = value; option.textContent = label; reviewNodes.author.append(option);
  }
}

async function reviewLoad(keepRef) {
  const response = await fetch(reviewBoot.api);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load review queue');
  reviewData = data;
  reviewOptions(); reviewRender(keepRef);
}

async function reviewPost(edit) {
  if (reviewBusy || !reviewData) return;
  reviewBusy = true; reviewMessage('Saving review…');
  for (const button of document.querySelectorAll('.review-actions button, #review-save')) button.disabled = true;
  try {
    const response = await fetch(reviewBoot.api, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source: reviewData.source, revision: reviewData.revision, edit})});
    const data = await response.json();
    if (!response.ok) throw new Error((data.error || 'Could not save review') + (response.status === 409 ? ' Refresh the queue before retrying.' : ''));
    await reviewLoad(); reviewMessage('Review saved. Derived views now use this correction.');
  } catch (error) { reviewMessage(error.message, true); }
  finally {
    reviewBusy = false;
    for (const button of document.querySelectorAll('.review-actions button, #review-save')) button.disabled = false;
    reviewNodes.dismiss.disabled = !reviewSelected;
  }
}

function reviewAuthorChanges(changes) {
  const value = reviewNodes.author.value;
  if (value === 'unchanged') return;
  changes.player_id = value === 'unresolved' || value === 'other_side' ? null : value;
  changes.other_side = value === 'other_side';
}

reviewNodes['contact-form'].onsubmit = event => {
  event.preventDefault();
  if (!reviewAdding && !reviewSelected) return;
  const frame = Number(reviewNodes.frame.value), type = reviewNodes.type.value, side = reviewNodes.side.value || null;
  if (!Number.isInteger(frame) || frame < 0 || (reviewBoot.totalFrames > 0 && frame >= reviewBoot.totalFrames) || !type) {
    reviewMessage('Choose a contact type and a frame inside the video.', true); return;
  }
  const changes = {};
  if (reviewAdding || frame !== reviewSelected.frame) changes.frame = frame;
  if (reviewAdding || type !== reviewSelected.type) changes.type = type;
  if (reviewAdding || side !== reviewSelected.side) changes.side = side;
  reviewAuthorChanges(changes);
  for (const [axis, field] of [['detection', 'detection_verified'], ['type', 'type_verified']]) {
    const value = reviewNodes[axis + '-verified'].checked;
    if (reviewAdding || value !== reviewAxisVerified(reviewSelected, axis)) changes[field] = value;
  }
  if (!Object.keys(changes).length) { reviewMessage('No correction selected. Confirm an axis, change a field or dismiss this queue item.'); return; }
  reviewPost(reviewAdding ? {action: 'add', review_id: crypto.randomUUID(), changes} : {action: 'update', event_index: reviewSelected.event_index, changes});
};
reviewNodes['point-form'].onsubmit = async event => {
  event.preventDefault();
  if (!reviewSelected || reviewSelected.kind !== 'point' || reviewBusy) return;
  reviewBusy = true;
  try {
    const value = reviewNodes['point-winner'].value;
    const response = await fetch(reviewBoot.pointApi, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source: reviewData.source, revision: reviewData.revision, attempt_ref: reviewSelected.attempt_ref,
        winner_side: value === 'a' || value === 'b' ? value : null, reset: value === 'reset'})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not save point winner');
    await reviewLoad(); reviewMessage('Point winner saved; event and author verification are unchanged.');
  } catch (error) { reviewMessage(error.message, true); }
  finally { reviewBusy = false; }
};
reviewNodes.remove.onclick = () => reviewSelected && reviewPost({action: 'remove', event_index: reviewSelected.event_index});
reviewNodes.reset.onclick = () => reviewSelected && reviewPost({action: 'reset', event_index: reviewSelected.event_index});
reviewNodes.dismiss.onclick = () => reviewSelected && reviewPost({action: 'dismiss', item_ref: reviewSelected.ref});
reviewNodes.prev.onclick = () => reviewSelect(reviewShown[reviewShown.indexOf(reviewSelected) - 1] || reviewSelected);
reviewNodes.next.onclick = () => reviewSelect(reviewShown[reviewShown.indexOf(reviewSelected) + 1] || reviewSelected);
document.getElementById('review-play').onclick = () => reviewPlay();
document.getElementById('review-add').onclick = () => {
  if (!reviewData || reviewBusy) return;
  reviewAdding = true; reviewSelected = null;
  reviewNodes['contact-form'].hidden = false; reviewNodes['point-form'].hidden = true;
  reviewNodes.frame.value = Math.max(0, Math.round(reviewVideo.currentTime * reviewBoot.fps));
  reviewNodes.type.value = ''; reviewNodes.side.value = ''; reviewNodes.author.value = 'unresolved';
  reviewNodes['detection-verified'].checked = true; reviewNodes['type-verified'].checked = true;
  reviewNodes.save.textContent = 'Add reviewed contact'; reviewNodes.remove.hidden = true; reviewNodes.reset.hidden = true;
  reviewNodes.current.textContent = 'Add a missed contact. No ball or pose coordinates will be created.';
  reviewNodes.reasons.textContent = 'Select the type and exact contact frame; author may remain unresolved.';
};
document.getElementById('review-use-time').onclick = () => { reviewNodes.frame.value = Math.max(0, Math.round(reviewVideo.currentTime * reviewBoot.fps)); };
function reviewStep(delta) { reviewVideo.pause(); reviewStopAt = null; reviewVideo.currentTime = Math.max(0, reviewVideo.currentTime + delta / reviewBoot.fps); }
document.getElementById('review-step-back').onclick = () => reviewStep(-1);
document.getElementById('review-step-forward').onclick = () => reviewStep(1);
reviewVideo.addEventListener('timeupdate', () => {
  if (!reviewVideo.paused && reviewStopAt !== null && reviewVideo.currentTime >= reviewStopAt) { reviewVideo.pause(); reviewStopAt = null; }
});
for (const name of ['filter-status', 'filter-issue', 'sort']) reviewNodes[name].onchange = () => reviewData && reviewRender();
document.getElementById('review-refresh').onclick = () => reviewLoad(reviewSelected?.ref)
  .then(() => reviewMessage('Queue refreshed.'))
  .catch(error => reviewMessage(error.message, true));
document.addEventListener('keydown', event => {
  if (event.target.closest('input,select,textarea,button') || event.ctrlKey || event.metaKey || event.altKey || reviewBusy) return;
  const key = event.key.toLowerCase();
  if (key === 'j') reviewNodes.prev.click(); else if (key === 'k') reviewNodes.next.click();
  else if (key === ' ') { event.preventDefault(); reviewPlay(); }
  else if (key === 'arrowleft') { event.preventDefault(); reviewStep(-1); }
  else if (key === 'arrowright') { event.preventDefault(); reviewStep(1); }
  else if (key === 'x' && reviewSelected?.kind === 'contact') reviewNodes.remove.click();
  else if (key === 'd') reviewNodes.dismiss.click();
  else if (key === '+' || key === '=') document.getElementById('review-add').click();
  else if (/^[1-6]$/.test(key) && reviewSelected?.kind === 'contact') reviewNodes.type.value = ['serve', 'reception', 'set', 'dig', 'attack', 'block'][Number(key) - 1];
  else if (key === 'enter' && reviewSelected?.kind === 'contact') reviewNodes['contact-form'].requestSubmit();
});
reviewLoad().catch(error => reviewMessage(error.message, true));
