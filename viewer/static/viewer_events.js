// Playback-driven contact history and the on-video event card. Saved events
// keep their original indexes so contact-author assignments remain aligned.
function contactTimeline(events, fps) {
  return events.map((event, index) => ({
    event, index,
    frame: Number.isFinite(event.frame) ? event.frame : Math.round(event.time * fps),
  })).filter(item => Number.isFinite(item.frame) && item.frame >= 0)
    .sort((a, b) => a.frame - b.frame || a.index - b.index);
}

// Upper bound: future contacts never become active, even just before a seek.
function contactCountAtFrame(timeline, frame) {
  let low = 0, high = timeline.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (timeline[mid].frame <= frame) low = mid + 1;
    else high = mid;
  }
  return low;
}

function contactLabel(type) {
  const labels = {setting_pass: 'Set', set: 'Set', pass: 'Set', reception_pass: 'Reception'};
  return labels[type] || String(type || 'Contact').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
}

const contactFeed = {
  timeline: contactTimeline(EVENTS, FPS), rows: [],
  count: -1, lastFrame: null, fullLog: false, badgeIndex: null,
};

function animateContact(element) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  element.getAnimations().forEach(animation => animation.cancel());
  element.animate([
    {opacity: 0, transform: 'translateY(10px) scale(.98)'},
    {opacity: 1, transform: 'translateY(0) scale(1)'},
  ], {duration: 320, easing: 'cubic-bezier(.2,.8,.2,1)'});
}

function renderEventList() {
  const box = document.getElementById('eventList');
  const rallyFirst = new Map();
  RALLIES.forEach((rally, index) => {
    const frame = rally.events?.[0]?.frame;
    if (Number.isFinite(frame)) rallyFirst.set(frame, index + 1);
  });
  contactFeed.rows = contactFeed.timeline.map(({event, index, frame}) => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'event-item';
    row.id = 'ev' + index;
    row.style.setProperty('--event-color', evColor(event.type));
    row.onclick = () => seekAndPlay(frame);
    const dot = document.createElement('span');
    dot.className = 'event-dot';
    dot.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'event-type';
    label.textContent = contactLabel(event.type);
    const time = document.createElement('span');
    time.className = 'event-time';
    time.textContent = formatTime(frame / FPS) + '.' + Math.floor(frame / FPS * 10) % 10;
    row.append(dot, label, time);
    if (event.inferred) {
      const inferred = document.createElement('small');
      inferred.className = 'event-inferred';
      inferred.textContent = 'Inferred';
      inferred.title = 'Recovered from the spotter serve probability; not human-verified.';
      row.append(inferred);
    }
    const separator = rallyFirst.has(frame) ? document.createElement('div') : null;
    if (separator) {
      separator.className = 'rally-sep';
      separator.textContent = 'Rally ' + rallyFirst.get(frame);
    }
    return {row, separator};
  });
  box.replaceChildren(...contactFeed.rows.flatMap(({row, separator}) => separator ? [separator, row] : [row]));
  contactFeed.count = -1;
  contactFeed.badgeIndex = null;
  updateActiveEvent(currentFrameIdx);
}

function toggleEventLog() {
  contactFeed.fullLog = !contactFeed.fullLog;
  const button = document.getElementById('eventLogToggle');
  button.setAttribute('aria-pressed', String(contactFeed.fullLog));
  button.textContent = contactFeed.fullLog ? 'Follow video' : 'Full log';
  contactFeed.count = -1;
  updateActiveEvent(currentFrameIdx);
}

function updateActiveEvent(frame = currentFrameIdx) {
  const state = contactFeed;
  const count = contactCountAtFrame(state.timeline, frame);
  const animate = state.count >= 0 && state.lastFrame !== null && frame > state.lastFrame &&
    frame - state.lastFrame <= FPS && !vid.paused && !vid.seeking;
  const mode = state.fullLog ? 'Full log' : vid.paused ? 'Paused' : 'Live';
  const modeBadge = document.getElementById('eventFeedMode');
  if (modeBadge.dataset.mode !== mode) {
    modeBadge.dataset.mode = mode;
    modeBadge.lastChild.textContent = ' ' + mode;
  }
  if (count !== state.count) {
    state.rows.forEach(({row, separator}, position) => {
      const visible = state.fullLog || position < count;
      const newlyVisible = row.hidden && visible;
      row.hidden = !visible;
      if (separator) separator.hidden = !visible;
      row.classList.toggle('future', position >= count);
      row.classList.toggle('active', position === count - 1);
      if (position === count - 1) row.setAttribute('aria-current', 'true');
      else row.removeAttribute('aria-current');
      if (newlyVisible && animate && position >= state.count) animateContact(row);
    });
    document.getElementById('eventFeedCount').textContent = `${count} / ${state.timeline.length} contacts`;
    const empty = document.getElementById('eventFeedEmpty');
    empty.hidden = state.fullLog ? state.timeline.length > 0 : count > 0;
    empty.textContent = state.timeline.length
      ? 'Play the video to follow each contact.' : 'No contact events in this result.';
    const active = state.rows[count - 1]?.row;
    const box = document.getElementById('eventList');
    // Scroll only this panel; the video and navigation must stay in place.
    if (active) {
      const rowRect = active.getBoundingClientRect();
      const listRect = box.getBoundingClientRect();
      if (rowRect.top < listRect.top || rowRect.bottom > listRect.bottom) {
        box.scrollTop += rowRect.bottom - listRect.bottom + 12;
      }
    } else box.scrollTop = 0;
    const latest = state.timeline[count - 1];
    document.getElementById('eventFeedStatus').textContent = latest
      ? `${contactLabel(latest.event.type)} · ${formatTime(latest.frame / FPS)} · Click any contact to replay.`
      : 'Contacts appear as the video plays. Click one to replay.';
    state.count = count;
  }
  state.lastFrame = frame;
  renderVideoEvent(frame, count, animate);
}

function renderVideoEvent(frame, count, animate) {
  const badge = document.getElementById('videoEventBadge');
  const latest = contactFeed.timeline[count - 1];
  const visible = latest && frame - latest.frame < 2.5 * FPS && document.getElementById('togEvents').checked;
  if (!visible) {
    badge.hidden = true;
    contactFeed.badgeIndex = null;
    return;
  }
  const changed = contactFeed.badgeIndex !== latest.index;
  const side = eventTeams[latest.index];
  const sideLabel = side === 'team_b' ? 'Near side' : side === 'team_a' ? 'Far side' : 'Unattributed';
  if (changed) {
    badge.style.setProperty('--event-color', evColor(latest.event.type));
    document.getElementById('videoEventLabel').textContent = contactLabel(latest.event.type);
    document.getElementById('videoEventMeta').textContent = sideLabel + (latest.event.inferred ? ' · Inferred' : '');
  }
  badge.hidden = false;
  if (changed && animate) animateContact(badge);
  contactFeed.badgeIndex = latest.index;
}
