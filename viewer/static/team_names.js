// Browser-local team names shared by static labels and dynamic viewer panels.
window.TeamNames = (() => {
  const memory = new Map();
  const prepared = new WeakSet();
  const fallbacks = new WeakMap();
  const sideOf = value => ({a: 'a', b: 'b', team_a: 'a', team_b: 'b'})[value];
  const job = () => document.querySelector('[data-team-job]')?.dataset.teamJob || '';
  const keyFor = (jobId, side) => JSON.stringify(['court-team-name', jobId, side]);
  const clean = value => String(value || '').trim().slice(0, 32);

  function read(jobId, side) {
    const key = keyFor(jobId, side);
    if (!memory.has(key)) {
      try { memory.set(key, clean(localStorage.getItem(key))); }
      catch { memory.set(key, ''); }
    }
    return memory.get(key);
  }

  function label(side, fallback = 'Unattributed', jobId = job()) {
    const normalized = sideOf(side);
    return normalized ? read(jobId, normalized) || fallback : fallback;
  }

  function refresh(origin) {
    for (const field of document.querySelectorAll('[data-team-job][data-team-side]')) {
      const editable = field instanceof HTMLInputElement;
      if (!fallbacks.has(field)) fallbacks.set(field, editable ? field.defaultValue : field.textContent.trim());
      const name = label(field.dataset.teamSide, fallbacks.get(field), field.dataset.teamJob);
      if (field !== origin) {
        if (editable) field.value = name;
        else field.textContent = name;
      }
      if (editable) {
        field.setAttribute('aria-label', name + ' name');
        field.style.width = Math.min(24, Math.max(7, Array.from(field.value).length + 1)) + 'ch';
        field.title = name;
      }
      if (editable && !prepared.has(field)) {
        prepared.add(field);
        field.addEventListener('input', () => {
          const key = keyFor(field.dataset.teamJob, sideOf(field.dataset.teamSide));
          const value = clean(field.value);
          memory.set(key, value);
          try { localStorage.setItem(key, value); } catch {}
          refresh(field);
          window.dispatchEvent(new CustomEvent('team-names-changed'));
        });
        field.addEventListener('blur', () => refresh());
      }
    }
  }

  window.addEventListener('storage', event => {
    if (event.key === null) memory.clear();
    else {
      let parts;
      try { parts = JSON.parse(event.key); } catch { return; }
      if (!Array.isArray(parts) || parts[0] !== 'court-team-name') return;
      memory.set(event.key, clean(event.newValue));
    }
    refresh();
    window.dispatchEvent(new CustomEvent('team-names-changed'));
  });

  // Names used in HTML table/legend templates must remain text.
  function html(side, fallback) {
    return label(side, fallback).replace(/[&<>"']/g, c =>
      ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[c]);
  }

  return {label, html, refresh};
})();
window.TeamNames.refresh();
