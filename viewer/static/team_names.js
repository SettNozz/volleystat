for (const field of document.querySelectorAll('[data-team-job][data-team-side]')) {
  const key = JSON.stringify(['court-team-name', field.dataset.teamJob, field.dataset.teamSide]);
  const editable = field instanceof HTMLInputElement;
  const fallback = editable ? field.defaultValue : field.textContent.trim();
  const display = name => {
    if (editable) field.value = name || fallback;
    else field.textContent = name || fallback;
  };
  try {
    const saved = localStorage.getItem(key);
    if (saved !== null) display(saved);
  } catch {}

  if (editable) {
    field.addEventListener('input', () => {
      try { localStorage.setItem(key, field.value.trim()); } catch {}
    });
  }
  window.addEventListener('storage', event => {
    if (event.key === key) display(event.newValue);
  });
}
