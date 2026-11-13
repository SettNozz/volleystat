// Shared, browser-local player profiles and source-scoped portrait choices.
const PlayerProfiles = {
  jobId: null, roster: [], portraits: {}, positions: {}, profiles: {}, photos: {}, draftPhotos: null,

  read(key, fallback) {
    try { return JSON.parse(localStorage.getItem(JSON.stringify(key))) ?? fallback; } catch { return fallback; }
  },
  write(key, value) {
    try { localStorage.setItem(JSON.stringify(key), JSON.stringify(value)); } catch {}
  },
  photoKey(id) { return ['player-photo', this.jobId, this.sourceVersion, id]; },
  configure(config) {
    Object.assign(this, config);
    this.profiles = this.read(['player-profiles', this.jobId], {});
    this.photos = {};
    for (const id of this.roster) {
      const saved = this.read(this.photoKey(id), 0);
      this.photos[id] = Number.isInteger(saved) && saved >= 0 && saved < (this.portraits[id] || 0) ? saved : 0;
    }
    this.setupEditor();
    this.refresh();
  },
  label(id) {
    const profile = this.profiles[id] || {};
    return profile.name || (profile.number ? '#' + profile.number : id);
  },
  reference(id) {
    if (this.label(id) === id) return '';
    const number = (this.profiles[id] || {}).number;
    return id + (number ? ' · #' + number : '');
  },
  photoUrl(id, candidate = this.photos[id] || 0) {
    return '/api/player-portrait/' + encodeURIComponent(this.jobId) + '/' + encodeURIComponent(id) + '?candidate=' + candidate;
  },
  identity(id, large = false) {
    const box = document.createElement('div');
    box.className = 'player-identity' + (large ? ' player-identity-large' : '');
    const portrait = document.createElement('span');
    portrait.className = 'portrait';
    portrait.setAttribute('aria-hidden', 'true');
    portrait.textContent = id;
    const photo = document.createElement('img');
    photo.dataset.playerPhoto = id;
    photo.alt = '';
    photo.loading = 'lazy';
    photo.onerror = () => { photo.hidden = true; };
    photo.onload = () => { photo.hidden = false; };
    if (this.portraits[id]) photo.src = this.photoUrl(id);
    else photo.hidden = true;
    portrait.append(photo);
    const copy = document.createElement('span');
    copy.className = 'player-identity-copy';
    const name = document.createElement('strong');
    name.dataset.playerLabel = id;
    name.textContent = this.label(id);
    const reference = document.createElement('small');
    reference.dataset.playerReference = id;
    reference.textContent = this.reference(id);
    const position = document.createElement('span');
    position.className = 'position';
    position.dataset.playerPosition = id;
    position.textContent = this.positions[(this.profiles[id] || {}).position] || '';
    copy.append(name, reference, position);
    box.append(portrait, copy);
    return box;
  },
  refresh() {
    for (const label of document.querySelectorAll('[data-player-label]')) label.textContent = this.label(label.dataset.playerLabel);
    for (const label of document.querySelectorAll('[data-player-reference]')) {
      const id = label.dataset.playerReference;
      label.textContent = this.reference(id);
    }
    for (const badge of document.querySelectorAll('[data-player-position]')) {
      badge.textContent = this.positions[(this.profiles[badge.dataset.playerPosition] || {}).position] || '';
    }
    for (const option of document.querySelectorAll('[data-player-option]')) {
      const id = option.dataset.playerOption;
      option.textContent = [this.label(id), this.positions[(this.profiles[id] || {}).position]].filter(Boolean).join(' · ');
    }
    for (const photo of document.querySelectorAll('[data-player-photo]')) {
      const id = photo.dataset.playerPhoto;
      const candidate = this.draftPhotos && photo.closest('#roster-picker') ? this.draftPhotos[id] : this.photos[id];
      const url = this.photoUrl(id, candidate || 0);
      if (!this.portraits[id]) { photo.hidden = true; continue; }
      if (photo.getAttribute('src') !== url) { photo.src = url; photo.hidden = false; }
    }
    const excel = document.querySelector('#export-excel input[name="profiles"]');
    if (excel) excel.value = JSON.stringify(this.profiles);
    window.dispatchEvent(new CustomEvent('player-profiles-changed'));
  },
  setupEditor() {
    const picker = document.getElementById('roster-picker');
    if (!picker || this.editorReady) return;
    this.editorReady = true;
    document.getElementById('roster-form').addEventListener('submit', event => {
      if (event.submitter?.value !== 'save') return;
      const profiles = {...this.profiles};
      for (const row of document.getElementById('roster-editor').children) {
        const id = row.dataset.player;
        profiles[id] = {
          name: row.querySelector('[data-player-name]').value.trim().slice(0, 64),
          number: row.querySelector('[data-player-number]').value.trim().slice(0, 8),
          position: row.querySelector('[data-player-pos]').value,
        };
      }
      this.profiles = profiles;
      this.photos = {...this.draftPhotos};
      this.write(['player-profiles', this.jobId], profiles);
      for (const id of this.roster) this.write(this.photoKey(id), this.photos[id]);
      this.draftPhotos = null;
      this.refresh();
    });
    picker.addEventListener('close', () => { this.draftPhotos = null; this.refresh(); });
    const photoPicker = document.getElementById('photo-picker');
    const preview = document.getElementById('photo-preview');
    preview.onload = () => {
      document.getElementById('save-photo').disabled = false;
      document.getElementById('photo-status').textContent = `Photo ${this.photoIndex + 1} of ${this.portraits[this.photoPlayer]}`;
    };
    preview.onerror = () => {
      document.getElementById('save-photo').disabled = true;
      document.getElementById('photo-status').textContent = 'Photo unavailable. Try the next one.';
    };
    document.getElementById('next-photo').onclick = () => this.stepPhoto(1);
    document.getElementById('prev-photo').onclick = () => this.stepPhoto(-1);
    document.getElementById('save-photo').onclick = () => {
      if (!this.draftPhotos) return;
      this.draftPhotos[this.photoPlayer] = this.photoIndex;
      this.refresh();
      photoPicker.close();
    };
    photoPicker.addEventListener('keydown', event => {
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault(); this.stepPhoto(event.key === 'ArrowRight' ? 1 : -1);
      }
    });
    window.addEventListener('storage', event => {
      if (event.key === JSON.stringify(['player-profiles', this.jobId]) || this.roster.some(id => event.key === JSON.stringify(this.photoKey(id)))) {
        this.configure({jobId: this.jobId});
      }
    });
  },
  openEditor() {
    if (!this.roster.length) return;
    const editor = document.getElementById('roster-editor');
    editor.replaceChildren();
    this.draftPhotos = {...this.photos};
    for (const id of this.roster) {
      const row = document.createElement('div');
      row.className = 'player-editor-row';
      row.dataset.player = id;
      const identity = this.identity(id);
      if (this.portraits[id] > 1) {
        const change = document.createElement('button');
        change.type = 'button';
        change.className = 'ctrl change-photo';
        change.textContent = 'Change photo';
        change.setAttribute('aria-label', 'Change photo for ' + id);
        change.onclick = () => this.openPhoto(id);
        identity.append(change);
      }
      const fields = document.createElement('div');
      fields.className = 'player-editor-fields';
      for (const [key, label, limit] of [['name', 'Name', 64], ['number', 'Jersey', 8]]) {
        const field = document.createElement('label');
        field.textContent = label;
        const input = document.createElement('input');
        input.dataset[key === 'name' ? 'playerName' : 'playerNumber'] = id;
        input.value = (this.profiles[id] || {})[key] || '';
        input.maxLength = limit;
        if (key === 'number') input.inputMode = 'numeric';
        input.setAttribute('aria-label', id + ' ' + (key === 'name' ? 'name' : 'jersey number'));
        field.append(input);
        fields.append(field);
      }
      const field = document.createElement('label');
      field.textContent = 'Position';
      const select = document.createElement('select');
      select.dataset.playerPos = id;
      select.setAttribute('aria-label', id + ' position');
      for (const [value, label] of [['', '—'], ...Object.entries(this.positions)]) {
        const option = document.createElement('option');
        option.value = value; option.textContent = label; select.append(option);
      }
      const position = (this.profiles[id] || {}).position;
      select.value = position in this.positions ? position : '';
      field.append(select); fields.append(field);
      row.append(identity, fields); editor.append(row);
    }
    this.refresh();
    document.getElementById('roster-picker').showModal();
  },
  openPhoto(id) {
    if (!this.draftPhotos || !document.getElementById('roster-picker').open) return;
    this.photoPlayer = id;
    this.photoIndex = this.draftPhotos[id] || 0;
    document.getElementById('photo-title').textContent = 'Change photo · ' + this.label(id);
    document.getElementById('photo-picker').showModal();
    this.stepPhoto(0);
  },
  stepPhoto(step) {
    const count = this.portraits[this.photoPlayer];
    this.photoIndex = (this.photoIndex + step + count) % count;
    document.getElementById('save-photo').disabled = true;
    document.getElementById('photo-status').textContent = 'Loading photo…';
    document.getElementById('photo-preview').src = this.photoUrl(this.photoPlayer, this.photoIndex);
  },
};
