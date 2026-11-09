// New analysis page: drop a video (or pick one) instead of typing its path.
// Browsers never expose a dropped file's local path, so the bytes are copied
// to data/uploads by /api/upload_video and the stored path fills the field.
(() => {
  const zone = document.getElementById('dropZone');
  const input = document.getElementById('videoPath');
  const picker = document.getElementById('videoFile');
  const box = document.getElementById('uploadProgress');
  const fill = document.getElementById('uploadFill');
  const text = document.getElementById('uploadText');
  const locked = ['analyzeBtn', 'previewCalibration'].map(id => document.getElementById(id));
  let busy = false;

  const gb = bytes => bytes < 2 ** 30
    ? Math.max(0.1, bytes / 2 ** 20).toFixed(1) + ' MB'
    : (bytes / 2 ** 30).toFixed(2) + ' GB';

  function show(message, fraction = null, error = false) {
    box.hidden = false;
    box.classList.toggle('error', error);
    fill.parentElement.hidden = fraction === null;
    if (fraction !== null) fill.style.width = (100 * fraction).toFixed(1) + '%';
    text.textContent = message;
  }

  let lockedBefore = [];
  function setBusy(value) {
    busy = value;
    // Restore each button as it was: a running analysis keeps Analyze disabled.
    if (value) lockedBefore = locked.map(button => button && button.disabled);
    locked.forEach((button, i) => { if (button) button.disabled = value || lockedBefore[i]; });
  }

  function usePath(path, message) {
    input.value = path;
    // Same event as typing: closes a calibration preview of the old video.
    input.dispatchEvent(new Event('input'));
    show(message);
  }

  function send(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/upload_video');
      xhr.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      const started = performance.now();
      xhr.upload.onprogress = event => {
        if (!event.lengthComputable) return;
        const seconds = (performance.now() - started) / 1000;
        const rate = seconds > 0.5 ? event.loaded / seconds : 0;
        const eta = rate ? Math.round((event.total - event.loaded) / rate) : null;
        show(`Copying ${file.name}: ${gb(event.loaded)} / ${gb(event.total)}` +
             (eta !== null ? ` · ${eta} s left` : ''), event.loaded / event.total);
      };
      xhr.onload = () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch {}
        if (xhr.status >= 200 && xhr.status < 300 && data.path) resolve(data);
        else reject(new Error(data.error || `Upload failed (HTTP ${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error('Upload failed: the viewer server is unreachable.'));
      xhr.send(file);
    });
  }

  async function upload(file) {
    if (!file || busy) return;
    setBusy(true);
    try {
      show(`Checking ${file.name}…`);
      const query = new URLSearchParams({ name: file.name, size: String(file.size) });
      const check = await fetch('/api/upload_video?' + query).then(r => r.json());
      if (check.error) throw new Error(check.error);
      if (check.path) {
        usePath(check.path, `Using the earlier copy of ${file.name}.`);
        return;
      }
      show(`Copying ${file.name}…`, 0);
      const data = await send(file);
      usePath(data.path, `Copied ${file.name} (${gb(file.size)}).`);
    } catch (error) {
      show(error.message, null, true);
    } finally {
      setBusy(false);
    }
  }

  const hasFiles = event => Array.from(event.dataTransfer?.types || []).includes('Files');
  let depth = 0;
  // Catch drops anywhere on the page so a missed target never navigates away
  // to the raw video file.
  document.addEventListener('dragenter', event => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth += 1;
    zone.classList.add('dragging');
  });
  document.addEventListener('dragleave', event => {
    if (!hasFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) zone.classList.remove('dragging');
  });
  document.addEventListener('dragover', event => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = busy ? 'none' : 'copy';
  });
  document.addEventListener('drop', event => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    zone.classList.remove('dragging');
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 1) show('Drop one video at a time; using the first.');
    upload(files[0]);
  });
  document.getElementById('browseVideo').addEventListener('click', () => picker.click());
  picker.addEventListener('change', () => {
    upload(picker.files[0]);
    picker.value = '';
  });
})();
