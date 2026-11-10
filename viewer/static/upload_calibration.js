const uploadCalibration = (() => {
  const el = id => document.getElementById(id);
  const input = el('videoPath'), panel = el('calibrationEditor');
  const canvas = el('calibrationCanvas'), image = el('calibrationImage');
  const setup = el('calibrationSetup'), mode = el('calibrationMode');
  const status = el('calibrationStatus'), hint = el('calibrationHint');
  const refresh = el('refreshNetFloor');
  const netProfile = el('netProfile');
  const time = el('previewTime'), timeline = el('previewTimeline');
  let lastSecond = null;
  let original, edit, anchors, drag, path, pending;
  let requestNumber = 0;

  function close() {
    ++requestNumber;
    pending = false;
    edit = original = drag = null;
    setPreviewTime(0);
    panel.hidden = true;
    hideLoupe();
    fullFrame = null;
    refresh.disabled = false;
    status.textContent = '';
  }

  function setPreviewTime(seconds) {
    const value = Math.max(0, Math.min(lastSecond ?? Infinity, seconds));
    time.value = value;
    timeline.value = value;
  }

  function seek(seconds) {
    setPreviewTime(seconds);
    load({ keepEdits: true });
  }

  function reset() {
    edit = structuredClone(original);
    anchors = courtEdgeAnchors(edit.court_corners, canvas.width, canvas.height);
    drag = null;
    draw();
  }

  async function load({ keepEdits = false, setupId = '' } = {}) {
    const selectedPath = input.value.trim();
    if (!selectedPath) {
      status.textContent = 'Enter a video file path first.';
      return;
    }
    const number = ++requestNumber;
    const seconds = Number(time.value);
    pending = true;
    status.textContent = 'Reading preview frame…';
    try {
      const response = await fetch('/api/analysis_calibration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: selectedPath, setup_id: setupId,
          seconds,
        }),
      });
      const contentType = response.headers.get('Content-Type') || '';
      if (!contentType.includes('application/json')) {
        throw new Error(response.status === 404 || response.status === 405
          ? 'Preview endpoint is unavailable. Restart the viewer from the updated checkout and reload this page.'
          : `Viewer returned HTTP ${response.status} instead of a preview. Check the viewer server log.`);
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Preview failed (HTTP ${response.status}).`);
      if (number !== requestNumber) return;
      const w = data.frame_width, h = data.frame_height;
      if (keepEdits && (w !== canvas.width || h !== canvas.height)) {
        throw new Error('Frame size changed. Reopen the calibration preview.');
      }
      canvas.width = w;
      canvas.height = h;
      image.src = data.image;
      loadFullFrame(selectedPath, seconds);
      lastSecond = data.last_second;
      timeline.hidden = !(lastSecond > 0);
      timeline.max = lastSecond ?? 0;
      setPreviewTime(seconds);
      path = selectedPath;
      if (!keepEdits) {
        setup.replaceChildren(new Option('Default calibration', ''));
        for (const item of data.setups) setup.add(new Option(item.name, item.id));
        setup.value = setupId;
        original = {
          frame_width: w, frame_height: h,
          court_corners: data.court_corners ||
            [[w * .2, h * .4], [w * .8, h * .4], [w * .97, h * .94], [w * .03, h * .94]],
          net_tape: data.net_tape.length === 3
            ? [[data.net_tape[0], data.net_tape[2]], [data.net_tape[1], data.net_tape[2]]]
            : data.net_tape,
        };
        reset();
        status.textContent = data.court_corners ? '' : 'No saved court found. Align the starting outline before analyzing.';
      } else {
        draw();
        status.textContent = '';
      }
      panel.hidden = false;
      refresh.disabled = true;
    } catch (error) {
      if (number === requestNumber) status.textContent = error.message;
    } finally {
      if (number === requestNumber) pending = false;
    }
  }

  const netPoints = () => edit.net_tape;

  function handles() {
    if (mode.value === 'net') return netPoints();
    if (mode.value === 'corners') return edit.court_corners;
    return anchors.flat();
  }

  // Draws the outline, guides, net tape and handles. `map` turns native frame
  // coordinates into the target canvas; the magnifier passes a zoomed map and
  // thin strokes so the lines never hide the edge being aligned.
  function drawScene(ctx, map, unit, hollowHandles = false) {
    function line(points, color, dashed = false) {
      ctx.beginPath();
      points.map(map).forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.setLineDash(dashed ? [6 * unit, 4 * unit] : []);
      ctx.strokeStyle = '#14202c';
      ctx.lineWidth = 4 * unit;
      ctx.stroke();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2 * unit;
      ctx.stroke();
    }
    const valid = courtCornersAreConvex(edit.court_corners);
    line([...edit.court_corners, edit.court_corners[0]], valid ? '#45ffc3' : '#ff6974');
    const guides = valid && courtGuideLines(edit.court_corners);
    if (guides) {
      for (const [name, segment] of Object.entries(guides)) {
        if (name !== 'center_line') line(segment, '#fff', true);
      }
      // Show the same three segments that the completed analysis will draw.
      // The net handles remain in native image pixels; neither post detection
      // nor a second display coordinate system supplies their X coordinates.
      const [a, b] = guides.center_line;
      if (Math.abs(b[0] - a[0]) > 1e-6) {
        const floorY = x => a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]);
        const floor = netPoints().map(([x]) => [x, floorY(x)]);
        const top = netPoints().reduce((sum, [, y]) => sum + y, 0) / 2;
        const floorMid = (floor[0][1] + floor[1][1]) / 2;
        const height = netProfile.value === 'men' ? 2.43 : 2.34;
        const depth = (floorMid - top) / height;
        line(netPoints().map(([x, y]) => [x, y + depth]), '#ff8c00');
        line(floor, '#ff00ff');
      }
    }
    line(netPoints(), '#ffcb55');
    const handleColor = mode.value === 'net' ? '#ffcb55' : '#45ffc3';
    for (const [x, y] of handles().map(map)) {
      ctx.beginPath();
      ctx.arc(x, y, 6 * unit, 0, 2 * Math.PI);
      ctx.setLineDash([]);
      if (hollowHandles) {
        ctx.lineWidth = 2 * unit;
        ctx.strokeStyle = handleColor;
        ctx.stroke();
        continue;
      }
      ctx.fillStyle = handleColor;
      ctx.fill();
      ctx.strokeStyle = '#14202c';
      ctx.stroke();
    }
    return valid;
  }

  function draw() {
    if (!edit) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const valid = drawScene(ctx, p => p, Math.max(1, canvas.width / 900));
    if (loupeAt) drawLoupe(loupeAt);
    hint.textContent = !valid ? 'Court edges cross. Adjust the handles before analyzing.' : {
      lines: 'Drag either handle to rotate an edge. Drag between handles to move the whole edge.',
      corners: 'Drag visible corners. Switch to Court lines for corners outside the frame.',
      net: 'Drag each gold handle independently to match its end of the net tape. The floor divider follows the court centre line.',
    }[mode.value];
  }

  // Magnifier: a native-resolution crop around the dragged handle (or a handle
  // under the pointer), so lines meet the real court paint, not a blurry
  // downscaled preview. It sits in the corner away from the pointer.
  const loupe = el('calibrationLoupe');
  const LOUPE_CSS_PX = 220;
  let fullFrame = null, loupeAt = null;
  // Native pixels shown across the magnifier; the mouse wheel changes it.
  let loupeSpan = null;

  function loadFullFrame(selectedPath, seconds) {
    const frame = new Image();
    frame.onload = () => { if (fullFrame === frame && loupeAt) drawLoupe(loupeAt); };
    frame.src = '/api/analysis_calibration/frame?' +
      new URLSearchParams({ video_path: selectedPath, seconds: String(seconds) });
    fullFrame = frame;
  }

  function hideLoupe() {
    loupeAt = null;
    loupe.hidden = true;
  }

  function drawLoupe(p) {
    loupeAt = p;
    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(LOUPE_CSS_PX * dpr);
    if (loupe.width !== size) loupe.width = loupe.height = size;
    loupeSpan ??= Math.max(60, Math.round(canvas.width / 16));
    const span = loupeSpan, x0 = p[0] - span / 2, y0 = p[1] - span / 2, k = size / span;
    const ctx = loupe.getContext('2d');
    ctx.fillStyle = '#0b1118';
    ctx.fillRect(0, 0, size, size);
    const sharp = fullFrame && fullFrame.complete && fullFrame.naturalWidth === canvas.width;
    const source = sharp ? fullFrame : image;
    const scale = source.naturalWidth / canvas.width;
    if (source.naturalWidth) {
      ctx.imageSmoothingEnabled = !sharp;
      ctx.drawImage(source, x0 * scale, y0 * scale, span * scale, span * scale, 0, 0, size, size);
    }
    drawScene(ctx, ([x, y]) => [(x - x0) * k, (y - y0) * k], 0.6 * dpr, true);
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(255,255,255,.85)';
    ctx.lineWidth = dpr;
    const c = size / 2, gap = 6 * dpr, arm = 16 * dpr;
    ctx.beginPath();
    ctx.moveTo(c - gap - arm, c); ctx.lineTo(c - gap, c);
    ctx.moveTo(c + gap, c); ctx.lineTo(c + gap + arm, c);
    ctx.moveTo(c, c - gap - arm); ctx.lineTo(c, c - gap);
    ctx.moveTo(c, c + gap); ctx.lineTo(c, c + gap + arm);
    ctx.stroke();
    ctx.font = `600 ${11 * dpr}px system-ui`;
    ctx.fillStyle = 'rgba(255,255,255,.9)';
    ctx.fillText(`${Math.round(p[0])}, ${Math.round(p[1])}${sharp ? '' : ' · loading full frame…'}`, 6 * dpr, size - 7 * dpr);
    const left = p[0] < canvas.width / 2;
    loupe.style.left = left ? '' : '8px';
    loupe.style.right = left ? '8px' : '';
    loupe.hidden = false;
  }

  function nearestHandle(p) {
    const limit = 14 * canvas.width / canvas.getBoundingClientRect().width;
    let best = null, bestDistance = limit;
    for (const q of handles()) {
      const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d < bestDistance) { best = q; bestDistance = d; }
    }
    return best;
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return [(event.clientX - rect.left) * canvas.width / rect.width,
            (event.clientY - rect.top) * canvas.height / rect.height];
  }

  canvas.addEventListener('pointerdown', event => {
    if (!edit || pending || drag || event.button !== 0) return;
    const p = point(event);
    const limit = 14 * canvas.width / canvas.getBoundingClientRect().width;
    const distance = q => Math.hypot(p[0] - q[0], p[1] - q[1]);
    let closest = limit;
    handles().forEach((q, index) => {
      const d = distance(q);
      if (d < closest) {
        closest = d;
        drag = { index };
      }
    });
    if (!drag && mode.value === 'lines') {
      anchors.forEach(([a, b], edge) => {
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
        const d = distance([a[0] + t * dx, a[1] + t * dy]);
        if (t >= 0 && t <= 1 && d < closest) {
          closest = d;
          drag = { edge };
        }
      });
    }
    if (drag) {
      drag.last = p;
      drag.pointerId = event.pointerId;
      canvas.setPointerCapture(event.pointerId);
      drawLoupe(drag.index !== undefined ? handles()[drag.index] : p);
    }
  });

  canvas.addEventListener('pointermove', event => {
    if (!edit) return;
    if (!drag) {
      if (event.pointerType !== 'mouse') return;
      const handle = nearestHandle(point(event));
      if (handle) drawLoupe(handle); else hideLoupe();
      return;
    }
    if (drag.pointerId !== event.pointerId) return;
    const p = point(event);
    if (mode.value === 'net') {
      const net = edit.net_tape;
      const x = drag.index === 0
        ? Math.max(0, Math.min(net[1][0] - 1, p[0]))
        : Math.max(net[0][0] + 1, Math.min(canvas.width - 1, p[0]));
      net[drag.index] = [x, Math.max(1, Math.min(canvas.height - 1, p[1]))];
    } else if (mode.value === 'corners') {
      edit.court_corners[drag.index] = p;
    } else {
      if (drag.edge !== undefined) {
        anchors[drag.edge] = anchors[drag.edge].map(([x, y]) =>
          [x + p[0] - drag.last[0], y + p[1] - drag.last[1]]);
      } else {
        anchors[Math.floor(drag.index / 2)][drag.index % 2] = p;
      }
      edit.court_corners = cornersFromEdgeAnchors(anchors, edit.court_corners);
    }
    drag.last = p;
    loupeAt = drag.index !== undefined ? handles()[drag.index] : p;
    draw();
  });

  canvas.addEventListener('lostpointercapture', () => { drag = null; hideLoupe(); });
  canvas.addEventListener('pointerleave', () => { if (!drag) hideLoupe(); });
  canvas.addEventListener('wheel', event => {
    if (!loupeAt) return;
    event.preventDefault();
    const factor = event.deltaY > 0 ? 1.2 : 1 / 1.2;
    loupeSpan = Math.max(24, Math.min(canvas.width / 4, loupeSpan * factor));
    drawLoupe(loupeAt);
  }, { passive: false });
  input.addEventListener('input', close);
  el('previewCalibration').addEventListener('click', () => load({ keepEdits: Boolean(edit) }));
  el('seekCalibration').addEventListener('click', () => seek(Number(time.value)));
  time.addEventListener('keydown', event => {
    if (event.key === 'Enter') seek(Number(time.value));
  });
  timeline.addEventListener('input', () => { time.value = timeline.value; });
  timeline.addEventListener('change', () => seek(Number(timeline.value)));
  el('previewBack').addEventListener('click', () => seek(Number(time.value) - 5));
  el('previewForward').addEventListener('click', () => seek(Number(time.value) + 5));
  setup.addEventListener('change', () => load({ setupId: setup.value }));
  el('resetCalibration').addEventListener('click', reset);
  el('cancelCalibration').addEventListener('click', close);
  mode.addEventListener('change', () => {
    drag = null;
    hideLoupe();
    if (mode.value === 'lines') anchors = courtEdgeAnchors(edit.court_corners, canvas.width, canvas.height);
    draw();
  });
  netProfile.addEventListener('change', draw);

  return {
    analysisOptions(selectedPath) {
      if (pending) throw new Error('Wait for the calibration preview to load.');
      if (!edit || selectedPath !== path) return {};
      if (!courtCornersAreConvex(edit.court_corners)) throw new Error('Fix the court outline before analyzing.');
      return { calibration: structuredClone(edit) };
    },
  };
})();
