const uploadCalibration = (() => {
  const el = id => document.getElementById(id);
  const input = el('videoPath'), panel = el('calibrationEditor');
  const canvas = el('calibrationCanvas'), image = el('calibrationImage');
  const setup = el('calibrationSetup'), mode = el('calibrationMode');
  const status = el('calibrationStatus'), hint = el('calibrationHint');
  const refresh = el('refreshNetFloor');
  let original, edit, anchors, drag, path, pending;
  let requestNumber = 0;

  function close() {
    ++requestNumber;
    pending = false;
    edit = original = drag = null;
    panel.hidden = true;
    refresh.disabled = false;
    status.textContent = '';
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
    pending = true;
    status.textContent = 'Reading preview frame…';
    try {
      const response = await fetch('/api/analysis_calibration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: selectedPath, setup_id: setupId,
          seconds: Number(el('previewTime').value),
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
      path = selectedPath;
      if (!keepEdits) {
        setup.replaceChildren(new Option('Default calibration', ''));
        for (const item of data.setups) setup.add(new Option(item.name, item.id));
        setup.value = setupId;
        original = {
          frame_width: w, frame_height: h,
          court_corners: data.court_corners ||
            [[w * .2, h * .4], [w * .8, h * .4], [w * .97, h * .94], [w * .03, h * .94]],
          net_tape: data.net_tape,
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

  const netPoints = () => {
    const [left, right, y] = edit.net_tape;
    return [[left, y], [right, y]];
  };

  function handles() {
    if (mode.value === 'net') return netPoints();
    if (mode.value === 'corners') return edit.court_corners;
    return anchors.flat();
  }

  function draw() {
    if (!edit) return;
    const ctx = canvas.getContext('2d');
    const unit = Math.max(1, canvas.width / 900);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    function line(points, color, dashed = false) {
      ctx.beginPath();
      points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
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
      for (const segment of Object.values(guides)) line(segment, '#fff', true);
    }
    line(netPoints(), '#ffcb55');
    for (const [x, y] of handles()) {
      ctx.beginPath();
      ctx.arc(x, y, 6 * unit, 0, 2 * Math.PI);
      ctx.fillStyle = mode.value === 'net' ? '#ffcb55' : '#45ffc3';
      ctx.fill();
      ctx.setLineDash([]);
      ctx.strokeStyle = '#14202c';
      ctx.stroke();
    }
    hint.textContent = !valid ? 'Court edges cross. Adjust the handles before analyzing.' : {
      lines: 'Drag either handle to rotate an edge. Drag between handles to move the whole edge.',
      corners: 'Drag visible corners. Switch to Court lines for corners outside the frame.',
      net: 'Drag the gold handles to set tape height and endpoints. The floor divider follows the court centre line.',
    }[mode.value];
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
    }
  });

  canvas.addEventListener('pointermove', event => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const p = point(event);
    if (mode.value === 'net') {
      const net = edit.net_tape;
      net[drag.index] = drag.index === 0
        ? Math.max(0, Math.min(net[1] - 1, p[0]))
        : Math.max(net[0] + 1, Math.min(canvas.width - 1, p[0]));
      net[2] = Math.max(1, Math.min(canvas.height - 1, p[1]));
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
    draw();
  });

  canvas.addEventListener('lostpointercapture', () => { drag = null; });
  input.addEventListener('input', close);
  el('previewCalibration').addEventListener('click', () => load({ keepEdits: Boolean(edit) }));
  el('seekCalibration').addEventListener('click', () => load({ keepEdits: true }));
  setup.addEventListener('change', () => load({ setupId: setup.value }));
  el('resetCalibration').addEventListener('click', reset);
  el('cancelCalibration').addEventListener('click', close);
  mode.addEventListener('change', () => {
    drag = null;
    if (mode.value === 'lines') anchors = courtEdgeAnchors(edit.court_corners, canvas.width, canvas.height);
    draw();
  });

  return {
    analysisOptions(selectedPath) {
      if (pending) throw new Error('Wait for the calibration preview to load.');
      if (!edit || selectedPath !== path) return {};
      if (!courtCornersAreConvex(edit.court_corners)) throw new Error('Fix the court outline before analyzing.');
      return { calibration: edit };
    },
  };
})();
