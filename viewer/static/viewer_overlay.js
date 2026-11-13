// Video overlay drawing: ball/trail, players, net/court, events (redraw).
// Moved verbatim from viewer/templates/viewer.html. Classic script: shares
// globals with the page boot constants and viewer_page.js. Declarations
// only; listeners and init live in viewer_page.js.

/* ── Canvas drawing (overlays) ───────────────────────── */
const NET_COLOR  = 'rgba(255,220,0,0.9)';
const BALL_COLOR = '#00ff44';

let _lastAnn = null;

function ballView(annotation) {
  const selector = document.getElementById('ballPathMode');
  const mode = selector ? selector.value : 'active';
  if (!annotation) return {point: null, source: 'missing', confidence: 0, mode};
  if (mode === 'raw') {
    return {
      point: annotation.ball_raw || null,
      source: annotation.ball_raw ? 'detected' : 'missing',
      confidence: Number(annotation.ball_raw_confidence || 0),
      mode,
    };
  }
  if (mode === 'conservative' || mode === 'contact_aware') {
    const key = 'ball_' + mode;
    return {
      point: annotation[key] || null,
      source: annotation[key + '_source'] || 'missing',
      confidence: Number(annotation[key + '_confidence'] || 0),
      mode,
    };
  }
  return {
    point: annotation.ball || null,
    source: annotation.ball_source || (annotation.ball_interp ? 'interpolated_short' : annotation.ball ? 'detected' : 'missing'),
    confidence: Number(annotation.ball_confidence || 0),
    mode,
  };
}

function redraw() {
  renderServeScore();
  renderAttackHeight();
  updateActiveEvent();
  if (!canvas.width) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!annotations) return;
  drawSetTrajectories();

  const ann = annotations[currentFrameIdx] || _lastAnn;
  if (!ann) return;
  if (annotations[currentFrameIdx]) _lastAnn = ann;

  const vw = vid.videoWidth || (window._annData && window._annData.frame_width) || 1280;
  const vh = vid.videoHeight || (window._annData && window._annData.frame_height) || 720;
  const sx = canvas.width / vw;
  const sy = canvas.height / vh;

  function px(x) { return x * sx; }
  function py(y) { return y * sy; }

  const showBall    = document.getElementById('togBall').checked;
  const showTrail   = document.getElementById('togTrail').checked;
  const showNet     = document.getElementById('togNet').checked;
  const showPlayers = document.getElementById('togPlayers').checked;
  const showState   = document.getElementById('togState').checked;
  const showModel   = document.getElementById('togModel').checked;
  const showAttackHeatmap = document.getElementById('togAttackHeatmap').checked;
  const showCourt   = document.getElementById('togCourt').checked;
  const currentBall = ballView(ann);

  if ((showCourt || courtEditor.active) && window._annData) {
    const corners = window._annData.court_corners;
    if (Array.isArray(corners) && corners.length === 4) {
      ctx.strokeStyle = 'rgba(0, 255, 170, 0.95)';
      ctx.fillStyle = 'rgba(0, 255, 170, 0.035)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px(corners[0][0]), py(corners[0][1]));
      for (let i = 1; i < corners.length; i++) {
        ctx.lineTo(px(corners[i][0]), py(corners[i][1]));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // The far baseline hides behind the net and the players, so the corners
      // alone give the user nothing to check the depth scale against. These
      // are the lines painted on every court.
      if (courtEditor.active) {
        const guides = courtGuideLines(corners);
        if (guides) {
          ctx.setLineDash([9, 6]);
          ctx.lineWidth = 2;
          for (const [name, segment] of Object.entries(guides)) {
            ctx.strokeStyle = name === 'center_line'
              ? 'rgba(255, 210, 0, 0.95)' : 'rgba(56, 189, 248, 0.8)';
            ctx.beginPath();
            ctx.moveTo(px(segment[0][0]), py(segment[0][1]));
            ctx.lineTo(px(segment[1][0]), py(segment[1][1]));
            ctx.stroke();
          }
          ctx.setLineDash([]);
        }
      }
      if (courtEditor.active && courtEditor.mode === 'lines' && courtEditor.anchors) {
        courtEditor.anchors.forEach((anchor, edge) => {
          const dragging = courtEditor.drag && courtEditor.drag.edge === edge;
          ctx.strokeStyle = dragging ? '#f97316' : 'rgba(0, 255, 170, 0.95)';
          ctx.lineWidth = dragging ? 5 : 4;
          ctx.beginPath();
          ctx.moveTo(px(anchor[0][0]), py(anchor[0][1]));
          ctx.lineTo(px(anchor[1][0]), py(anchor[1][1]));
          ctx.stroke();
          anchor.forEach((handle, index) => {
            const active = dragging && courtEditor.drag.handle === index;
            ctx.fillStyle = active ? '#f97316' : '#00ffae';
            ctx.strokeStyle = '#0f172a';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(px(handle[0]), py(handle[1]), 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          });
          const midX = (anchor[0][0] + anchor[1][0]) / 2;
          const midY = (anchor[0][1] + anchor[1][1]) / 2;
          ctx.fillStyle = '#0f172a';
          ctx.strokeStyle = 'rgba(0, 255, 170, 0.95)';
          ctx.lineWidth = 3;
          ctx.font = `bold ${Math.max(11, Math.round(12 * sx))}px system-ui`;
          ctx.strokeText(EDGE_NAMES[edge], px(midX) + 10, py(midY) - 8);
          ctx.fillText(EDGE_NAMES[edge], px(midX) + 10, py(midY) - 8);
        });
        // A corner outside the picture still has to be findable, so it is
        // pinned to the frame edge with its true offset spelled out.
        corners.forEach((point, index) => {
          const outside = point[0] < 0 || point[1] < 0
            || point[0] > window._annData.frame_width
            || point[1] > window._annData.frame_height;
          if (!outside) return;
          const cx = Math.max(14, Math.min(canvas.width - 14, px(point[0])));
          const cy = Math.max(14, Math.min(canvas.height - 14, py(point[1])));
          ctx.fillStyle = 'rgba(249, 115, 22, 0.9)';
          ctx.beginPath();
          ctx.arc(cx, cy, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.font = `bold ${Math.max(10, Math.round(11 * sx))}px system-ui`;
          ctx.fillText(
            `${['A', 'B', 'C', 'D'][index]} off-frame`,
            cx + 9, cy + 4,
          );
        });
      } else if (courtEditor.active) {
        const names = ['A', 'B', 'C', 'D'];
        corners.forEach((point, index) => {
          ctx.fillStyle = index === courtEditor.dragIndex ? '#f97316' : '#00ffae';
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(px(point[0]), py(point[1]), 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = '#0f172a';
          ctx.font = `bold ${Math.max(11, Math.round(13 * sx))}px system-ui`;
          ctx.fillText(names[index], px(point[0]) + 11, py(point[1]) - 10);
        });
      }
    }
  }

  if (showNet && window._annData) {
    const d = window._annData;
    // Drawn tape endpoints own both coordinates. Post X is a fallback for
    // automatic detections, whose tape is represented by a height alone.
    const postLeftX = !d.net_tape_endpoint_y && d.net_floor_left ? d.net_floor_left[0] : d.net_x_left;
    const postRightX = !d.net_tape_endpoint_y && d.net_floor_right ? d.net_floor_right[0] : d.net_x_right;
    const xL = px(Math.min(postLeftX, postRightX));
    const xR = px(Math.max(postLeftX, postRightX));

    if (d.net_tape_top != null && d.net_tape_bottom != null) {
      const [leftY, rightY] = d.net_tape_endpoint_y || [d.net_tape_top, d.net_tape_top];
      const yTop = py(leftY), yTopRight = py(rightY);
      const depth = py(d.net_tape_bottom) - py(d.net_tape_top);
      const yBot = yTop + depth, yBotRight = yTopRight + depth;

      ctx.fillStyle = 'rgba(255, 220, 80, 0.08)';
      ctx.beginPath();
      ctx.moveTo(xL, yTop);
      ctx.lineTo(xR, yTopRight);
      ctx.lineTo(xR, yBotRight);
      ctx.lineTo(xL, yBot);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xL, yTop);
      ctx.lineTo(xR, yTopRight);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(255, 140, 0, 0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xL, yBot);
      ctx.lineTo(xR, yBotRight);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.font = `bold ${Math.round(12*sx)}px system-ui`;
      ctx.fillText('NET', xL + 5, yTop - 4);
    }

    ctx.strokeStyle = NET_COLOR;
    ctx.lineWidth = 2;
    ctx.setLineDash([Math.round(10*sx), Math.round(6*sx)]);
    ctx.beginPath();
    ctx.moveTo(xL, py(d.net_tape_endpoint_y?.[0] ?? d.net_y));
    ctx.lineTo(xR, py(d.net_tape_endpoint_y?.[1] ?? d.net_y));
    ctx.stroke();
    ctx.setLineDash([]);
    if (d.net_tape_top == null) {
      ctx.fillStyle = NET_COLOR;
      ctx.font = `bold ${Math.round(14*sx)}px system-ui`;
      ctx.fillText('NET', xL, py(d.net_y) - 8);
    }

    if (d.net_floor_left && d.net_floor_right) {
      const [ax, ay] = d.net_floor_left;
      const [bx, by] = d.net_floor_right;
      ctx.strokeStyle = 'rgba(255, 0, 255, 0.92)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px(ax), py(ay));
      ctx.lineTo(px(bx), py(by));
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 0, 255, 0.98)';
      ctx.font = `bold ${Math.max(11, Math.round(13 * sx))}px system-ui`;
      ctx.fillText('A', px(ax) + 6, py(ay) - 7);
      ctx.fillText('B', px(bx) + 6, py(by) - 7);
    }
  }

  if (showBall) {
    const TRAIL_FRAMES = 40;
    const MAX_TRAIL    = 20;

    const trailPos = [];
    for (let fi = currentFrameIdx - 1;
         fi >= Math.max(0, currentFrameIdx - TRAIL_FRAMES) && trailPos.length < MAX_TRAIL;
         fi--) {
      const a = annotations[fi];
      const viewed = ballView(a);
      if (a && viewed.point) {
        trailPos.push({ x: viewed.point[0], y: viewed.point[1], age: currentFrameIdx - fi,
                        newTrack: !!a.ball_new });
        // The tracker re-acquired the ball here; anything older belongs to a
        // different hypothesis and must not be joined to this one.
        if (a.ball_new) break;
      }
    }

    // The offline pipeline interpolates at most four missing frames (a span
    // of five). Longer reacquisitions are separate hypotheses and drawing a
    // chord between them creates the full-image jumps seen on 4K footage.
    const GAP_BREAK = 5;
    // Same physical speed ceiling as the offline DP (90 px/frame at 1280p).
    // This display-side guard also protects old annotation files that did not
    // preserve solver restart markers.
    const MAX_TRAIL_STEP = 90 * (vw / 1280);
    if (showTrail && trailPos.length > 0 && currentBall.point) {
      const pts = [...trailPos].reverse();
      pts.push({ x: currentBall.point[0], y: currentBall.point[1], age: 0,
                 newTrack: !!ann.ball_new });

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (let i = 1; i < pts.length; i++) {
        const p0 = pts[i - 1];
        const p1 = pts[i];
        const frameGap = p0.age - p1.age;
        const step = Math.hypot(p1.x - p0.x, p1.y - p0.y) /
          Math.max(frameGap, 1);
        if (frameGap > GAP_BREAK || p1.newTrack || step > MAX_TRAIL_STEP) continue;
        const frac  = 1 - p1.age / TRAIL_FRAMES;
        const alpha = Math.pow(frac, 1.1) * 0.9;
        ctx.strokeStyle = `rgba(0,255,68, ${alpha})`;
        ctx.lineWidth   = 1 + frac * 2.5;
        ctx.beginPath();
        ctx.moveTo(px(p0.x), py(p0.y));
        ctx.lineTo(px(p1.x), py(p1.y));
        ctx.stroke();
      }
    }

    if (currentBall.point) {
      const [bx, by] = currentBall.point;
      // Keep the display uniform; annotations retain coordinate provenance.
      ctx.shadowColor = BALL_COLOR;
      ctx.shadowBlur  = 5;
      ctx.fillStyle   = BALL_COLOR;
      ctx.beginPath();
      ctx.arc(px(bx), py(by), 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur  = 0;
    } else if (ann.ball_out_of_frame && currentBall.mode === 'active') {
      ctx.fillStyle = '#f59e0b';
      ctx.font = `bold ${Math.max(12, Math.round(15 * sx))}px system-ui`;
      ctx.fillText('BALL OUT OF FRAME', 20, canvas.height - 24);
    }
  }

  if (showAttackHeatmap && window._annData && window._annData.attack_heatmap) {
    const d = window._annData;
    const counts = attackZoneCounts(d.attack_heatmap);
    let floorLeft = d.net_floor_left || [d.net_x_left, d.net_y];
    let floorRight = d.net_floor_right || [d.net_x_right, d.net_y];
    if (floorLeft[0] > floorRight[0]) [floorLeft, floorRight] = [floorRight, floorLeft];
    const topY = d.net_tape_top != null ? d.net_tape_top : d.net_y;
    const ids = ['left', 'center', 'right'];
    const maxCount = Math.max(1, ...ids.map(id => counts[id]));

    for (let index = 0; index < 3; index++) {
      const t0 = index / 3;
      const t1 = (index + 1) / 3;
      const x0 = floorLeft[0] + (floorRight[0] - floorLeft[0]) * t0;
      const x1 = floorLeft[0] + (floorRight[0] - floorLeft[0]) * t1;
      const floorY0 = floorLeft[1] + (floorRight[1] - floorLeft[1]) * t0;
      const floorY1 = floorLeft[1] + (floorRight[1] - floorLeft[1]) * t1;
      const count = counts[ids[index]];
      const alpha = count ? 0.12 + 0.38 * count / maxCount : 0.035;

      ctx.fillStyle = `rgba(255, 125, 0, ${alpha})`;
      ctx.strokeStyle = count ? 'rgba(255, 175, 45, 0.9)' : 'rgba(255, 175, 45, 0.22)';
      ctx.lineWidth = count ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(px(x0), py(topY));
      ctx.lineTo(px(x1), py(topY));
      // Mirror the near-side floor line across the net. Attack zones belong
      // above the net in the visual overlay, not on the camera-side floor.
      const mirroredY1 = 2 * topY - floorY1;
      const mirroredY0 = 2 * topY - floorY0;
      ctx.lineTo(px(x1), py(mirroredY1));
      ctx.lineTo(px(x0), py(mirroredY0));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,245,220,0.95)';
      ctx.font = '700 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        `${ids[index].toUpperCase()} · ${count}`,
        px((x0 + x1) / 2),
        py(topY) - 8,
      );
    }
    ctx.textAlign = 'left';
  }

  if (showPlayers) {
    const COCO_BONES = [
      [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12],
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1],
      [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
    ];
    for (const [poseIndex, pose] of (ann.poses || []).entries()) {
      const hasIdentity = Object.prototype.hasOwnProperty.call(pose, 'player_id');
      // A roster identity must not follow the legacy last-annotation fallback.
      if (hasIdentity && Number(ann.f) !== currentFrameIdx) continue;
      const keypoints = pose.kp || [];
      const isNearTeam = pose.team === 'b' || pose.team === 'team_b';
      const { color, width, radius } = playerPoseStyle(pose, poseIndex, currentFrameIdx);

      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      for (const [a, b] of COCO_BONES) {
        const pa = keypoints[a], pb = keypoints[b];
        if (!pa || !pb || pa[2] < 0.2 || pb[2] < 0.2) continue;
        ctx.beginPath();
        ctx.moveTo(px(pa[0]), py(pa[1]));
        ctx.lineTo(px(pb[0]), py(pb[1]));
        ctx.stroke();
      }
      ctx.fillStyle = color;
      for (const point of keypoints) {
        if (!point || point[2] < 0.2) continue;
        ctx.beginPath();
        ctx.arc(px(point[0]), py(point[1]), radius, 0, Math.PI * 2);
        ctx.fill();
      }
      if ((isNearTeam || hasIdentity) && (hasIdentity || pose.id != null)) {
        const head = keypoints.find(p => p && p[2] >= 0.2);
        const anchor = head || (pose.bb ? [pose.bb[0], pose.bb[1]] : null);
        if (anchor) {
          ctx.font = `bold ${Math.max(10, Math.round(12 * sx))}px system-ui`;
          ctx.fillStyle = color;
          ctx.fillText(hasIdentity ? (pose.player_id || 'P?') : `track #${pose.id}`,
                       px(anchor[0]) + 5, py(anchor[1]) - 5);
        }
      }
      if (pose.team_prior && pose.bb && (!selectedPlayerId || pose.player_id === selectedPlayerId)) {
        ctx.font = `bold ${Math.max(9, Math.round(10 * sx))}px system-ui`;
        ctx.fillStyle = '#ff42e6';
        ctx.fillText('BACK/NET prior', px(pose.bb[0]), py(pose.bb[1]) - 6);
      }
    }
  }

  if (showPlayers) drawPoseEstimates(ctx, ann, currentFrameIdx, px, py, sx);
  if (showPlayers) drawPlayerContactCandidate(ctx, ann, currentFrameIdx, px, py, sx);

  let stateY = 70;
  if (showState) {
    const lc = ann.lifecycle;
    const lifecycleLabels = {
      unconfirmed: 'WAITING FOR SERVE',
      serve_candidate: 'SERVE CANDIDATE',
      serve_confirmed: 'SERVE CONFIRMED',
      reception_pending: 'RECEPTION CONFIRMED',
      playable_rally: 'PLAYABLE RALLY',
      ace_no_reception: 'ACE / NO RECEPTION',
      reception_failed: 'RECEPTION FAILED',
      serve_fault: 'SERVE FAULT',
    };
    const stateColor = lc ? ({
      unconfirmed: '#00eeff',
      serve_candidate: '#ffaa00',
      serve_confirmed: '#eab308',
      reception_pending: '#10b981',
      playable_rally: '#00ff88',
      ace_no_reception: '#facc15',
      reception_failed: '#ff6666',
      serve_fault: '#ff6666',
    }[lc.status] || '#00eeff') : ({
      'serve_prep':   '#ffaa00',
      'serve_ready':  '#00eeff',
      'serving':      '#00eeff',
      'rally_active': '#00ff88',
      'rally_end':    '#ff6666',
    }[ann.state] || '#00eeff');
    ctx.font = `bold ${Math.max(14, Math.round(22 * sx))}px system-ui`;
    ctx.fillStyle = stateColor;
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 6;
    let line1;
    if (lc) {
      line1 = lifecycleLabels[lc.status] || String(lc.status || '').replace(/_/g, ' ').toUpperCase();
      if (['reception_pending', 'playable_rally'].includes(lc.status) && ann.current_team) {
        line1 += ` · ${ann.current_team} [${ann.contacts || 0}]`;
      }
    } else {
      line1 = (ann.state || '').replace(/_/g, ' ').toUpperCase();
      if (ann.phase) line1 += ' · ' + (ann.phase || '').replace(/_/g, ' ');
    }
    if (showModel && ann.event_model != null && ann.event_model !== '-') {
      line1 += '  |  Model: ' + (ann.event_model || '').replace(/_/g, ' ');
    }
    ctx.fillText(line1, px(20), py(stateY));
    ctx.shadowBlur = 0;
    stateY += 28;
  } else if (showModel && ann.event_model != null && ann.event_model !== '-') {
    ctx.font = `bold ${Math.max(14, Math.round(22 * sx))}px system-ui`;
    ctx.fillStyle = '#f0b030';
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 6;
    ctx.fillText('Model: ' + (ann.event_model || '').replace(/_/g, ' '), px(20), py(stateY));
    ctx.shadowBlur = 0;
    stateY += 28;
  }

  if (showModel && ann.spotting_model) {
    ctx.font = `bold ${Math.max(13, Math.round(20 * sx))}px system-ui`;
    ctx.fillStyle = ann.spotting_peak ? '#ff4dd2' : '#f58adf';
    ctx.shadowColor = '#000';
    ctx.shadowBlur = 6;
    const spottingScore = Number(ann.spotting_confidence || 0).toFixed(2);
    ctx.fillText(
      'Spotting: ' + String(ann.spotting_model).replace(/_/g, ' ') +
      ' (' + spottingScore + ')',
      px(20), py(stateY),
    );
    ctx.shadowBlur = 0;
    stateY += 28;
  }

  ctx.font = `${Math.round(11 * sx)}px monospace`;
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.textAlign = 'right';
  ctx.fillText(`f${currentFrameIdx}`, canvas.width - 4, canvas.height - 4);
  ctx.textAlign = 'left';
}

