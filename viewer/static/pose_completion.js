/* Estimated anatomy is separate from detector keypoints and contact inputs. */
function drawPoseEstimates(ctx, ann, frameIndex, px, py, scale) {
  const toggle = document.getElementById('togPoseEstimates');
  if (!toggle || !toggle.checked || toggle.disabled || Number(ann.f) !== frameIndex) return;
  const bones = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],
    [11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4],[0,5],[0,6]];
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 2;
  for (const estimate of (ann.pose_estimates || [])) {
    if (estimate.source !== 'temporal_pose_completion_v1') continue;
    const active = !selectedPlayerId || estimate.player_id === selectedPlayerId;
    ctx.strokeStyle = active ? '#00b9e8' : 'rgba(150,170,190,0.50)';
    ctx.fillStyle = ctx.strokeStyle;
    const joints = new Set(estimate.estimated_joints);
    const points = estimate.keypoints;
    for (const [a, b] of bones) {
      if ((!joints.has(a) && !joints.has(b)) || !points[a] || !points[b]) continue;
      ctx.beginPath();
      ctx.moveTo(px(points[a][0]), py(points[a][1]));
      ctx.lineTo(px(points[b][0]), py(points[b][1]));
      ctx.stroke();
    }
    for (const joint of joints) {
      const point = points[joint];
      if (point) ctx.strokeRect(px(point[0])-2.5, py(point[1])-2.5, 5, 5);
    }
    if (estimate.observed_pose_index === null) {
      const anchor = points.find(p => p);
      if (anchor) {
        ctx.font = `bold ${Math.max(10, Math.round(12 * scale))}px system-ui`;
        ctx.fillText(`${estimate.player_id} · estimated`, px(anchor[0])+5, py(anchor[1])-8);
      }
    }
  }
  ctx.restore();
}

function initPoseEstimates(data) {
  const toggle = document.getElementById('togPoseEstimates');
  const status = document.getElementById('poseEstimateStatus');
  const report = data.pose_completion;
  const available = report && report.schema === 'temporal_pose_completion_v1';
  toggle.disabled = !available;
  if (!available) {
    status.textContent = 'No temporal pose estimates in this result.';
    return;
  }
  const counts = report.counts;
  status.textContent = `Cyan dashed: estimated hidden joints. ${counts.missing_person_frames || 0} ` +
    `missing + ${counts.partial_person_frames || 0} partial player-frames. ` +
    'Uses past and future observations; contact counts still use observed poses.';
}
