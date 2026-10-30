/* Read-only experimental review. Source events, poses and identities stay intact. */
const LAB_LABELS = {
  spotter: 'Existing spotter + local author',
  spotter_sequence: 'Existing spotter + sequence author',
  spotter_joint: 'Spotter + joint timing/author',
  ballistic: 'Piecewise flight change',
  interaction: 'Moving hands / forearms',
  joint: 'Flight + limb interaction',
  consensus: 'Spotter + extra joint candidates',
};
let labBundle, labAnnotations, labFrames, labMethod, labPlayer = null, labActive = null, labLastFrame = -1;
const labNode = id => document.getElementById(id);
const labTime = frame => `${Math.floor(frame/labBundle.fps/60)}:${String(Math.floor(frame/labBundle.fps)%60).padStart(2,'0')}`;

function labOriginal(event) {
  return labBundle.methods.spotter.events.find(e => e.source_event_index === event.source_event_index);
}
function labChanged(event) {
  const old = event.source_event_index == null ? null : labOriginal(event);
  return Boolean(old && (old.player_id !== event.player_id || old.side_proposal !== event.side_proposal));
}
function labSeek(event) {
  labActive = event;
  const f = Number.isFinite(event.contact_frame) ? event.contact_frame : event.frame;
  labNode('labVideo').pause();
  // Seek inside the frame, avoiding fractional-fps boundary rounding.
  labNode('labVideo').currentTime = (f+.25)/labBundle.fps;
  labLastFrame = -1;
  const identity = event.player_id || event.proposed_player_id || (event.side_proposal === 'a' ? 'Far player' : 'Unknown near player');
  labNode('labDetails').textContent = `${identity} · ${event.type} · detector f${event.frame}, evidence f${f}. ` +
    `${event.reason.replace(/_/g,' ')}; near-side score ${event.near_side_score.toFixed(2)} (uncalibrated). ` +
    `Flight ${event.detection_evidence.ballistic.toFixed(2)}, interaction ${event.detection_evidence.interaction.toFixed(2)}. ` +
    (event.source_event_index == null ? 'New untyped proposal; no source event.' : `Source event ${event.source_event_index}, original f${event.original_frame}.`);
  labRenderEvents();
}
function labRenderEvents() {
  if (!labBundle) return;
  const method = labBundle.methods[labMethod];
  const events = method.events;
  const assigned = events.filter(e => e.player_id && (!labPlayer || e.player_id === labPlayer));
  const suggestions = labPlayer ? events.filter(e => !e.player_id && e.proposed_player_id === labPlayer) : [];
  const changes = events.filter(labChanged).length;
  labNode('labSummary').textContent = `${events.length} proposals · ${assigned.length} assigned${labPlayer ? ` to ${labPlayer}` : ''} · ${suggestions.length} uncertain suggestions · ${changes} author/side changes`;
  const list = labNode('labEvents'); list.replaceChildren();
  for (const event of events) {
    if (labPlayer && event.player_id !== labPlayer && event.proposed_player_id !== labPlayer) continue;
    if (labNode('labOnlyNew').checked && event.source_event_index != null) continue;
    if (labNode('labOnlyChanges').checked && !labChanged(event)) continue;
    const button = document.createElement('button');
    button.className = 'event' + (!event.player_id ? ' uncertain' : '') + (event === labActive ? ' selected' : '') + (event.source_event_index == null ? ' new' : '');
    const actor = event.player_id || (event.status === 'other_side' ? 'Other side' : 'Unresolved');
    button.textContent = `${labTime(event.frame)} · ${event.type} · ${actor}`;
    const provenance = document.createElement('small');
    const old = event.source_event_index == null ? null : labOriginal(event);
    provenance.textContent = (old && labChanged(event) ? `Changed: ${old.player_id || old.status} → ${actor}; side ${old.side_proposal} → ${event.side_proposal}. ` : '') +
      (event.source_event_index == null ? 'NEW · untyped · ' : '') + event.reason.replace(/_/g,' ') +
      (event.proposed_player_id && !event.player_id ? ` · possible ${event.proposed_player_id}; excluded from total` : '');
    button.appendChild(provenance); button.onclick = () => labSeek(event); list.appendChild(button);
  }
  labLastFrame = -1;
}
function labRenderMethod() {
  labActive = null;
  labNode('labDetails').textContent = 'Choose a contact to inspect its evidence.';
  labNode('labMethod').value = labMethod;
  const method = labBundle.methods[labMethod];
  const timing = method.detection_lovo.all.at_5;
  labNode('labMethodInfo').textContent = `Held-out contact detection: F1 ${(timing.f1*100).toFixed(1)}%, precision ${(timing.precision*100).toFixed(1)}%, recall ${(timing.recall*100).toFixed(1)}%. ${method.author_method}.`;
  labRenderEvents();
}
function labDraw() {
  requestAnimationFrame(labDraw);
  if (!labBundle || !labAnnotations) return;
  const video = labNode('labVideo');
  const f = Math.floor(video.currentTime*labBundle.fps+.00001);
  const canvas = labNode('labCanvas');
  const width = Math.round(video.clientWidth), height = Math.round(video.clientHeight);
  if (!width || !height || (labLastFrame === f && canvas.width === width && canvas.height === height)) return;
  labLastFrame = f; canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.scale(width/labAnnotations.frame_width, height/labAnnotations.frame_height);
  const unit = labAnnotations.frame_width/width;
  const row = labFrames.get(f);
  labNode('labFrame').textContent = `f${f} · ${labTime(f)}`;
  if (!row) return;
  const candidate = labActive && (labActive.candidates || [])[0];
  const bones = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  (row.poses || []).forEach((pose,index) => {
    const focused = candidate && candidate.frame === f && candidate.pose_index === index && candidate.raw_track_id === pose.id;
    const near = pose.team === 'b' || pose.team === 'team_b';
    const selected = focused || (near && (!labPlayer || pose.player_id === labPlayer));
    ctx.strokeStyle = ctx.fillStyle = selected ? '#ffdc00' : 'rgba(170,185,200,.65)';
    ctx.lineWidth = (selected ? 2.5 : 1.3)*unit;
    const point = i => (pose.kp || [])[i];
    const valid = p => p && p[2] >= .25 && Number.isFinite(p[0]) && Number.isFinite(p[1]);
    for (const [a,b] of bones) if (valid(point(a)) && valid(point(b))) {
      ctx.beginPath(); ctx.moveTo(...point(a).slice(0,2)); ctx.lineTo(...point(b).slice(0,2)); ctx.stroke();
    }
    for (const p of pose.kp || []) if (valid(p)) {ctx.beginPath();ctx.arc(p[0],p[1],2.5*unit,0,Math.PI*2);ctx.fill();}
    if (pose.player_id && pose.bb) {ctx.font = `bold ${12*unit}px system-ui`;ctx.fillText(pose.player_id,pose.bb[0],pose.bb[1]-5*unit);}
  });
  if (row.ball && !row.ball_out_of_frame) {
    const derived = row.ball_interp || row.ball_reconstructed || /reconstruct|interpolat/.test(row.ball_source || '');
    ctx.strokeStyle = derived ? '#f6ac35' : '#00e7a0';ctx.lineWidth = 2*unit;
    ctx.beginPath();ctx.arc(row.ball[0],row.ball[1],7*unit,0,Math.PI*2);ctx.stroke();
  }
  if (candidate && candidate.frame === f && candidate.bbox) {
    const b = candidate.bbox;ctx.strokeStyle = '#f6ac35';ctx.lineWidth = 2*unit;ctx.setLineDash([5*unit,3*unit]);
    ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.setLineDash([]);
    ctx.fillStyle = '#f6ac35';ctx.font = `bold ${12*unit}px system-ui`;
    ctx.fillText(`${labActive.player_id || labActive.proposed_player_id || 'Actor'} · proposal`,b[0],b[1]-20*unit);
  }
}
async function labInit() {
  try {
    const responses = await Promise.all([fetch('/api/contact_lab/'+CONTACT_LAB_JOB),fetch('/api/annotations/'+CONTACT_LAB_JOB)]);
    for (const response of responses) if (!response.ok) throw new Error((await response.json()).error || 'Experiment unavailable');
    [labBundle,labAnnotations] = await Promise.all(responses.map(r=>r.json()));
    labFrames = new Map(labAnnotations.frames.map(row=>[row.f,row]));
    labMethod = labBundle.selected_method;
    for (const name of Object.keys(labBundle.methods)) {
      const option = document.createElement('option');option.value = name;option.textContent = LAB_LABELS[name] || name;labNode('labMethod').appendChild(option);
    }
    labNode('labMethod').disabled = false;
    labNode('labMethod').onchange = e => {labMethod = e.target.value;labRenderMethod();};
    for (const id of [null,...labBundle.roster_ids]) {
      const button = document.createElement('button');button.textContent = id || 'All';button.className = id === labPlayer ? 'active' : '';
      button.onclick = () => {labPlayer = id;labActive = null;labNode('labDetails').textContent = 'Choose a contact to inspect its evidence.';for (const child of labNode('labPlayers').children) child.className = child === button ? 'active' : '';labRenderEvents();};
      labNode('labPlayers').appendChild(button);
    }
    for (const id of ['labOnlyNew','labOnlyChanges']) labNode(id).onchange = labRenderEvents;
    for (const [id,delta] of [['labBack',-1],['labForward',1]]) labNode(id).onclick = () => {
      const video = labNode('labVideo');video.pause();video.currentTime = Math.max(0,video.currentTime+delta/labBundle.fps);
    };
    const team = labBundle.comparison.team;
    const detected = labBundle.comparison.detected_team.pooled;
    labNode('labQuality').textContent = `On ${detected.local.all.matched_known_team} matched spotter contacts: correct team ${(detected.local.all.team_accuracy_on_matched*100).toFixed(1)}% → ${(detected.sequence.all.team_accuracy_on_matched*100).toFixed(1)}%; joint detection + team F1 ${(detected.local.all.joint_detection_team.f1*100).toFixed(1)}% → ${(detected.sequence.all.joint_detection_team.f1*100).toFixed(1)}%. Known human timing/type control: ${(team.flat_extra_trees.accuracy*100).toFixed(1)}% → ${(team.sequence.accuracy*100).toFixed(1)}%. P1–P7 accuracy is unknown. Three-video model-selection estimates; JIZ3 is unlabelled. Warm-up is not gated.`;
    const table = document.createElement('table');
    const heading = document.createElement('tr');
    for (const name of ['Method','F1 ±5f','F1 excluding human non-game']) {const th = document.createElement('th');th.textContent = name;heading.appendChild(th);}table.appendChild(heading);
    for (const [name,metrics] of Object.entries(labBundle.comparison.detection)) {
      const tr = document.createElement('tr');
      for (const value of [LAB_LABELS[name],(metrics.all.at_5.f1*100).toFixed(1)+'%',(metrics.excluding_non_game.at_5.f1*100).toFixed(1)+'%']) {const td = document.createElement('td');td.textContent = value;tr.appendChild(td);}table.appendChild(tr);
    }
    labNode('labComparison').appendChild(table);labRenderMethod();
  } catch (error) {labNode('labError').textContent = error.message;}
}
document.addEventListener('DOMContentLoaded',labInit);
requestAnimationFrame(labDraw);
