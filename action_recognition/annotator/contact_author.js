/* Human author labels. Predictions never prefill a decision. */
"use strict";
function frameTime(frame, fps) { return (frame + 0.25) / fps; }
function rectangle(start, end, width, height) {
  return [Math.max(0, Math.min(start[0], end[0])), Math.max(0, Math.min(start[1], end[1])),
    Math.min(width, Math.max(start[0], end[0])), Math.min(height, Math.max(start[1], end[1]))];
}
function sourcePoint(event, bounds, width, height) {
  return [(event.clientX - bounds.left) * width / bounds.width, (event.clientY - bounds.top) * height / bounds.height];
}
if (typeof module !== "undefined") module.exports = {frameTime, rectangle, sourcePoint};
if (typeof document !== "undefined") boot();

async function boot() {
  const $ = id => document.getElementById(id);
  const video = $("video"), canvas = $("cropCanvas"), context = canvas.getContext("2d");
  const names = {serve:"Подача", reception:"Прийом", pass:"Пас", set:"Передача", dig:"Захист", attack:"Атака", block:"Блок", free_ball:"Free ball", other:"Інший контакт"};
  let project, state, progress, videoId, rows = [], index = 0, busy = false, crop = null, drag = null, box = null;
  let stopAt = null, pendingFrame = null, loadVersion = 0, boxFrame = null;
  let referenceFocus = [];
  let pendingRemoval = null, cardsSignature = null;
  let pendingDuplicate = null;
  let mode = "contacts", segments = [], movementIndex = 0, boxes = [], boxVersion = 0;
  let movementLoading = false, pendingMove = false;
  const current = () => rows[index];
  const reviewedType = contact => state.event_reviews?.[contact.id]?.type || contact.type;
  const metadata = () => project.videos[videoId];
  const roster = () => state.rosters[videoId];
  const segment = () => segments[movementIndex];
  const spans = (key = segment()?.id) => state.movement_reviews?.[videoId]?.[key] || [];
  const startOf = item => (spans(item.id).at(-1)?.end ?? item.start - 1) + 1;
  const contactFrame = contact => state.timing_reviews?.[contact.id]?.frame ?? contact.frame;
  const selectedPlayer = () => mode === "movement" ? spans().at(-1)?.player_id : state.labels[current().id]?.player_id;
  const labelText = label => !label ? "Без мітки" : label.decision === "player" ? label.player_id :
    ({other_team:"Інша команда", unsure:"Не впевнений", not_contact:"Це не контакт"}[label.decision]);
  const movementLabels = {other:"Поза складом", unsure:"Не впевнений", false_positive:"Хибна детекція"};
  const error = message => { $("error").textContent = message || ""; };

  async function json(url, options) {
    const controller = new AbortController(), deadline = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, {...options, signal:controller.signal});
      const result = await response.json();
      if (!response.ok) {
        const failure = new Error(result.error || "Помилка сервера");
        failure.status = response.status; failure.overlap = result.overlap;
        throw failure;
      }
      return result;
    } catch (failure) {
      if (failure.name === "AbortError") { const timeout = new Error("Сервер не відповів за 15 секунд."); timeout.timeout = true; throw timeout; }
      throw failure;
    } finally { clearTimeout(deadline); }
  }
  function lock(value) {
    busy = value;
    for (const id of ["previous","next","unlabelled","videoSelect","contactSelect","confirmRoster","addPlayer","saveCrop"]) $(id).disabled = value;
    $("saveCrop").disabled = value || !box;
    updateButtons();
  }
  function updateButtons() {
    const disabled = busy || !roster().confirmed || !!crop || !!pendingRemoval;
    const inMotion = mode === "movement";
    const unavailable = inMotion && (!segment() || movementLoading || startOf(segment()) > segment().end);
    for (const id of ["other","uncertain","notContact","reset"]) $(id).disabled = disabled;
    $("other").disabled = $("uncertain").disabled = disabled || unavailable;
    $("falsePositive").disabled = disabled || unavailable;
    $("confirmDuplicate").disabled = disabled || unavailable;
    $("cancelDuplicate").disabled = busy;
    $("reset").disabled = disabled || (inMotion && !spans().length);
    document.querySelectorAll(".pick").forEach(button => { button.disabled = disabled || unavailable; });
    document.querySelectorAll(".editReference").forEach(button => { button.disabled = busy || !!pendingRemoval; });
    document.querySelectorAll(".removePlayer").forEach(button => { button.disabled = busy || !!crop || !!pendingRemoval; });
    $("confirmRemove").disabled = busy || !pendingRemoval;
    $("cancelRemove").disabled = busy;
    $("restorePlayer").disabled = busy || !!crop || !!pendingRemoval;
    $("rosterFrame").disabled = busy || !roster().players.length;
    $("confirmRoster").disabled = busy || !roster().players.length;
    $("eventType").disabled = busy || !!crop || !!pendingRemoval;
    $("resetEventType").disabled = busy || !!crop || !!pendingRemoval || !state.event_reviews?.[current().id];
    $("movementMode").disabled = $("contactMode").disabled = busy || !!crop || !!pendingRemoval || pendingMove;
    $("saveTiming").disabled = $("addContact").disabled = busy || !!crop || !!pendingRemoval;
  }
  async function mutate(url, payload) {
    if (busy) return false;
    clearDuplicate();
    lock(true); error(""); $("saved").textContent = "Зберігаю…";
    try {
      const result = await json(url, {method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({...payload, revision:state.revision})});
      state = result.state; progress = result.progress;
      $("saved").textContent = `Збережено · ревізія ${state.revision}`;
      return true;
    } catch (failure) {
      if (url === "/api/movement" && failure.overlap) {
        pendingDuplicate = {overlap:failure.overlap, playerId:payload.player_id, end:payload.end};
        $("duplicateMessage").textContent = `${failure.message} Жовта рамка — попередня мітка, бірюзова — поточна. Якщо це та сама людина на f${startOf(segment())}–${payload.end}, натисніть Enter або виберіть ${payload.player_id} ще раз.`;
        $("confirmDuplicate").textContent = `Так, це ${payload.player_id} · Enter`;
        $("duplicatePanel").classList.remove("hidden");
        $("saved").textContent = "Потрібне підтвердження";
        seek(failure.overlap.frame);
        return false;
      }
      $("saved").textContent = "Мітку не збережено";
      if (failure.status === 409 || !failure.status) {
        try { const latest = await json("/api/project"); state = latest.state; progress = latest.progress;
          if (failure.timeout) $("saved").textContent = `Стан оновлено · ревізія ${state.revision}. Перевірте поточну мітку.`;
        }
        catch (_) { /* Source conflicts stay visible; never retry a write. */ }
      }
      error(failure.message + (failure.status === 409 || failure.timeout ? " Перевірте поточну мітку перед повтором; якщо стан не оновився, перезавантажте сторінку." : ""));
      return false;
    } finally { refreshRows(current().id); lock(false); render(); }
  }
  function clearDuplicate() {
    if (!pendingDuplicate) return;
    pendingDuplicate = null;
    $("duplicatePanel").classList.add("hidden");
    $("saved").textContent = "";
    paint();
  }
  function seek(frame) {
    frame = Math.max(0, Math.min(metadata().num_frames - 1, Math.round(frame)));
    video.pause(); stopAt = null; pendingFrame = frame; referenceFocus = []; paint();
    if (video.readyState >= 1 && !video.error) {
      video.currentTime = frameTime(frame, metadata().fps); pendingFrame = null;
    }
    if (crop) { box = null; paint(); $("saveCrop").disabled = true; }
  }
  function observedFrame() { return Math.max(0, Math.min(metadata().num_frames - 1, Math.floor(video.currentTime * metadata().fps + 0.001))); }
  function seekReference(player, view = 0, all = false) {
    const reference = player.references[view];
    seek(reference.frame);
    referenceFocus = (all ? roster().players : [player]).flatMap(p =>
      p.references.filter(r => r.frame === reference.frame).map(r => ({...r, player_id:p.player_id})));
    paint();
  }
  function showContact(nextIndex) {
    if (busy) return;
    cancelRemoval();
    index = Math.max(0, Math.min(rows.length - 1, nextIndex));
    cancelCrop(); seek(contactFrame(current())); render();
  }
  function refreshRows(keepId) {
    rows = [...project.contacts, ...Object.values(state.added_contacts || {})]
      .filter(c => c.video_id === videoId).sort((a,b) => contactFrame(a) - contactFrame(b));
    index = Math.max(0, rows.findIndex(c => c.id === keepId));
  }
  function selectVideo(id) {
    if (busy) return;
    clearDuplicate();
    cancelRemoval();
    cancelCrop(); videoId = id; mode = "contacts"; boxVersion++; pendingMove = false;
    refreshRows();
    index = rows.findIndex(c => !state.labels[c.id]); if (index < 0) index = 0;
    canvas.width = metadata().width; canvas.height = metadata().height;
    const version = ++loadVersion;
    video.src = "/video/" + encodeURIComponent(id);
    video.onloadedmetadata = () => { if (version === loadVersion) { video.currentTime = frameTime(pendingFrame ?? contactFrame(current()), metadata().fps); pendingFrame = null; } };
    $("seek").max = metadata().num_frames - 1;
    seek(contactFrame(current())); render();
  }
  function render() {
    if (!project || !videoId) return;
    $("videoSelect").value = videoId;
    const inMotion = mode === "movement";
    for (const id of ["contactActions","notContact"]) $(id).classList.toggle("hidden", inMotion);
    for (const id of ["movementActions","priorityLabel","falsePositive"]) $(id).classList.toggle("hidden", !inMotion);
    $("movementMode").classList.toggle("hidden", !metadata().movement_available);
    $("movementMode").classList.toggle("primary", inMotion);
    $("contactMode").classList.toggle("primary", !inMotion);
    $("movementMode").setAttribute?.("aria-pressed", inMotion);
    $("contactMode").setAttribute?.("aria-pressed", !inMotion);
    $("other").textContent = inMotion ? "Поза складом · D" : "Інша команда · O";
    $("reset").textContent = inMotion ? "Скасувати останню мітку відрізка" : "Скинути мітку";
    $("play").textContent = inMotion ? "▶ Відрізок · A" : "▶ ±2 с";
    $("backFrame").title = inMotion ? "Кадр назад" : "Кадр назад · A";
    $("nextFrame").title = inMotion ? "Кадр вперед" : "Кадр вперед · D";
    $("toContact").textContent = inMotion ? "До початку відрізка" : "До контакту";
    if (inMotion) renderMovement(); else renderContacts();
    $("rosterStatus").textContent = roster().confirmed ? "Картки підтверджені" : "Спочатку перевірте й підтвердьте картки";
    $("confirmRoster").classList.toggle("hidden", roster().confirmed);
    $("confirmRoster").textContent = `Підтверджую: це ${roster().players.length} різних гравців`;
    const removed = roster().removed_players || [], lastRemoved = removed[removed.length - 1];
    $("restorePlayer").classList.toggle("hidden", !lastRemoved);
    $("restorePlayer").textContent = lastRemoved ? `Відновити ${lastRemoved.player.player_id}` : "Відновити видалену картку";
    renderCards(); updateButtons();
  }
  function renderContacts() {
    $("contactSelect").replaceChildren(...rows.map((c, i) => {
      const option = document.createElement("option"); option.value = i;
      option.textContent = `${state.labels[c.id] ? "✓" : "○"} ${i + 1}. f${contactFrame(c)} · ${names[reviewedType(c)] || reviewedType(c)} · ${labelText(state.labels[c.id])}`;
      return option;
    }));
    $("contactSelect").value = index;
    const contact = current(), type = reviewedType(contact), review = state.event_reviews?.[contact.id];
    $("event").textContent = `${index + 1}/${rows.length} · ${names[type] || type} · f${contactFrame(contact)}`;
    $("eventMeta").textContent = `Мітка: ${labelText(state.labels[contact.id])} · Час: ${contact.timing_verified || state.timing_reviews?.[contact.id] ? "перевірений людиною" : "початкова позначка, час не перевірений"}${!review && contact.event_label_source === "model" ? " · Подія знайдена моделлю; тип ще не перевірений" : ""}`;
    $("eventType").replaceChildren(...Object.entries(names).map(([value, text]) => {
      const option = document.createElement("option"); option.value = value; option.textContent = text; return option;
    }));
    if (!names[type]) { const option = document.createElement("option"); option.value = type; option.textContent = type; $("eventType").append(option); }
    $("eventType").value = type;
    $("eventTypeStatus").textContent = review ? `Тип підтверджений людиною · початковий: ${names[contact.type] || contact.type}` : "Виберіть інший тип — зміна збережеться автоматично. Автор контакту залишиться тим самим.";
    const total = Object.values(progress).reduce((sum, p) => sum + p.total, 0);
    const reviewed = Object.values(progress).reduce((sum, p) => sum + p.reviewed, 0);
    $("progress").max = total; $("progress").value = reviewed;
    const p = progress[videoId];
    $("summary").textContent = `Відео: ${p.reviewed}/${p.total} переглянуто · лишилось ${p.unlabelled} · Весь проєкт: ${reviewed}/${total} · Не впевнений: ${p.decisions.unsure || 0}`;
    $("labelGuard").textContent = roster().confirmed ? "" : roster().players.length ? "Щоб присвоювати контакти, перевірте картки справа й натисніть «Підтверджую». Зайві картки можна видалити." : "Додайте картки гравців кнопкою «Новий гравець / заміна», потім підтвердьте їх.";
  }
  function renderMovement() {
    const item = segment();
    $("contactSelect").replaceChildren(...segments.map((s, i) => {
      const option = document.createElement("option"); option.value = i;
      const status = startOf(s) > s.end ? (spans(s.id).some(span => !span.verified) ? "?" : "✓") : "○";
      option.textContent = `${status} ${s.id} · f${s.start}–${s.end}`;
      return option;
    }));
    $("contactSelect").value = movementIndex;
    $("event").textContent = item ? `Рух · f${item.start}–${item.end}` : "Немає збережених поз гравців";
    $("eventMeta").textContent = item && movementLoading ? "Завантажую рамку…" : spans().map(span => {
      return `f${span.start}–${span.end} · ${span.player_id || movementLabels[span.decision]}`;
    }).join("  |  ");
    const total = segments.reduce((sum, s) => sum + s.end - s.start + 1, 0);
    const reviewed = segments.reduce((sum, s) => sum + startOf(s) - s.start, 0);
    const unsure = Object.values(state.movement_reviews?.[videoId] || {}).flat()
      .reduce((sum, span) => sum + (span.verified ? 0 : span.end - span.start + 1), 0);
    $("progress").max = total || 1; $("progress").value = reviewed;
    $("summary").textContent = `${reviewed}/${total} збережених спостережень переглянуто · ${unsure} лишилися сумнівними`;
    $("labelGuard").textContent = !roster().confirmed ? "Спочатку підтвердьте картки гравців." : item && startOf(item) <= item.end ?
      `Ще без мітки: f${startOf(item)}–${item.end}` : "";
  }
  async function showMovement(nextIndex) {
    if (busy || !segments.length) return;
    clearDuplicate();
    cancelRemoval(); cancelCrop();
    movementIndex = Math.max(0, Math.min(segments.length - 1, nextIndex));
    const item = segment(), version = ++boxVersion;
    movementLoading = true; boxes = [];
    $("partialMovement").checked = false;
    seek(Math.min(startOf(item), item.end)); render(); error("");
    try {
      const data = await json(`/api/movement/${encodeURIComponent(videoId)}/${encodeURIComponent(item.id)}`);
      if (version !== boxVersion) return;
      boxes = data.boxes; movementLoading = false; render(); paint();
    } catch (failure) {
      if (version === boxVersion) error(failure.message);
    }
  }
  function orderSegments() {
    segments.sort((a,b) => ($("priority").checked ? b.priority[0] - a.priority[0] || b.priority[1] - a.priority[1] : 0)
      || a.start - b.start);
  }
  async function changeMode(nextMode) {
    if (busy || crop || pendingRemoval || pendingMove || mode === nextMode) return;
    clearDuplicate();
    video.pause(); stopAt = null; boxVersion++; boxes = []; error("");
    if (nextMode === "contacts") { mode = nextMode; render(); paint(); return; }
    const id = videoId;
    pendingMove = true; updateButtons();
    $("saved").textContent = "Готую відрізки з кешованих треків…";
    try {
      const data = await json(`/api/movement/${encodeURIComponent(id)}`);
      if (id !== videoId) return;
      $("saved").textContent = "";
      segments = data.segments; mode = nextMode; orderSegments();
      if (segments.length) await showMovement(segments.findIndex(s => startOf(s) <= s.end));
      else render();
    } catch (failure) { if (id === videoId) { $("saved").textContent = ""; error(failure.message); } }
    finally { if (id === videoId) { pendingMove = false; updateButtons(); } }
  }
  function renderCards() {
    const signature = JSON.stringify([videoId, roster().players]);
    if (cardsSignature === signature) {
      document.querySelectorAll(".card").forEach((card, i) => card.classList.toggle("selected", selectedPlayer() === roster().players[i]?.player_id));
      return;
    }
    cardsSignature = signature;
    $("cards").replaceChildren();
    for (const player of roster().players) {
      const card = document.createElement("div"); card.className = "card";
      if (selectedPlayer() === player.player_id) card.classList.add("selected");
      const pick = document.createElement("button"); pick.className = "pick";
      const image = document.createElement("img"); image.alt = `Фото ${player.player_id}`;
      const title = document.createElement("strong"); title.textContent = player.player_id;
      let view = 0;
      const setView = number => { view = number; image.src = `/portrait/${encodeURIComponent(videoId)}/${player.player_id}/${number}?v=${encodeURIComponent(JSON.stringify(player.references[number]))}`; };
      setView(0); pick.append(image, title); pick.onclick = () => assign("player", player.player_id); card.append(pick);
      const views = document.createElement("div"); views.className = "views";
      player.references.forEach((_, n) => {
        if (player.references.length < 2) return;
        const button = document.createElement("button"); button.textContent = n + 1; button.onclick = () => setView(n); views.append(button);
      }); card.append(views);
      const tools = document.createElement("div"); tools.className = "cardTools";
      const jump = document.createElement("button"); jump.textContent = "Кадр фото"; jump.onclick = () => seekReference(player, view);
      const edit = document.createElement("button"); edit.className = "editReference";
      edit.textContent = roster().confirmed ? "+ Ракурс" : "Замінити фото";
      edit.onclick = () => startCrop(roster().confirmed ? "add_view" : "replace_reference", player.player_id);
      const remove = document.createElement("button"); remove.className = "removePlayer"; remove.textContent = `Видалити ${player.player_id}`;
      remove.style = "width:100%;margin-top:6px;color:#ffb6ac"; remove.onclick = () => requestRemoval(player.player_id);
      tools.append(jump, edit); card.append(tools, remove); $("cards").append(card);
    }
  }
  function cancelRemoval() {
    pendingRemoval = null; $("removePanel").classList.add("hidden");
    if (state && videoId) updateButtons();
  }
  function requestRemoval(playerId) {
    if (busy || crop) return;
    const affected = rows.filter(c => state.labels[c.id]?.player_id === playerId).length;
    pendingRemoval = {video_id:videoId, player_id:playerId, affected};
    $("removeTitle").textContent = `Видалити картку ${playerId}?`;
    $("removeMessage").textContent = affected ? `Цьому гравцю присвоєно ${affected} контактів. Вони повернуться до нерозмічених. Картку й попередні мітки можна відновити.` : "Картка зникне з меню. Інші ID не зміняться. Видалення можна скасувати кнопкою відновлення.";
    $("confirmRemove").textContent = affected ? `Видалити й зняти ${affected} міток` : "Видалити картку";
    $("removePanel").classList.remove("hidden"); $("removePanel").scrollIntoView?.({block:"nearest"}); updateButtons();
  }
  $("cancelRemove").onclick = cancelRemoval;
  $("confirmRemove").onclick = async () => {
    if (busy || !pendingRemoval) return;
    const removal = {...pendingRemoval};
    await mutate("/api/roster", {action:"remove_player", video_id:removal.video_id, player_id:removal.player_id,
      removal_confirmed:true, unassign_contacts_confirmed:removal.affected > 0});
    cancelRemoval(); render();
  };
  $("restorePlayer").onclick = () => {
    const removed = roster().removed_players || [], last = removed[removed.length - 1];
    if (!busy && !crop && !pendingRemoval && last) return mutate("/api/roster", {video_id:videoId, action:"restore_player", player_id:last.player.player_id});
  };
  async function assign(decision, playerId = null) {
    if (busy || crop || pendingRemoval || !roster().confirmed) return;
    if (mode === "movement") return assignMovement(decision, playerId);
    const contactId = current().id, oldIndex = index, oldVideo = videoId;
    if (await mutate("/api/label", {contact_id:contactId, decision, player_id:playerId})) {
      if (decision !== "reset" && $("autoNext").checked && videoId === oldVideo && current().id === contactId) showContact(oldIndex + 1);
    }
  }
  async function assignMovement(decision, playerId) {
    const item = segment();
    if (!item || movementLoading || decision === "not_contact") return;
    const start = startOf(item), undo = decision === "reset";
    if (!undo && start > item.end) return;
    if (undo && !spans().length) return;
    video.pause(); stopAt = null;
    const confirmation = decision === "player" && pendingDuplicate?.playerId === playerId ? pendingDuplicate : null;
    const partial = $("partialMovement").checked && !confirmation;
    if (!undo && partial && (video.seeking || video.readyState < 2)) return;
    const end = confirmation?.end ?? (partial ? observedFrame() : item.end);
    const saved = await mutate("/api/movement", {video_id:videoId, segment_id:item.id,
      decision:undo ? "undo" : decision === "other_team" ? "other" : decision, player_id:playerId, end,
      ...(confirmation ? {confirm_overlap:true} : {})});
    if (!saved) return;
    if (!undo && $("autoNext").checked && startOf(item) > item.end) {
      const next = segments.findIndex((s,i) => i > movementIndex && startOf(s) <= s.end);
      const first = segments.findIndex(s => startOf(s) <= s.end);
      if (first >= 0) return showMovement(next >= 0 ? next : first);
    }
    $("partialMovement").checked = false;
    seek(Math.min(startOf(item), item.end));
  }
  async function changeEvent(action, type) {
    if (busy || crop || pendingRemoval) return false;
    return mutate("/api/event", {contact_id:current().id, action, ...(action === "set" ? {type} : {})});
  }
  $("eventType").onchange = event => changeEvent("set", event.target.value);
  $("resetEventType").onclick = () => changeEvent("reset");
  async function saveContactFrame(action) {
    if (busy || crop || pendingRemoval || video.seeking || video.readyState < 2) return;
    video.pause(); stopAt = null;
    const frame = observedFrame(), kind = $("newContactType").value;
    if (action === "add" && !kind) { error("Спочатку оберіть тип пропущеного контакту."); return; }
    const id = action === "timing" ? current().id : null;
    if (await mutate("/api/contact-frame", {video_id:videoId, frame, action, contact_id:id, type:kind})) {
      if (action === "add") {
        $("newContactType").value = "";
        showContact(rows.findIndex(c => c.id === `${videoId}:human:r${state.revision}`));
      }
    }
  }
  $("saveTiming").onclick = () => saveContactFrame("timing");
  $("addContact").onclick = () => saveContactFrame("add");
  for (const [kind, label] of Object.entries(names)) {
    const option = document.createElement("option"); option.value = kind; option.textContent = label;
    $("newContactType").append(option);
  }
  function startCrop(action, playerId = null) {
    if (busy || pendingRemoval) return;
    clearDuplicate();
    video.pause(); stopAt = null; referenceFocus = []; crop = {action, player_id:playerId}; box = null; drag = null;
    canvas.classList.add("drawing"); $("cropPanel").classList.remove("hidden");
    $("cropTitle").textContent = action === "add_player" ? "Новий гравець: обведіть людину, якої ще немає на картках" : `${playerId}: обведіть саме цю людину`;
    $("saveCrop").textContent = action === "add_player" ? "Підтверджую нового гравця" : action === "add_view" ? `Це той самий ${playerId} — додати ракурс` : `Замінити початкове фото ${playerId}`;
    $("saveCrop").disabled = true; updateButtons(); paint();
  }
  function cancelCrop() {
    crop = null; box = null; drag = null; boxFrame = null;
    canvas.classList.remove("drawing"); $("cropPanel").classList.add("hidden"); paint();
    if (state && videoId) updateButtons();
  }
  function paint(frame = videoId ? observedFrame() : null) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!crop && referenceFocus.length && project && videoId && !video.seeking && video.readyState >= 2) {
      context.lineWidth = 5; context.strokeStyle = "#72dbc0"; context.fillStyle = "#72dbc0";
      context.font = "bold 44px sans-serif";
      for (const reference of referenceFocus) {
        if (reference.frame !== frame) continue;
        const [x1,y1,x2,y2] = reference.bbox;
        context.strokeRect(x1,y1,x2-x1,y2-y1); context.fillText(reference.player_id,x1,Math.max(46,y1-10));
      }
    }
    if (!crop && mode === "movement" && !video.seeking && video.readyState >= 2) {
      const overlap = pendingDuplicate?.overlap;
      if (overlap?.frame === frame) {
        const [x1,y1,x2,y2] = overlap.bbox;
        context.lineWidth = 8; context.strokeStyle = "#ebc66f";
        context.strokeRect(x1,y1,x2-x1,y2-y1);
      }
      const bbox = boxes[frame - (segment()?.start ?? frame)];
      if (bbox) {
        const [x1,y1,x2,y2] = bbox;
        context.lineWidth = 8; context.strokeStyle = "#72dbc0";
        context.strokeRect(x1,y1,x2-x1,y2-y1);
      }
    }
    if (!box) return;
    context.lineWidth = 6; context.strokeStyle = "#72dbc0";
    context.fillStyle = "#72dbc022"; context.fillRect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
    context.strokeRect(box[0], box[1], box[2]-box[0], box[3]-box[1]);
  }
  canvas.onpointerdown = event => {
    if (!crop || busy || video.seeking || video.readyState < 2) return;
    video.pause(); boxFrame = observedFrame(); drag = sourcePoint(event, canvas.getBoundingClientRect(), canvas.width, canvas.height);
    canvas.setPointerCapture(event.pointerId); box = null;
  };
  canvas.onpointermove = event => {
    if (!drag) return;
    box = rectangle(drag, sourcePoint(event, canvas.getBoundingClientRect(), canvas.width, canvas.height), canvas.width, canvas.height); paint();
  };
  canvas.onpointerup = event => {
    if (!drag) return;
    box = rectangle(drag, sourcePoint(event, canvas.getBoundingClientRect(), canvas.width, canvas.height), canvas.width, canvas.height);
    drag = null; paint(); $("saveCrop").disabled = !box || box[2]-box[0] < 12 || box[3]-box[1] < 12;
  };
  $("saveCrop").onclick = async () => {
    if (!crop || !box || busy) return;
    const payload = {...crop, video_id:videoId, frame:boxFrame, bbox:box,
      new_person_confirmed:crop.action === "add_player", same_person_confirmed:crop.action === "add_view"};
    if (await mutate("/api/roster", payload)) { cancelCrop(); render(); }
  };
  $("cancelCrop").onclick = cancelCrop;
  $("confirmRoster").onclick = () => mutate("/api/roster", {video_id:videoId, action:"confirm"});
  $("addPlayer").onclick = () => startCrop("add_player");
  $("rosterFrame").onclick = () => { if (roster().players.length) seekReference(roster().players[0], 0, true); };
  $("videoSelect").onchange = event => selectVideo(event.target.value);
  $("contactMode").onclick = () => changeMode("contacts");
  $("movementMode").onclick = () => changeMode("movement");
  $("priority").onchange = () => {
    if (busy) return;
    orderSegments(); showMovement(segments.findIndex(s => startOf(s) <= s.end));
  };
  const show = next => mode === "movement" ? showMovement(next) : showContact(next);
  const selectedIndex = () => mode === "movement" ? movementIndex : index;
  $("contactSelect").onchange = event => show(Number(event.target.value));
  $("previous").onclick = () => show(selectedIndex() - 1); $("next").onclick = () => show(selectedIndex() + 1);
  $("unlabelled").onclick = () => {
    if (mode === "movement") {
      const next = segments.findIndex((s, i) => i > movementIndex && startOf(s) <= s.end);
      return showMovement(next >= 0 ? next : segments.findIndex(s => startOf(s) <= s.end));
    }
    const next = rows.findIndex((c, i) => i > index && !state.labels[c.id]);
    showContact(next >= 0 ? next : Math.max(0, rows.findIndex(c => !state.labels[c.id])));
  };
  $("other").onclick = () => assign("other_team"); $("uncertain").onclick = () => assign("unsure");
  $("notContact").onclick = () => assign("not_contact"); $("reset").onclick = () => assign("reset");
  $("falsePositive").onclick = () => assign("false_positive");
  $("confirmDuplicate").onclick = () => pendingDuplicate && assign("player", pendingDuplicate.playerId);
  $("cancelDuplicate").onclick = clearDuplicate;
  $("partialMovement").onchange = clearDuplicate;
  $("backFrame").onclick = () => seek(observedFrame() - 1); $("nextFrame").onclick = () => seek(observedFrame() + 1);
  $("toContact").onclick = () => {
    if (mode === "movement") { if (segment()) seek(Math.min(startOf(segment()), segment().end)); }
    else seek(contactFrame(current()));
  };
  $("seek").oninput = event => seek(Number(event.target.value));
  $("speed").onchange = event => { video.playbackRate = Number(event.target.value); };
  $("play").onclick = () => {
    if (crop) return;
    if (!video.paused) { video.pause(); stopAt = null; return; }
    if (mode === "movement") {
      if (!segment() || movementLoading) return;
      seek(startOf(segment()) <= segment().end ? startOf(segment()) : segment().start);
      stopAt = Math.min(metadata().num_frames, segment().end + 1) / metadata().fps;
    } else {
      seek(contactFrame(current()) - Math.round(metadata().fps * 2));
      stopAt = frameTime(Math.min(metadata().num_frames - 1, contactFrame(current()) + Math.round(metadata().fps * 2)), metadata().fps);
    }
    video.play().catch(failure => error(failure.message));
  };
  video.ontimeupdate = () => {
    if (!project || !videoId) return;
    const f = observedFrame(); $("seek").value = f;
    $("frame").textContent = `f${f} · ${(f / metadata().fps).toFixed(2)} с`;
    if (stopAt !== null && video.currentTime >= stopAt) {
      video.pause(); stopAt = null;
      if (mode === "movement" && segment()) seek(segment().end);
    }
    if (referenceFocus.length || (mode === "movement" && !video.requestVideoFrameCallback)) paint();
  };
  video.onerror = () => error("Не вдалося відкрити відео. Перевірте доступність вихідного файлу.");
  video.onseeking = () => {
    if (crop) { box = null; drag = null; boxFrame = null; paint(); $("saveCrop").disabled = true; }
  };
  video.onplay = () => { if (crop) video.pause(); };
  video.onseeked = () => paint();
  if (video.requestVideoFrameCallback) {
    const onFrame = (_, info) => {
      if (mode === "movement" && videoId && !video.paused) {
        const frame = Math.round(info.mediaTime * metadata().fps);
        if (stopAt !== null && segment() && frame >= segment().end) seek(segment().end);
        else paint(frame);
      }
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  }
  document.onkeydown = event => {
    if (!project || busy || event.repeat || event.ctrlKey || event.altKey || event.metaKey ||
        event.target.isContentEditable) return;
    const key = event.key.toLowerCase();
    const {tagName, type} = event.target;
    const inMotion = mode === "movement";
    const playSegment = inMotion && (event.code === "KeyA" || key === "a");
    const markOutside = inMotion && (event.code === "KeyD" || key === "d");
    const markUnsure = inMotion && (event.code === "KeyU" || key === "u");
    const flagDetection = inMotion && (event.code === "KeyF" || key === "f");
    if (["SELECT","TEXTAREA"].includes(tagName) ||
        (tagName === "INPUT" && !["checkbox","range"].includes(type))) return;
    if (inMotion && pendingDuplicate && key === "enter" && event.target !== $("cancelDuplicate")) {
      event.preventDefault();
      return $("confirmDuplicate").click();
    }
    if (["INPUT","VIDEO"].includes(tagName) && !playSegment && !markOutside && !markUnsure && !flagDetection) return;
    if (/^[1-9]$/.test(key)) { const p = roster().players.find(p => p.player_id === "P" + key); if (p) assign("player", p.player_id); }
    else if (markOutside || (!inMotion && key === "o")) assign("other_team");
    else if (markUnsure || key === "u") assign("unsure");
    else if (key === "n") assign("not_contact");
    else if (key === "arrowleft") show(selectedIndex() - 1); else if (key === "arrowright") show(selectedIndex() + 1);
    else if (!inMotion && key === "a") seek(observedFrame() - 1); else if (!inMotion && key === "d") seek(observedFrame() + 1);
    else if (flagDetection) $("falsePositive").click();
    else if (key === " " || playSegment) $("play").click(); else return;
    event.preventDefault();
  };
  try {
    project = await json("/api/project"); state = project.state; progress = project.progress;
    $("videoSelect").replaceChildren(...Object.keys(project.videos).map(id => {
      const option = document.createElement("option"); option.value = id; option.textContent = id; return option;
    }));
    $("loading").classList.add("hidden"); $("workspace").classList.remove("hidden");
    const first = Object.keys(project.videos).find(id => progress[id].unlabelled > 0) || Object.keys(project.videos)[0];
    selectVideo(first);
  } catch (failure) { $("loading").textContent = failure.message; }
}
