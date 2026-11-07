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
  const current = () => rows[index];
  const reviewedType = contact => state.event_reviews?.[contact.id]?.type || contact.type;
  const metadata = () => project.videos[videoId];
  const roster = () => state.rosters[videoId];
  const labelText = label => !label ? "Без мітки" : label.decision === "player" ? label.player_id :
    ({other_team:"Інша команда", unsure:"Не впевнений", not_contact:"Це не контакт"}[label.decision]);
  const error = message => { $("error").textContent = message || ""; };

  async function json(url, options) {
    const controller = new AbortController(), deadline = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, {...options, signal:controller.signal});
      const result = await response.json();
      if (!response.ok) { const failure = new Error(result.error || "Помилка сервера"); failure.status = response.status; throw failure; }
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
    for (const id of ["other","uncertain","notContact","reset"]) $(id).disabled = disabled;
    document.querySelectorAll(".pick").forEach(button => { button.disabled = disabled; });
    document.querySelectorAll(".editReference").forEach(button => { button.disabled = busy || !!pendingRemoval; });
    document.querySelectorAll(".removePlayer").forEach(button => { button.disabled = busy || !!crop || !!pendingRemoval; });
    $("confirmRemove").disabled = busy || !pendingRemoval;
    $("cancelRemove").disabled = busy;
    $("restorePlayer").disabled = busy || !!crop || !!pendingRemoval;
    $("rosterFrame").disabled = busy || !roster().players.length;
    $("confirmRoster").disabled = busy || !roster().players.length;
    $("eventType").disabled = busy || !!crop || !!pendingRemoval;
    $("resetEventType").disabled = busy || !!crop || !!pendingRemoval || !state.event_reviews?.[current().id];
  }
  async function mutate(url, payload) {
    if (busy) return false;
    lock(true); error(""); $("saved").textContent = "Зберігаю…";
    try {
      const result = await json(url, {method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({...payload, revision:state.revision})});
      state = result.state; progress = result.progress;
      $("saved").textContent = `Збережено · ревізія ${state.revision}`;
      return true;
    } catch (failure) {
      $("saved").textContent = "Мітку не збережено";
      if (failure.status === 409 || !failure.status) {
        try { const latest = await json("/api/project"); state = latest.state; progress = latest.progress;
          if (failure.timeout) $("saved").textContent = `Стан оновлено · ревізія ${state.revision}. Перевірте поточну мітку.`;
        }
        catch (_) { /* Source conflicts stay visible; never retry a write. */ }
      }
      error(failure.message + (failure.status === 409 || failure.timeout ? " Перевірте поточну мітку перед повтором; якщо стан не оновився, перезавантажте сторінку." : ""));
      return false;
    } finally { lock(false); render(); }
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
    cancelCrop(); seek(current().frame); render();
  }
  function selectVideo(id) {
    if (busy) return;
    cancelRemoval();
    cancelCrop(); videoId = id; rows = project.contacts.filter(c => c.video_id === id);
    index = rows.findIndex(c => !state.labels[c.id]); if (index < 0) index = 0;
    canvas.width = metadata().width; canvas.height = metadata().height;
    const version = ++loadVersion;
    video.src = "/video/" + encodeURIComponent(id);
    video.onloadedmetadata = () => { if (version === loadVersion) { video.currentTime = frameTime(pendingFrame ?? current().frame, metadata().fps); pendingFrame = null; } };
    $("seek").max = metadata().num_frames - 1;
    seek(current().frame); render();
  }
  function render() {
    if (!project || !videoId) return;
    $("videoSelect").value = videoId;
    $("contactSelect").replaceChildren(...rows.map((c, i) => {
      const option = document.createElement("option"); option.value = i;
      option.textContent = `${state.labels[c.id] ? "✓" : "○"} ${i + 1}. f${c.frame} · ${names[reviewedType(c)] || reviewedType(c)} · ${labelText(state.labels[c.id])}`;
      return option;
    }));
    $("contactSelect").value = index;
    const contact = current(), type = reviewedType(contact), review = state.event_reviews?.[contact.id];
    $("event").textContent = `${index + 1}/${rows.length} · ${names[type] || type} · f${contact.frame}`;
    $("eventMeta").textContent = `Мітка: ${labelText(state.labels[contact.id])} · Час: ${contact.timing_verified ? "перевірений людиною" : "початкова позначка, час не перевірений"}${!review && contact.event_label_source === "model" ? " · Подія знайдена моделлю; тип ще не перевірений" : ""}`;
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
    $("rosterStatus").textContent = roster().confirmed ? "Картки підтверджені · оберіть автора контакту" : "Спочатку перевірте й підтвердьте картки";
    $("confirmRoster").classList.toggle("hidden", roster().confirmed);
    $("confirmRoster").textContent = `Підтверджую: це ${roster().players.length} різних гравців`;
    $("labelGuard").textContent = roster().confirmed ? "" : roster().players.length ? "Щоб присвоювати контакти, перевірте картки справа й натисніть «Підтверджую». Зайві картки можна видалити." : "Додайте картки гравців кнопкою «Новий гравець / заміна», потім підтвердьте їх.";
    const removed = roster().removed_players || [], lastRemoved = removed[removed.length - 1];
    $("restorePlayer").classList.toggle("hidden", !lastRemoved);
    $("restorePlayer").textContent = lastRemoved ? `Відновити ${lastRemoved.player.player_id}` : "Відновити видалену картку";
    renderCards(); updateButtons();
  }
  function renderCards() {
    const signature = JSON.stringify([videoId, roster().players]);
    if (cardsSignature === signature) {
      document.querySelectorAll(".card").forEach((card, i) => card.classList.toggle("selected", state.labels[current().id]?.player_id === roster().players[i]?.player_id));
      return;
    }
    cardsSignature = signature;
    $("cards").replaceChildren();
    for (const player of roster().players) {
      const card = document.createElement("div"); card.className = "card";
      if (state.labels[current().id]?.player_id === player.player_id) card.classList.add("selected");
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
    const contactId = current().id, oldIndex = index, oldVideo = videoId;
    if (await mutate("/api/label", {contact_id:contactId, decision, player_id:playerId})) {
      if (decision !== "reset" && $("autoNext").checked && videoId === oldVideo && current().id === contactId) showContact(oldIndex + 1);
    }
  }
  async function changeEvent(action, type) {
    if (busy || crop || pendingRemoval) return false;
    return mutate("/api/event", {contact_id:current().id, action, ...(action === "set" ? {type} : {})});
  }
  $("eventType").onchange = event => changeEvent("set", event.target.value);
  $("resetEventType").onclick = () => changeEvent("reset");
  function startCrop(action, playerId = null) {
    if (busy || pendingRemoval) return;
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
  function paint() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!crop && referenceFocus.length && project && videoId && !video.seeking && video.readyState >= 2) {
      context.lineWidth = 5; context.strokeStyle = "#72dbc0"; context.fillStyle = "#72dbc0";
      context.font = "bold 44px sans-serif";
      for (const reference of referenceFocus) {
        if (reference.frame !== observedFrame()) continue;
        const [x1,y1,x2,y2] = reference.bbox;
        context.strokeRect(x1,y1,x2-x1,y2-y1); context.fillText(reference.player_id,x1,Math.max(46,y1-10));
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
  $("contactSelect").onchange = event => showContact(Number(event.target.value));
  $("previous").onclick = () => showContact(index - 1); $("next").onclick = () => showContact(index + 1);
  $("unlabelled").onclick = () => {
    const next = rows.findIndex((c, i) => i > index && !state.labels[c.id]);
    showContact(next >= 0 ? next : Math.max(0, rows.findIndex(c => !state.labels[c.id])));
  };
  $("other").onclick = () => assign("other_team"); $("uncertain").onclick = () => assign("unsure");
  $("notContact").onclick = () => assign("not_contact"); $("reset").onclick = () => assign("reset");
  $("backFrame").onclick = () => seek(observedFrame() - 1); $("nextFrame").onclick = () => seek(observedFrame() + 1);
  $("toContact").onclick = () => seek(current().frame);
  $("seek").oninput = event => seek(Number(event.target.value));
  $("speed").onchange = event => { video.playbackRate = Number(event.target.value); };
  $("play").onclick = () => {
    if (crop) return;
    if (!video.paused) { video.pause(); stopAt = null; return; }
    seek(current().frame - Math.round(metadata().fps * 2));
    stopAt = frameTime(Math.min(metadata().num_frames - 1, current().frame + Math.round(metadata().fps * 2)), metadata().fps);
    video.play().catch(failure => error(failure.message));
  };
  video.ontimeupdate = () => {
    if (!project || !videoId) return;
    const f = observedFrame(); $("seek").value = f;
    $("frame").textContent = `f${f} · ${(f / metadata().fps).toFixed(2)} с`;
    if (stopAt !== null && video.currentTime >= stopAt) { video.pause(); stopAt = null; }
    if (referenceFocus.length) paint();
  };
  video.onerror = () => error("Не вдалося відкрити відео. Перевірте доступність вихідного файлу.");
  video.onseeking = () => {
    if (crop) { box = null; drag = null; boxFrame = null; paint(); $("saveCrop").disabled = true; }
  };
  video.onplay = () => { if (crop) video.pause(); };
  video.onseeked = paint;
  document.onkeydown = event => {
    if (!project || busy || event.repeat || ["INPUT","SELECT","TEXTAREA","VIDEO"].includes(event.target.tagName)) return;
    const key = event.key.toLowerCase();
    if (/^[1-9]$/.test(key)) { const p = roster().players.find(p => p.player_id === "P" + key); if (p) assign("player", p.player_id); }
    else if (key === "o") assign("other_team"); else if (key === "u") assign("unsure"); else if (key === "n") assign("not_contact");
    else if (key === "arrowleft") showContact(index - 1); else if (key === "arrowright") showContact(index + 1);
    else if (key === "a") seek(observedFrame() - 1); else if (key === "d") seek(observedFrame() + 1);
    else if (key === " ") $("play").click(); else return;
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
