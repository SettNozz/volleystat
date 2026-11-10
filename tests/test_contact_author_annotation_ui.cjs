const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const {frameTime, rectangle, sourcePoint} = require('../action_recognition/annotator/contact_author.js');
const fps = 30000 / 1001;
for (const f of [0,1,925,5012,21300]) assert.equal(Math.floor(frameTime(f, fps)*fps), f);
assert.deepEqual(sourcePoint({clientX:150,clientY:100},{left:50,top:50,width:200,height:100},3840,2160),[1920,1080]);
assert.deepEqual(rectangle([120,100],[30,40],200,150),[30,40,120,100]);
assert.deepEqual(rectangle([10,20],[-20,200],200,150),[0,20,10,150]);
console.log('Contact-author UI frame and crop checks passed');

async function fixture({people=[1,2],confirmed=true,movement=false} = {}) {
  class Element {
    children = []; value = ''; disabled = false; checked = true; readyState = 1; currentTime = 0; paused = true;
    classList = {add() {}, remove() {}, toggle() {}};
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    getContext() { return {clearRect() {},fillRect() {},strokeRect() {},fillText() {}}; }
    pause() { this.paused = true; }
    play() { this.paused = false; return Promise.resolve(); }
    click() { if (!this.disabled) return this.onclick?.(); }
  }
  const html = readFileSync('action_recognition/annotator/contact_author.html', 'utf8');
  const nodes = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1],new Element()]));
  const project = {videos:{game:{fps:30,width:200,height:100,num_frames:200,movement_available:movement}},
    contacts:[{id:'c0',video_id:'game',frame:30,type:'pass',timing_verified:true},
      {id:'c1',video_id:'game',frame:60,type:'attack',timing_verified:true}],
    state:{revision:0,labels:{},rosters:{game:{confirmed,players:people.map(n=>({player_id:`P${n}`,references:[{frame:10,bbox:[10,10,30,80]}]}))}}},
    progress:{game:{total:2,reviewed:0,unlabelled:2,decisions:{}}}};
  const requests = [], deadlines = []; let finish, commit;
  const document = {getElementById:id=>nodes[id],createElement:()=>new Element(),
    querySelectorAll(selector) {
      const found = [];
      function walk(element) { if (element.className === selector.slice(1)) found.push(element); for (const child of element.children) walk(child); }
      walk(nodes.cards); return found;
    }};
  const context = vm.createContext({module:{exports:{}},console,AbortController,
    setTimeout(callback) { const timer={callback,active:true};deadlines.push(timer);return timer; },
    clearTimeout(timer) { timer.active=false; },
    fetch:async (url,options) => {
    if (url === '/api/movement/game') return {ok:true,json:async()=>({segments:[
      {id:'3:0',start:3,end:5,priority:[0,0]}, {id:'10:0',start:10,end:12,priority:[1,0]}]})};
    if (url.startsWith('/api/movement/game/')) return {ok:true,json:async()=>({boxes:[[5,5,20,80],[5,5,20,80],[5,5,20,80]]})};
    if (!options?.method) return {ok:true,json:async()=>project};
    requests.push(JSON.parse(options.body));
    return new Promise((resolve,reject) => { options.signal?.addEventListener('abort',()=>reject(Object.assign(new Error('Aborted'),{name:'AbortError'})));
      commit = () => {
      const payload = requests.at(-1);
        project.state = structuredClone(project.state); project.state.revision++;
        const roster=project.state.rosters.game;
        if (url==='/api/movement') {
          const all = (project.state.movement_reviews ||= {game:{}}).game;
          const spans = (all[payload.segment_id] ||= []);
          spans.push({start:(spans.at(-1)?.end ?? 2)+1,end:payload.end,decision:payload.decision,player_id:payload.player_id,verified:true});
        } else if (url==='/api/event') {
          project.state.event_reviews||={};
          if(payload.action==='reset')delete project.state.event_reviews[payload.contact_id];
          else project.state.event_reviews[payload.contact_id]={type:payload.type,label_source:'human',verified:true};
        } else if (payload.action==='remove_player') {
          const player=roster.players.find(p=>p.player_id===payload.player_id);
          const affected=Object.fromEntries(Object.entries(project.state.labels).filter(([_,label])=>label.player_id===payload.player_id));
          roster.removed_players||=[];roster.removed_players.push({player,unassigned_labels:affected});
          roster.players=roster.players.filter(p=>p!==player);
          for(const key of Object.keys(affected))delete project.state.labels[key];
        } else if(payload.action==='restore_player') {
          const entry=roster.removed_players.pop();roster.players.push(entry.player);
          for(const[key,label]of Object.entries(entry.unassigned_labels))if(!project.state.labels[key])project.state.labels[key]=label;
        } else project.state.labels[payload.contact_id] = {decision:payload.decision,player_id:payload.player_id};
        const count=Object.keys(project.state.labels).length;
        project.progress.game = {total:2,reviewed:count,unlabelled:2-count,decisions:{player:count}};
      };
      finish = (ok=true, overlap=null) => {
      if(ok)commit();
        resolve({ok,status:ok?200:overlap?400:409,json:async()=>ok ? {state:project.state,progress:project.progress} : {error:overlap?'P1 вже позначено':'Stale revision',overlap}});
    }; });
  }});
  vm.runInContext(readFileSync('action_recognition/annotator/contact_author.js','utf8'),context);
  context.document = document; await vm.runInContext('boot()',context);
  return {nodes,document,requests,finish:(...args)=>finish(...args),commit:()=>commit(),project,
          timeout:()=>deadlines.filter(t=>t.active).at(-1).callback()};
}

test('movement partial save binds the displayed frame and advances only past the completed segment', async () => {
  const f = await fixture({movement:true});
  f.nodes.priority.checked = false;
  await f.nodes.movementMode.onclick();
  f.nodes.video.readyState = 2;
  f.nodes.video.currentTime = frameTime(4,30);
  f.nodes.partialMovement.checked = true;
  const pick = f.document.querySelectorAll('.pick')[0];
  const saving = pick.onclick();
  assert.deepEqual(f.requests[0], {video_id:'game',segment_id:'3:0',decision:'player',player_id:'P1',end:4,revision:0});
  f.finish(); await saving;
  assert.equal(Math.floor(f.nodes.video.currentTime*30),5);
  assert.equal(f.nodes.contactSelect.value,0);
  const completing = f.document.querySelectorAll('.pick')[1].onclick();
  f.finish(); await completing;
  assert.equal(f.nodes.contactSelect.value,1);
  assert.equal(Math.floor(f.nodes.video.currentTime*30),10);
});

test('physical A plays movement and D marks outside the roster on a Ukrainian keyboard layout', async () => {
  const f = await fixture({movement:true});
  const event = {key:'ф', code:'KeyA', target:{tagName:'BODY'}, preventDefault() {}};
  f.document.onkeydown(event);
  assert.equal(f.nodes.video.paused,true);
  await f.nodes.movementMode.onclick();
  f.document.onkeydown({...event, ctrlKey:true});
  assert.equal(f.nodes.video.paused,true);
  f.document.onkeydown(event);
  assert.equal(f.nodes.video.paused,false);
  f.document.onkeydown(event);
  assert.equal(f.nodes.video.paused,true);
  const frame = f.nodes.video.currentTime;
  f.document.onkeydown({...event, key:'в', code:'KeyD'});
  assert.equal(f.requests[0].decision,'other');
  assert.equal(f.nodes.video.currentTime,frame);
  f.finish();
});

test('Enter confirms a pending duplicate with the original partial range', async () => {
  const f = await fixture({movement:true});
  f.nodes.priority.checked = false;
  await f.nodes.movementMode.onclick();
  f.nodes.video.readyState = 2;
  f.nodes.video.currentTime = frameTime(4,30);
  f.nodes.partialMovement.checked = true;
  const enter = {key:'Enter', target:{tagName:'VIDEO'}, preventDefault() {}};
  f.document.onkeydown(enter);
  assert.equal(f.requests.length,0);
  const pick = f.document.querySelectorAll('.pick')[0];
  const saving = pick.onclick();
  f.finish(false,{frame:3,bbox:[5,5,20,80]}); await saving;
  assert.equal(Math.floor(f.nodes.video.currentTime*30),3);
  assert.match(f.nodes.duplicateMessage.textContent,/f3–4/);
  const confirmed = f.document.onkeydown(enter);
  assert.deepEqual(f.requests[1], {...f.requests[0],confirm_overlap:true});
  f.finish(); await confirmed;
  assert.equal(Math.floor(f.nodes.video.currentTime*30),5);
});

test('save binds original contact and blocks navigation until acknowledgement', async () => {
  const f = await fixture();
  const saving = f.nodes.other.onclick();
  assert.equal(f.requests[0].contact_id,'c0');
  assert.equal(f.requests[0].decision,'other_team');
  f.nodes.next.onclick();
  assert.equal(f.nodes.contactSelect.value,0);
  assert.equal(f.nodes.videoSelect.disabled,true);
  f.finish(); await saving;
  assert.equal(f.nodes.contactSelect.value,1);
  assert.equal(Math.floor(f.nodes.video.currentTime*30),60);
  assert.match(f.nodes.saved.textContent,/Збережено/);
});

test('rejected save does not auto-advance and crop cancellation restores choices', async () => {
  const f = await fixture();
  f.nodes.addPlayer.onclick(); assert.equal(f.nodes.other.disabled,true);
  f.nodes.cancelCrop.onclick(); assert.equal(f.nodes.other.disabled,false);
  const saving = f.nodes.other.onclick(); f.finish(false); await saving;
  assert.equal(f.nodes.contactSelect.value,0);
  assert.match(f.nodes.error.textContent,/Stale revision/);
  assert.match(f.nodes.saved.textContent,/не збережено/);
  assert.equal(f.project.state.revision,0);
});

test('remove card asks before sending, cancels safely, and restores without changing other IDs', async () => {
  const f=await fixture();
  f.document.querySelectorAll('.removePlayer')[1].onclick();
  assert.equal(f.requests.length,0);assert.equal(f.nodes.other.disabled,true);
  assert.match(f.nodes.removeTitle.textContent,/P2/);
  f.nodes.cancelRemove.onclick();assert.equal(f.nodes.other.disabled,false);
  f.document.querySelectorAll('.removePlayer')[1].onclick();
  const removing=f.nodes.confirmRemove.onclick();
  assert.deepEqual(f.requests[0],{action:'remove_player',video_id:'game',player_id:'P2',removal_confirmed:true,unassign_contacts_confirmed:false,revision:0});
  f.finish();await removing;
  assert.deepEqual(f.project.state.rosters.game.players.map(p=>p.player_id),['P1']);
  assert.match(f.nodes.restorePlayer.textContent,/P2/);
  const restoring=f.nodes.restorePlayer.onclick();f.finish();await restoring;
  assert.deepEqual(f.project.state.rosters.game.players.map(p=>p.player_id),['P1','P2']);
});

test('navigation dismisses a pending removal and a stale confirmation cannot delete a card',async()=>{
  const f=await fixture();f.document.querySelectorAll('.removePlayer')[0].onclick();
  f.nodes.next.onclick();await f.nodes.confirmRemove.onclick();
  assert.equal(f.requests.length,0);assert.equal(f.nodes.contactSelect.value,1);
});

test('timeout after a committed save reconciles state, unlocks navigation and never retries or advances',async()=>{
  const f=await fixture();const saving=f.nodes.other.onclick();
  assert.equal(f.nodes.videoSelect.disabled,true);
  f.commit();f.timeout();await saving;
  assert.equal(f.requests.length,1);assert.equal(f.project.state.revision,1);
  assert.equal(f.nodes.videoSelect.disabled,false);assert.equal(f.nodes.contactSelect.value,0);
  assert.match(f.nodes.error.textContent,/15 секунд/);assert.match(f.nodes.saved.textContent,/Стан оновлено/);
  assert.match(f.nodes.eventMeta.textContent,/Інша команда/);
});

test('saving another contact retains existing photo elements and stable portrait URLs',async()=>{
  const f=await fixture();const cards=f.nodes.cards.children,src=cards[0].children[0].children[0].src;
  const saving=f.nodes.other.onclick();f.finish();await saving;
  assert.equal(f.nodes.cards.children,cards);
  assert.equal(cards[0].children[0].children[0].src,src);
});

test('numeric shortcuts use persistent player ID after gaps in the roster',async()=>{
  const f=await fixture({people:[1,3]});
  const event=key=>({key,repeat:false,target:{tagName:'BODY'},preventDefault(){}});
  f.document.onkeydown(event('2'));assert.equal(f.requests.length,0);
  f.document.onkeydown(event('3'));assert.equal(f.requests[0].player_id,'P3');f.finish();
});

test('changing event type saves the original contact, keeps author and does not auto-advance',async()=>{
  const f=await fixture();f.project.state.labels.c0={decision:'player',player_id:'P1'};
  const author=structuredClone(f.project.state.labels.c0);
  const saving=f.nodes.eventType.onchange({target:{value:'set'}});
  assert.deepEqual(f.requests[0],{contact_id:'c0',action:'set',type:'set',revision:0});
  assert.equal(f.nodes.videoSelect.disabled,true);assert.equal(f.nodes.eventType.disabled,true);
  f.nodes.next.onclick();assert.equal(f.nodes.contactSelect.value,0);
  f.finish();await saving;
  assert.deepEqual(f.project.state.labels.c0,author);assert.equal(f.nodes.contactSelect.value,0);
  assert.equal(f.nodes.eventType.value,'set');assert.match(f.nodes.event.textContent,/Передача/);
  assert.match(f.nodes.eventTypeStatus.textContent,/людиною/);assert.equal(f.nodes.resetEventType.disabled,false);
  const resetting=f.nodes.resetEventType.onclick();f.finish();await resetting;
  assert.equal(f.nodes.eventType.value,'pass');assert.equal(f.nodes.resetEventType.disabled,true);
  assert.equal(f.project.contacts[0].type,'pass');assert.deepEqual(f.project.state.labels.c0,author);
});

test('rejected event change reverts selector and cannot edit the next contact',async()=>{
  const f=await fixture();f.nodes.eventType.value='attack';
  const saving=f.nodes.eventType.onchange({target:{value:'attack'}});f.nodes.next.onclick();f.finish(false);await saving;
  assert.equal(f.nodes.contactSelect.value,0);assert.equal(f.nodes.eventType.value,'pass');
  assert.equal(f.project.state.event_reviews,undefined);assert.deepEqual(f.project.state.labels,{});
});

test('event type can be corrected before confirming person cards',async()=>{
  const f=await fixture({confirmed:false});assert.equal(f.nodes.other.disabled,true);assert.equal(f.nodes.eventType.disabled,false);
  const saving=f.nodes.eventType.onchange({target:{value:'dig'}});f.finish();await saving;
  assert.equal(f.nodes.eventType.value,'dig');assert.equal(f.nodes.other.disabled,true);
  assert.deepEqual(f.project.state.labels,{});assert.equal(f.project.progress.game.reviewed,0);
});
