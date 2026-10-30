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

async function fixture() {
  class Element {
    children = []; value = ''; disabled = false; checked = true; readyState = 1; currentTime = 0; paused = true;
    classList = {add() {}, remove() {}, toggle() {}};
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    getContext() { return {clearRect() {},fillRect() {},strokeRect() {},fillText() {}}; }
    pause() { this.paused = true; }
    play() { this.paused = false; return Promise.resolve(); }
  }
  const html = readFileSync('action_recognition/annotator/contact_author.html', 'utf8');
  const nodes = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1],new Element()]));
  const project = {videos:{game:{fps:30,width:200,height:100,num_frames:200}},
    contacts:[{id:'c0',video_id:'game',frame:30,type:'pass',timing_verified:true},
      {id:'c1',video_id:'game',frame:60,type:'attack',timing_verified:true}],
    state:{revision:0,labels:{},rosters:{game:{confirmed:true,players:[1,2].map(n=>({player_id:`P${n}`,references:[{frame:10,bbox:[10,10,30,80]}]}))}}},
    progress:{game:{total:2,reviewed:0,unlabelled:2,decisions:{}}}};
  const requests = []; let finish;
  const document = {getElementById:id=>nodes[id],createElement:()=>new Element(),
    querySelectorAll(selector) {
      const found = [];
      function walk(element) { if (element.className === selector.slice(1)) found.push(element); for (const child of element.children) walk(child); }
      walk(nodes.cards); return found;
    }};
  const context = vm.createContext({module:{exports:{}},console,fetch:async (url,options) => {
    if (!options) return {ok:true,json:async()=>project};
    requests.push(JSON.parse(options.body));
    return new Promise(resolve => { finish = (ok=true) => {
      const payload = requests.at(-1);
      if (ok) {
        project.state = structuredClone(project.state); project.state.revision++;
        project.state.labels[payload.contact_id] = {decision:payload.decision,player_id:payload.player_id};
        project.progress.game = {total:2,reviewed:1,unlabelled:1,decisions:{player:1}};
      }
      resolve({ok,status:ok?200:409,json:async()=>ok ? {state:project.state,progress:project.progress} : {error:'Stale revision'}});
    }; });
  }});
  vm.runInContext(readFileSync('action_recognition/annotator/contact_author.js','utf8'),context);
  context.document = document; await vm.runInContext('boot()',context);
  return {nodes,document,requests,finish:ok=>finish(ok),project};
}

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
