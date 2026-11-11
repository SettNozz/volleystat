const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');

async function fixture({stale = false, point = false} = {}) {
  class Element {
    children = []; dataset = {}; attributes = {}; listeners = {}; value = ''; checked = false; disabled = false; hidden = false;
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    setAttribute(key, value) { this.attributes[key] = value; }
    querySelectorAll() { return this.children.flatMap(item => item.children); }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    click() { this.onclick?.(); }
    requestSubmit() { this.onsubmit?.({preventDefault() {}}); }
  }
  const names = ['summary','status','list','current','reasons','contact-form','point-form','frame','type','side','author',
    'detection-verified','type-verified','save','remove','reset','dismiss','point-winner','before','after','filter-status',
    'filter-issue','sort','history','prev','next','play','add','use-time','step-back','step-forward','refresh','video'];
  const nodes = Object.fromEntries(names.map(name => ['review-' + name, new Element()]));
  nodes['developer-review-data'] = {textContent: JSON.stringify({api:'/api/developer_review/test',pointApi:'/api/statistics/test',
    fps:30,totalFrames:300,source:{stats_sha256:'s',annotations_sha256:'a'},revision:'r'})};
  nodes['review-filter-status'].value = 'pending'; nodes['review-filter-issue'].value = 'all'; nodes['review-sort'].value = 'priority';
  nodes['review-before'].value = '1'; nodes['review-after'].value = '2';
  const video = nodes['review-video']; video.currentTime = 0;
  video.pause = () => { video.paused = true; }; video.play = () => { video.paused = false; return Promise.resolve(); };
  const item = {ref:'event:0',kind:'contact',event_index:0,frame:60,type:'attack',side:null,player_id:null,
    status:'pending',priority:80,reasons:['unresolved_author'],review_axes:{}};
  if (point) Object.assign(item,{ref:'point:service_attempts_fsm:0',kind:'point',attempt_ref:'service_attempts_fsm:0',type:'point',reasons:['point_winner_unresolved']});
  const data = {source:{stats_sha256:'s',annotations_sha256:'a'},revision:'r',history:[],
    queue:{items:[item],summary:{total:1,pending:1},contact_types:['attack','set'],roster_ids:['P1']}};
  const calls = [], listeners = {};
  const context = vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new Element(),
      querySelectorAll:()=>[],addEventListener:(name,listener)=>{listeners[name]=listener;}},
    crypto:{randomUUID:()=> 'review-id'},fetch:async(url,options)=>{
      if (options) calls.push({url,body:JSON.parse(options.body)});
      return {ok:!options || !stale,status:options && stale ? 409 : 200,
        json:async()=> options && stale ? {error:'Analysis changed'} : data};
    }});
  vm.runInContext(readFileSync('viewer/static/developer_review.js','utf8'),context);
  await new Promise(setImmediate);
  return {nodes,video,calls,listeners,context};
}

test('author-only save never claims event detection or type confirmation', async()=>{
  const {nodes,calls}=await fixture();
  nodes['review-frame'].value='60'; nodes['review-type'].value='attack'; nodes['review-side'].value=''; nodes['review-author'].value='P1';
  nodes['review-contact-form'].onsubmit({preventDefault(){}});
  await new Promise(setImmediate);
  assert.deepEqual(calls[0].body,{source:{stats_sha256:'s',annotations_sha256:'a'},revision:'r',edit:{
    action:'update',event_index:0,changes:{player_id:'P1',other_side:false}}});
});

test('missed contact uses current frame and stable review id without geometry', async()=>{
  const {nodes,video,calls}=await fixture();
  video.currentTime=3;
  nodes['review-add'].onclick();
  assert.equal(nodes['review-frame'].value,90);
  nodes['review-type'].value='set';
  nodes['review-contact-form'].onsubmit({preventDefault(){}});
  await new Promise(setImmediate);
  assert.equal(calls[0].body.edit.action,'add');
  assert.equal(calls[0].body.edit.review_id,'review-id');
  assert.equal(calls[0].body.edit.changes.frame,90);
  assert.equal(calls[0].body.edit.changes.player_id,null);
  assert.equal('ball' in calls[0].body.edit.changes,false);
});

test('stale revision stays visible and does not pretend the save succeeded', async()=>{
  const {nodes}=await fixture({stale:true});
  nodes['review-dismiss'].onclick();
  await new Promise(setImmediate);
  assert.match(nodes['review-status'].textContent,/Analysis changed.*Refresh the queue/);
  assert.equal(nodes['review-status'].attributes.role,'alert');
});

test('clip playback stops at requested boundary and keyboard ignores focused inputs',async()=>{
  const {nodes,video,listeners}=await fixture();
  nodes['review-play'].onclick();
  assert.equal(video.currentTime,1);
  video.currentTime=4.1; video.listeners.timeupdate(); assert.equal(video.paused,true);
  let prevented=false;
  listeners.keydown({target:{closest:()=>true},key:'x',preventDefault(){prevented=true;}});
  assert.equal(prevented,false);
});

test('point review includes the current workspace revision',async()=>{
  const {nodes,calls}=await fixture({point:true});
  nodes['review-point-winner'].value='b';
  await nodes['review-point-form'].onsubmit({preventDefault(){}});
  assert.deepEqual(calls[0],{url:'/api/statistics/test',body:{source:{stats_sha256:'s',annotations_sha256:'a'},
    revision:'r',attempt_ref:'service_attempts_fsm:0',winner_side:'b',reset:false}});
});
