const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const {readFileSync}=require('node:fs');

function fixture(){
  class Element {
    children=[];checked=false;textContent='';currentTime=0;paused=false;
    replaceChildren(...children){this.children=children;}
    appendChild(child){this.children.push(child);}
    pause(){this.paused=true;}
  }
  const nodes=Object.fromEntries(['labEvents','labSummary','labOnlyNew','labOnlyChanges','labVideo','labDetails','labMethod','labMethodInfo'].map(id=>[id,new Element()]));
  const event={event_index:0,source_event_index:0,frame:30,original_frame:30,contact_frame:32,type:'pass',player_id:'P1',proposed_player_id:'P1',status:'assigned',side_proposal:'b',near_side_score:.9,reason:'test_proposal',detection_evidence:{ballistic:.7,interaction:.8}};
  const novel={...event,event_index:1,source_event_index:null,frame:60,contact_frame:61,type:'contact',player_id:null,status:'unresolved',proposed_player_id:'P1'};
  const timing={all:{at_5:{f1:.8,precision:.8,recall:.8}}};
  const bundle={fps:30,roster_ids:['P1','P2'],methods:{spotter:{events:[event],detection_lovo:timing},joint:{events:[event,novel],detection_lovo:timing}}};
  const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new Element(),addEventListener(){}},requestAnimationFrame(){},bundle});
  vm.runInContext(readFileSync('viewer/static/contact_lab.js','utf8'),context);
  vm.runInContext("labBundle=bundle;labMethod='joint';labPlayer='P1';labRenderEvents()",context);
  return {context,nodes,bundle,event,novel};
}

test('new detections and uncertain authors stay separate from assigned totals',()=>{
  const {context,nodes}=fixture();
  assert.equal(nodes.labEvents.children.length,2);
  assert.match(nodes.labSummary.textContent,/1 assigned to P1.*1 uncertain suggestions/);
  nodes.labOnlyNew.checked=true;vm.runInContext('labRenderEvents()',context);
  assert.equal(nodes.labEvents.children.length,1);
  assert.match(nodes.labEvents.children[0].textContent,/contact.*Unresolved/);
  assert.match(nodes.labEvents.children[0].children[0].textContent,/NEW.*excluded from total/);
});
test('click pauses at source-backed evidence and preserves zero source index',()=>{
  const {nodes}=fixture();nodes.labEvents.children[0].onclick();
  assert.equal(nodes.labVideo.paused,true);assert.equal(Math.floor(nodes.labVideo.currentTime*30),32);
  assert.ok(nodes.labVideo.currentTime<33/30);
  assert.match(nodes.labDetails.textContent,/detector f30, evidence f32/);
  assert.match(nodes.labDetails.textContent,/Source event 0/);
});
test('author change filter compares the same source event across methods',()=>{
  const {context,nodes,bundle,event}=fixture();
  bundle.methods.joint.events[0]={...event,player_id:'P2',proposed_player_id:'P2'};
  nodes.labOnlyChanges.checked=true;
  vm.runInContext("labPlayer=null;labRenderEvents()",context);
  assert.equal(nodes.labEvents.children.length,1);
  assert.match(nodes.labEvents.children[0].children[0].textContent,/Changed: P1 → P2/);
});
test('fractional-fps seek lands inside requested evidence frame',()=>{
  const {nodes,bundle,event}=fixture();
  bundle.fps=30000/1001;event.contact_frame=1051;
  nodes.labEvents.children[0].onclick();
  assert.equal(Math.floor(nodes.labVideo.currentTime*bundle.fps),1051);
  assert.match(nodes.labDetails.textContent,/evidence f1051/);
});
