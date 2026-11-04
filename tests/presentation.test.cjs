const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const elements=new Map();
function element(id) {
  if(!elements.has(id)) elements.set(id,{textContent:'old result',dataset:{rawText:'old result'},style:{},hidden:false,disabled:false});
  return elements.get(id);
}
const context=vm.createContext({document:{getElementById:element,addEventListener(){}},window:{},
  proseHighlight:{render(){throw Error('Intermediate draft was rendered');}}});
vm.runInContext(fs.readFileSync('frontend/app.js','utf8'),context);
for(const status of ['running','failed','interrupted']) {
  context.showResult({id:'task',status,result:{draft:'SECRET INTERMEDIATE',best:'SECRET INTERMEDIATE',edits:[{}],events:[]}});
  assert(!element('userOutputDisplay').textContent.includes('SECRET'));
  assert.equal(element('userOutputDisplay').dataset.rawText,'');
  assert.equal(element('userModSummaryCard').hidden,true);
  assert.equal(element('userCheckAiResultBar').style.display,'none');
  assert.equal(element('userCopyBtn').disabled,true);
}
console.log('3 pending/failure presentation checks passed');
