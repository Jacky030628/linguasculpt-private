const assert=require('node:assert/strict');
const {changedSegments}=require('../frontend/highlight.js');
function check(before,after,changed) {
  const segments=changedSegments(before,after);
  assert.equal(segments.map(s=>s.text).join(''),after);
  assert.deepEqual(segments.filter(s=>s.changed).map(s=>s.text),changed);
}
check('The old draft.','The new draft.',['new']);
check('A draft.','A clear draft.',['clear']);
check('A clear draft.','A draft.',[]);
check('Same text.','Same text.',[]);
check('First\n\nSecond.','First\n\nSecond!', ['!']);
check('','<script>alert("x")</script>', ['<','script','>','alert','(','"','x','"',')','<','/','script','>']);
check('你好，世界。','你好，新世界。',['新世界']);
console.log('7 highlight checks passed');
