import test from 'node:test';
import assert from 'node:assert/strict';
let targets,mutation;
try{({targets,mutation}=await import('./generation_targets.mjs'));}catch{}
test('position requires visible non-inline box and preserves transform',()=>{
 assert.equal(typeof targets,'function','Box target selector missing');
 const common={id:'a',width:100,height:30,backgroundColor:'rgb(0, 0, 0)',borderTopWidth:'0px',borderTopStyle:'none',transform:'scale(1)',display:'inline'};
 const box={...common,id:'b',display:'inline-block'};assert.deepEqual(targets([common,box],'position').map(x=>x.id),['b']);assert.equal(mutation(box,'position',1).value,'scale(1) translateX(2px)');
});
test('justify excludes filled flex rows without free layout width',()=>{
 assert.equal(typeof targets,'function','Flex target selector missing');
 const filled={id:'a',width:300,height:40,display:'flex',children:2,visibleChildren:2,flexDirection:'row',flexWrap:'nowrap',freeWidth:0};
 const free={...filled,id:'b',freeWidth:50};assert.deepEqual(targets([filled,free],'justify').map(x=>x.id),['b']);
});
