import test from 'node:test';
import assert from 'node:assert/strict';
let validateRepeat;
try{({validateRepeat}=await import('./contract.mjs'));}catch{}
const a={ready:true,networkErrors:[],resourceSignature:'r',artifactSha:'s',launchId:'one',screenshot:'one.png',stateSignature:'state',targetMediaCount:0,missingRequired:[]};
test('repeat contract rejects missing CSS even when pixels happen to match',()=>{assert.equal(typeof validateRepeat,'function');assert.equal(validateRepeat(a,{...a,launchId:'two',screenshot:'two.png',missingRequired:['simple.css']}).valid,false);});
test('repeat contract rejects a changed page state or reused browser',()=>{assert.equal(typeof validateRepeat,'function');assert.equal(validateRepeat(a,{...a,launchId:'two',screenshot:'two.png',stateSignature:'changed'}).valid,false);assert.equal(validateRepeat(a,{...a,screenshot:'two.png'}).valid,false);});
test('blocked unrelated demo media does not replace target resource checks',()=>{assert.equal(typeof validateRepeat,'function');assert.equal(validateRepeat(a,{...a,launchId:'two',screenshot:'two.png',blockedNonTarget:['kitten.jpg']}).valid,true);assert.equal(validateRepeat(a,{...a,launchId:'two',screenshot:'two.png',targetMediaCount:1}).valid,false);});
