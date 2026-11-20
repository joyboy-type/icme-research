import test from 'node:test';
import assert from 'node:assert/strict';
let assessPair;
try{({assessPair}=await import('./capture_contract.mjs'));}catch{}
const base={ready:true,networkErrors:[],resourceSignature:'same',artifactSha:'artifact',screenshot:'a.png',launchId:'a'};
test('a reused browser or PNG cannot count as an independent repeat',()=>{
 assert.equal(typeof assessPair,'function','Repeat contract missing');assert.equal(assessPair(base,{...base,screenshot:'b.png'}).valid,false);assert.equal(assessPair(base,{...base,launchId:'b'}).valid,false);
});
test('failed resources and differing source bytes invalidate a no-change control',()=>{
 assert.equal(typeof assessPair,'function','Resource contract missing');const b={...base,launchId:'b',screenshot:'b.png'};assert.equal(assessPair(base,{...b,networkErrors:['missing.js']}).valid,false);assert.equal(assessPair(base,{...b,resourceSignature:'different'}).valid,false);
});
test('fresh launches of same complete source remain controls even if pixels differ',()=>{
 assert.equal(typeof assessPair,'function','Independent capture contract missing');assert.equal(assessPair({...base,pixelSha:'x'},{...base,launchId:'b',screenshot:'b.png',pixelSha:'y'}).valid,true);
});
let summarizeResources;
try{({summarizeResources}=await import('./capture_contract.mjs'));}catch{}
test('captured resource rows use URL string properties and expose conflicting bytes',()=>{
 assert.equal(typeof summarizeResources,'function','Resource summarizer missing');
 const a={url:'https://example.test/app.js',type:'script',sha256:'one'};const result=summarizeResources([a,{...a},{...a,sha256:'two'}]);assert.equal(result.resources.length,1);assert.equal(result.errors.length,1);assert.equal(result.resources[0].url,'https://example.test/app.js');assert.equal(typeof result.signature,'string');
});
