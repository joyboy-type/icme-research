import test from 'node:test';
import assert from 'node:assert/strict';
import {plan} from './mutations.mjs';
const nodes=[{id:'0',tag:'P',text:'A source paragraph with several words',leaf:true,x:20,y:40,width:300,height:50,fontSize:'16px',fontWeight:'400',lineHeight:'24px',letterSpacing:'normal',color:'rgb(30, 40, 50)',opacity:'1',transform:'none',paddingLeft:'0px',backgroundColor:'rgba(0, 0, 0, 0)',borderTopWidth:'0px',borderTopStyle:'none',borderTopLeftRadius:'0px',display:'block',children:0},
 {id:'1',tag:'BUTTON',text:'A source button',leaf:true,x:30,y:120,width:140,height:40,fontSize:'16px',fontWeight:'400',lineHeight:'24px',letterSpacing:'normal',color:'rgb(10, 10, 10)',opacity:'1',transform:'none',paddingLeft:'12px',backgroundColor:'rgb(230, 235, 240)',borderTopWidth:'1px',borderTopStyle:'solid',borderTopLeftRadius:'4px',display:'inline-block',children:0}];
test('position changes exactly one existing element transform',()=>{const p=plan(nodes,'position',0);assert.equal(p.property,'transform');assert.equal(p.value,'translateX(1px)');assert.equal(p.target,'0');});
test('border requires an actual existing visible border',()=>{const p=plan(nodes,'border',0);assert.equal(p.target,'1');assert.equal(p.property,'border-top-width');assert.equal(p.value,'2px');});
test('rounding and gradient use a visible filled box',()=>{assert.equal(plan(nodes,'rounded',0).target,'1');assert.equal(plan(nodes,'gradient',0).target,'1');});
test('unsupported flex alignment is explicitly skipped',()=>{assert.equal(plan(nodes,'justify',0),null);});
test('text swap preserves the existing word set',()=>{const p=plan(nodes,'text',0);assert.equal(p.value,'source A paragraph with several words');});
