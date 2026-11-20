import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {targets,mutation} from './generation_targets.mjs';
const require=createRequire(process.env.WEB_PROBE_NODE_MODULES?path.join(process.env.WEB_PROBE_NODE_MODULES,'package.json'):import.meta.url);
const {chromium}=require('playwright'),sharp=require('sharp');
const run=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),project=path.resolve(run,'../..'),data=path.join(project,'data/template-probes-01'),out=path.join(run,'outputs'),images=path.join(project,'data/template-probes-devcheck-01');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),write=async(p,x)=>fs.writeFile(p,JSON.stringify(x,null,2)+'\n'),hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const protocol=await read(path.join(run,'protocol.json')),old=await read(path.join(data,'manifest.json')),oldbases=await read(path.join(data,'baselines.json'));
await fs.mkdir(images,{recursive:false});
const lockpaths=[path.join(run,'protocol.json'),...['generation_check.mjs','generation_targets.mjs'].map(n=>path.join(run,'code',n)),path.join(run,'../20261120-mac-05/code/mutations.mjs'),...protocol.generation_check.families.map(f=>path.join(data,'pages',f+'.html'))];
await write(path.join(run,'generation_lock.json'),{scope:'Development-only pixel feasibility, no model inference',hashes:Object.fromEntries(await Promise.all(lockpaths.map(async p=>[path.relative(project,p),hash(await fs.readFile(p))]))) });
const browser=await chromium.launch({channel:'chrome',headless:true}),context=await browser.newContext({viewport:{width:1280,height:800},deviceScaleFactor:1,colorScheme:'light',locale:'en-US',timezoneId:'Asia/Shanghai'});await context.route('**/*',r=>r.abort());const page=await context.newPage(),results=[],viewstats=[];
const pixel=async()=>sharp(await page.screenshot({animations:'disabled'})).removeAlpha().raw().toBuffer();
async function load(html,y){await page.setContent(html,{waitUntil:'domcontentloaded'});await page.evaluate(()=>document.fonts.ready);await page.evaluate(y=>window.scrollTo(0,y),y);await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));}
try{
 for(const family of protocol.generation_check.families){
  const html=await fs.readFile(path.join(data,'pages',family+'.html'),'utf8');
  for(const y of protocol.generation_check.views){
   await load(html,y);const actualY=await page.evaluate(()=>scrollY);const nodes=await page.evaluate(()=>{
    const result=[];let i=0;
    for(const el of document.body.querySelectorAll('*')){
     const r=el.getBoundingClientRect(),s=getComputedStyle(el);if(r.width<8||r.height<8||r.left<4||r.top<4||r.right>1276||r.bottom>796||s.visibility==='hidden'||s.display==='none'||Number(s.opacity)<.05||el.closest('pre,code,svg')||['SCRIPT','STYLE','SVG','PATH'].includes(el.tagName))continue;
     const id=String(i++);el.setAttribute('data-check-id',id);const children=[...el.children].filter(c=>getComputedStyle(c).display!=='none'&&c.getBoundingClientRect().width>0);
     const num=x=>parseFloat(x)||0;const used=children.reduce((sum,c)=>{const cs=getComputedStyle(c);return sum+c.getBoundingClientRect().width+num(cs.marginLeft)+num(cs.marginRight);},0)+num(s.columnGap)*Math.max(0,children.length-1);
     result.push({id,tag:el.tagName,display:s.display,width:r.width,height:r.height,children:el.children.length,visibleChildren:children.length,backgroundColor:s.backgroundColor,borderTopWidth:s.borderTopWidth,borderTopStyle:s.borderTopStyle,transform:s.transform,justifyContent:s.justifyContent,flexDirection:s.flexDirection,flexWrap:s.flexWrap,freeWidth:el.clientWidth-num(s.paddingLeft)-num(s.paddingRight)-used});
    }return result;
   });
   const pristine=await page.content();await load(pristine,actualY);const before=await pixel(),beforeHash=hash(before),baselineId=`${family}_v${y}`;const expected=oldbases.find(b=>b.id===baselineId);if(expected&&expected.before_sha256!==beforeHash)throw new Error('Baseline mismatch to frozen run05 '+baselineId);
   const seen=new Map(old.filter(r=>r.baseline_id===baselineId).map(r=>[r.after_sha256,r.label]));const stats={family,baseline_id:baselineId,eligible_position:targets(nodes,'position').length,eligible_justify:targets(nodes,'justify').length,restoration_identical:false};
   for(const label of protocol.generation_check.labels){
    const selected=targets(nodes,label);if(!selected.length)results.push({family,baseline_id:baselineId,label,status:'no_eligible_layout_target'});
    for(const t of selected)for(const level of protocol.generation_check.levels){
     await load(pristine,actualY);if(hash(await pixel())!==beforeHash)throw new Error('Pristine restoration drift');const change=mutation(t,label,level),id=`${baselineId}_${label}_target${t.id}_level${level}`;
     await page.evaluate(c=>document.querySelector(`[data-check-id="${c.target}"]`).style.setProperty(c.property,c.value,'important'),change);await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
     const png=await page.screenshot({animations:'disabled'}),after=await sharp(png).removeAlpha().raw().toBuffer(),afterHash=hash(after);let changed=0;for(let i=0;i<before.length;i+=3)if(before[i]!==after[i]||before[i+1]!==after[i+1]||before[i+2]!==after[i+2])changed++;
     const collision=seen.get(afterHash),status=!changed?'no_changed_pixels':collision?'pixel_collision_'+collision:'valid_new_pixels';
     results.push({id,family,baseline_id:baselineId,label,level,target_id:t.id,status,changed_pixels:changed,before_sha256:beforeHash,after_sha256:afterHash,property:change.property,value:change.value});
     if(status==='valid_new_pixels'){seen.set(afterHash,label);await fs.writeFile(path.join(images,id+'.png'),png);}
    }
   }
   await load(pristine,actualY);if(hash(await pixel())!==beforeHash)throw new Error('Final restoration drift');stats.restoration_identical=true;viewstats.push(stats);console.log('generation checked',baselineId,JSON.stringify(stats));
  }
 }
 const counts={};for(const label of protocol.generation_check.labels){counts[label]={};for(const family of protocol.generation_check.families){const rows=results.filter(r=>r.label===label&&r.family===family);counts[label][family]={attempted:rows.filter(r=>r.id).length,valid_new:rows.filter(r=>r.status==='valid_new_pixels').length,status_counts:Object.fromEntries([...new Set(rows.map(r=>r.status))].map(s=>[s,rows.filter(r=>r.status===s).length]))};}}
 await write(path.join(out,'generation_decisions.json'),results);await write(path.join(out,'generation_summary.json'),{scope:'Development-only eligibility, no model scores and no addition to frozen corpus',families:protocol.generation_check.families,views:viewstats,counts,required_conditions:{position:'Visible filled/bordered non-inline box; moving an unboxed text may be pixel-identical to spacing.',justify:'Horizontal flex container with2+ children and unused free width. Need suitable independent training-family pages; success here is not training coverage.'},browser_version:browser.version(),confirmation_sources_used:false});console.log('GENERATION CHECK COMPLETE',JSON.stringify(counts));
}finally{await context.close();await browser.close();}
