import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {plan} from './mutations.mjs';
const require=createRequire(process.env.WEB_PROBE_NODE_MODULES?path.join(process.env.WEB_PROBE_NODE_MODULES,'package.json'):import.meta.url);
const {chromium}=require('playwright'),sharp=require('sharp');
const run=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),project=path.resolve(run,'../..'),data=path.join(project,'data/template-probes-01');
const read=async p=>JSON.parse(await fs.readFile(p,'utf8')),write=async(p,x)=>fs.writeFile(p,JSON.stringify(x,null,2)+'\n');
const protocol=await read(path.join(run,'protocol.json')),sources=await read(path.join(data,'sources.json'));
const browser=await chromium.launch({channel:'chrome',headless:true});
const context=await browser.newContext({viewport:{width:1280,height:800},deviceScaleFactor:1,colorScheme:'light',locale:'en-US',timezoneId:'Asia/Shanghai'});
const blocked=[];await context.route('**/*',route=>{blocked.push(route.request().url());return route.abort();});
const page=await context.newPage(),rows=[],baselines=[],skipped=[],audits=[];
const pixel=async buffer=>sharp(buffer).removeAlpha().raw().toBuffer();
const hash=buffer=>crypto.createHash('sha256').update(buffer).digest('hex');
const settle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
async function load(html,y){await page.setContent(html,{waitUntil:'domcontentloaded'});await page.evaluate(()=>document.fonts.ready);await page.evaluate(y=>window.scrollTo(0,y),y);await settle();}
async function snapshot(){return page.evaluate(()=>{
 const values=[];let i=0;
 for(const el of document.body.querySelectorAll('*')){
  const r=el.getBoundingClientRect(),s=getComputedStyle(el);
  if(r.width<8||r.height<8||r.left<4||r.top<4||r.right>1276||r.bottom>796||s.visibility==='hidden'||s.display==='none'||Number(s.opacity)<.05)continue;
  if(['SCRIPT','STYLE','PRE','CODE','SVG','PATH'].includes(el.tagName)||el.closest('pre,code,svg'))continue;
  const id=String(i++);el.setAttribute('data-probe-id',id);
  const keys=['fontSize','fontWeight','lineHeight','letterSpacing','color','opacity','transform','paddingLeft','backgroundColor','borderTopWidth','borderTopStyle','borderTopLeftRadius','display','justifyContent'];
  values.push({id,tag:el.tagName,text:el.textContent||'',leaf:el.children.length===0,children:el.children.length,x:r.x,y:r.y,width:r.width,height:r.height,...Object.fromEntries(keys.map(k=>[k,s[k]]))});
 }return values;
});}
try{
 for(const source of sources){
  const family=source.family,sourceHtml=await fs.readFile(path.join(data,source.page),'utf8'),familyHashes=new Set();let countFamily=0;
  for(const requestedY of protocol.generation.scroll_views){
   await load(sourceHtml,requestedY);const actualY=await page.evaluate(()=>window.scrollY),nodes=await snapshot(),pristine=await page.content();
   // Capture the same pristine DOM twice before accepting this baseline.
   await load(pristine,actualY);const baseline=await page.screenshot({animations:'disabled'}),before=await pixel(baseline),beforeHash=hash(before);
   await load(pristine,actualY);if(hash(await pixel(await page.screenshot({animations:'disabled'})))!==beforeHash)throw new Error('Unstable initial baseline '+family+' '+actualY);
   if(familyHashes.has(beforeHash)){skipped.push({family,requestedY,reason:'duplicate_baseline_view'});continue;}familyHashes.add(beforeHash);
   const view=`v${requestedY}`,baselineId=`${family}_${view}`,folder=path.join(data,'images',baselineId);await fs.mkdir(folder,{recursive:true});await fs.writeFile(path.join(folder,'before.png'),baseline);
   const beforePath=`data/template-probes-01/images/${baselineId}/before.png`;
   baselines.push({id:baselineId,family,split:protocol.family_split[family],requested_scroll_y:requestedY,actual_scroll_y:actualY,before_path:beforePath,before_sha256:beforeHash});
   const seen=new Map();let accepted=0;
   for(const label of protocol.generation.labels)for(let level=0;level<3;level++){
    const id=`${baselineId}_${label}_${level}`,change=plan(nodes,label,level);
    if(!change){skipped.push({id,family,label,level,reason:'no_eligible_visible_source_element'});continue;}
    await load(pristine,actualY);
    const check=await pixel(await page.screenshot({animations:'disabled'}));if(hash(check)!==beforeHash)throw new Error('Pristine baseline drift '+id);
    const original=await page.evaluate(change=>{
     const el=document.querySelector(`[data-probe-id="${change.target}"]`),r=el.getBoundingClientRect();
     const old={rect:[r.x,r.y,r.width,r.height],computed:change.property==='textContent'?el.textContent:getComputedStyle(el).getPropertyValue(change.property)};
     if(change.property==='textContent')el.textContent=change.value;else el.style.setProperty(change.property,change.value,'important');
     return old;
    },change);
    await settle();const png=await page.screenshot({animations:'disabled'}),after=await pixel(png),afterHash=hash(after);
    let count=0,over4=0,max=0,x0=1280,y0=800,x1=0,y1=0;
    for(let i=0;i<before.length;i+=3){const d=Math.max(Math.abs(before[i]-after[i]),Math.abs(before[i+1]-after[i+1]),Math.abs(before[i+2]-after[i+2]));if(d){count++;const pos=i/3,x=pos%1280,y=Math.floor(pos/1280);x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x+1);y1=Math.max(y1,y+1);}if(d>4)over4++;max=Math.max(max,d);}
    if(!count){skipped.push({id,family,label,level,reason:'no_changed_pixels',change});continue;}
    if(seen.has(afterHash)){skipped.push({id,family,label,level,reason:'duplicate_after_pixels',duplicate_of:seen.get(afterHash)});continue;}seen.set(afterHash,id);
    await fs.writeFile(path.join(folder,id+'.png'),png);accepted++;countFamily++;
    rows.push({id,baseline_id:baselineId,family,split:protocol.family_split[family],label,level,before_path:beforePath,after_path:`data/template-probes-01/images/${baselineId}/${id}.png`,before_sha256:beforeHash,after_sha256:afterHash,changed_pixels:count,changed_gt4:over4,max_delta:max,diff_bbox_xyxy:[x0,y0,x1,y1],change,original_target_rect:original.rect,old_computed:original.computed,source_url:`https://github.com/${source.repository}/blob/${source.commit_sha}/${source.html}`});
   }
   await load(pristine,actualY);const finalHash=hash(await pixel(await page.screenshot({animations:'disabled'})));if(finalHash!==beforeHash)throw new Error('Final baseline drift '+baselineId);
   audits.push({family,baseline_id:baselineId,before_sha256:beforeHash,restored_sha256:finalHash,restoration_identical:true,accepted,visible_nodes:nodes.length});
   console.log('rendered',baselineId,accepted,'valid CSS pairs');
  }
  console.log('family complete',family,countFamily);
 }
 await write(path.join(data,'manifest.json'),rows);await write(path.join(data,'baselines.json'),baselines);await write(path.join(data,'exclusions.json'),skipped);
 await write(path.join(run,'outputs/render_audit.json'),{browser_version:browser.version(),families:audits,blocked_request_count:blocked.length,network_policy:'all requests aborted; source and CSS supplied inline',pairs:rows.length,unique_baselines:baselines.length,skipped:skipped.length,per_pair_pristine_check:true});
 console.log(JSON.stringify({complete:true,pairs:rows.length,unique_baselines:baselines.length,skipped:skipped.length}));
}finally{await context.close();await browser.close();}
