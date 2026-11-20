import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {assessPair,summarizeResources} from './capture_contract.mjs';
const require=createRequire(process.env.WEB_PROBE_NODE_MODULES?path.join(process.env.WEB_PROBE_NODE_MODULES,'package.json'):import.meta.url),{chromium}=require('playwright');
const run=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),project=path.resolve(run,'../..'),data=path.join(project,'data/real-version-replay-02'),out=path.join(run,'outputs');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex'),read=async p=>JSON.parse(await fs.readFile(p,'utf8')),dump=async(p,x)=>fs.writeFile(p,JSON.stringify(x,null,2)+'\n');
const cfg=await read(path.join(run,'protocol.json'));await fs.mkdir(path.join(data,'captures'),{recursive:true});await fs.mkdir(path.join(data,'har'),{recursive:true});
const paths=['protocol.json','source_specs.json','code/replay.mjs','code/capture_contract.mjs'];await dump(path.join(run,'capture_lock.json'),{at_utc:new Date().toISOString(),hashes:Object.fromEntries(await Promise.all(paths.map(async p=>[p,hash(await fs.readFile(path.join(run,p)))])))});
const server=http.createServer(async(req,res)=>{
 try{const relative=decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/^\//,'');const target=path.resolve(data,relative);if(!target.startsWith(path.join(data,'mdn')+path.sep))throw new Error('Not a fixture');const bytes=await fs.readFile(target);const ext=path.extname(target);res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.jpg':'image/jpeg','.svg':'image/svg+xml'})[ext]||'application/octet-stream');res.setHeader('Cache-Control','no-store');res.end(bytes);}catch{res.statusCode=404;res.end('Not found');}
});await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;
let sequence=0;const captures=[],controls=[],seeds=[],failures=[];
function auditNetwork(page){
 const resources=[],errors=[],pending=[];let bytes=0;
 const needed=r=>['document','script','stylesheet','font','image'].includes(r.resourceType())&&!new URL(r.url()).pathname.endsWith('/favicon.ico');
 page.on('requestfailed',r=>{if(needed(r))errors.push({url:r.url(),error:r.failure()?.errorText||'failed'});});
 page.on('response',r=>{if(!needed(r.request()))return;pending.push((async()=>{try{if(r.status()>=400){errors.push({url:r.url(),status:r.status()});return;}const b=await r.body();bytes+=b.length;resources.push({url:r.url(),type:r.request().resourceType(),status:r.status(),bytes:b.length,sha256:hash(b)});}catch(e){errors.push({url:r.url(),error:String(e)});}})());});
 return {resources,errors,async finish(){await Promise.allSettled(pending);return {resources,errors,bytes};}};
}
async function ready(page,ionic){
 await page.evaluate(()=>document.fonts.ready);
 await page.waitForFunction(()=>[...document.images].every(x=>x.complete&&x.naturalWidth>0),{},{timeout:15000});
 if(ionic)await page.waitForFunction(()=>document.querySelector('ion-app')&&[...document.querySelectorAll('ion-app,ion-content,ion-toolbar,ion-checkbox,ion-button')].every(e=>e.classList.contains('hydrated')),{},{timeout:15000});
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));await page.waitForTimeout(500);
 return page.evaluate(()=>({title:document.title,body_text_length:document.body?.innerText.length||0,images:[...document.images].map(i=>({loaded:i.complete&&i.naturalWidth>0,width:i.naturalWidth,height:i.naturalHeight})),scroll_height:document.documentElement.scrollHeight,fonts:document.fonts.status,ionic_mode:document.documentElement.getAttribute('mode')}));
}
function contextOptions(){return {viewport:{width:cfg.replay.viewport[0],height:cfg.replay.viewport[1]},deviceScaleFactor:1,locale:cfg.replay.locale,timezoneId:cfg.replay.timezone,colorScheme:cfg.replay.color_scheme,serviceWorkers:'block'};}
async function capture(spec,rep){
 const launchId=`launch_${++sequence}`,browser=await chromium.launch({channel:'chrome',headless:true}),context=await browser.newContext(contextOptions());
 try{
  if(spec.har)await context.routeFromHAR(spec.har,{notFound:'abort'});else await context.route('**/*',r=>new URL(r.request().url()).hostname==='127.0.0.1'?r.continue():r.abort());
  const page=await context.newPage(),network=auditNetwork(page),started=new Date().toISOString();await page.goto(spec.url,{waitUntil:'load',timeout:45000});const state=await ready(page,Boolean(spec.har));const screenshot=path.join(data,'captures',`${spec.id}_${rep}.png`);await page.screenshot({path:screenshot,animations:'disabled',caret:'hide'});const n=await network.finish();
  const summary=summarizeResources(n.resources),resources=summary.resources,signature=summary.signature;n.errors.push(...summary.errors);
  const row={id:spec.id,rep,group:spec.group,source_kind:spec.kind,launchId,started_at:started,finished_at:new Date().toISOString(),browser_version:browser.version(),url:spec.url,artifactSha:spec.artifactSha,resourceSignature:signature,resource_count:resources.length,networkErrors:n.errors,ready:state.body_text_length>20&&state.fonts==='loaded',state,screenshot:path.relative(project,screenshot),png_sha256:hash(await fs.readFile(screenshot)),resources};captures.push(row);console.log('CAPTURED',spec.id,rep,'launch',launchId,'resources',resources.length,'errors',n.errors.length);return row;
 }finally{await context.close();await browser.close();}
}
async function seed(url,id){
 const inventory=await read(path.join(out,'seed_inventory.json')).catch(()=>[]),cached=inventory.find(x=>x.id===id&&x.url===url);
 if(cached){const har=path.join(project,cached.har);if(hash(await fs.readFile(har))!==cached.artifact_sha256)throw new Error('Cached archive changed');seeds.push({...cached,reused_after_logging_fix:true});console.log('REUSED ARCHIVE',id);return {id,url,har,artifactSha:cached.artifact_sha256,group:'ionic-team/ionic-framework',kind:'archived_live_project_artifact'};}
 const browser=await chromium.launch({channel:'chrome',headless:true}),har=path.join(data,'har',id+'.har');const context=await browser.newContext({...contextOptions(),recordHar:{path:har,content:'embed',mode:'full'}});let count=0;
 try{
  await context.route('**/*',r=>{count++;if(count>cfg.replay.ionic_archive.network_requests_per_page_cap)return r.abort();return r.continue();});const page=await context.newPage(),network=auditNetwork(page);await page.goto(url,{waitUntil:'load',timeout:45000});const state=await ready(page,true);const n=await network.finish();
  if(count>cfg.replay.ionic_archive.network_requests_per_page_cap||n.bytes>cfg.replay.ionic_archive.total_response_bytes_per_page_cap)throw new Error('Acquisition budget exceeded');
  if(n.errors.length)throw new Error('Required resource acquisition errors: '+JSON.stringify(n.errors));
  await context.close();const bytes=await fs.readFile(har);const row={id,url,artifact_sha256:hash(bytes),har:path.relative(project,har),response_bytes:n.bytes,request_count:count,ready_state:state,commit_identity:'Not exposed by live alias; artifact bytes only, not claimed to reproduce2024 commits.'};seeds.push(row);console.log('ARCHIVED',id,count,'requests',n.bytes,'bytes');return {id,url,har,artifactSha:row.artifact_sha256,group:'ionic-team/ionic-framework',kind:'archived_live_project_artifact'};
 }finally{await context.close().catch(()=>{});await browser.close();}
}
try{
 for(const version of ['before','after']){
  const file=path.join(data,'mdn',version,'media-objects--download.html');const pieces=[await fs.readFile(file),await fs.readFile(path.join(data,'mdn',version,'balloon-sq2.jpg')),await fs.readFile(path.join(data,'mdn',version,'sharp-account_box-24px.svg'))];const spec={id:'mdn_'+version,url:`http://127.0.0.1:${port}/mdn/${version}/media-objects--download.html`,artifactSha:hash(Buffer.concat(pieces)),group:'mdn/css-examples',kind:'pinned_documentation_example_version'};
  try{const a=await capture(spec,1),b=await capture(spec,2);controls.push({id:spec.id,group:spec.group,a:a.screenshot,b:b.screenshot,...assessPair(a,b)});}catch(e){failures.push({id:spec.id,stage:'mdn_capture',error:String(e)});console.log('FAILED',spec.id,String(e));}
 }
 for(let i=0;i<cfg.replay.ionic_archive.urls.length;i++){
  const id='ionic_'+i;try{const spec=await seed(cfg.replay.ionic_archive.urls[i],id);const a=await capture(spec,1),b=await capture(spec,2);controls.push({id,group:spec.group,a:a.screenshot,b:b.screenshot,...assessPair(a,b)});}catch(e){failures.push({id,stage:'ionic_archive_or_capture',error:String(e)});console.log('FAILED',id,String(e));}
 }
 await dump(path.join(out,'captures.json'),captures);await dump(path.join(out,'repeat_controls.json'),controls);await dump(path.join(out,'archive_seeds.json'),seeds);await dump(path.join(out,'capture_failures.json'),failures);
 await dump(path.join(out,'capture_summary.json'),{counted_capture_images:captures.length,independent_processes_launched:sequence,control_pairs:controls.length,valid_same_artifact_repeat_pairs:controls.filter(x=>x.valid).length,failures,groups:[...new Set(controls.filter(x=>x.valid).map(x=>x.group))],synthetic_css_or_dom_changes:false,seed_acquisitions_not_counted_as_controls:true});console.log('REPLAY COMPLETE',captures.length,'captures',controls.filter(x=>x.valid).length,'valid controls','failures',failures.length);
}finally{await new Promise(r=>server.close(r));}
