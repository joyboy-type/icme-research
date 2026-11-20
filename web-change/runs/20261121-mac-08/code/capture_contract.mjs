export function assessPair(a,b){
 const reasons=[];
 if(!a.ready||!b.ready)reasons.push('page_not_ready');
 if(a.launchId===b.launchId)reasons.push('shared_browser_process');
 if(a.screenshot===b.screenshot)reasons.push('reused_png_path');
 if(a.artifactSha!==b.artifactSha)reasons.push('different_source_artifact');
 if(a.resourceSignature!==b.resourceSignature)reasons.push('different_render_resources');
 if(a.networkErrors.length||b.networkErrors.length)reasons.push('required_resource_failure');
 return {valid:reasons.length===0,reasons};
}
import crypto from 'node:crypto';
export function summarizeResources(rows){
 const unique=new Map(),errors=[];
 for(const r of rows){const k=r.type+' '+r.url;if(unique.has(k)&&unique.get(k).sha256!==r.sha256)errors.push({url:r.url,error:'Resource bytes changed within capture'});unique.set(k,r);}
 const resources=[...unique.values()].sort((a,b)=>a.url.localeCompare(b.url));
 return {resources,errors,signature:crypto.createHash('sha256').update(JSON.stringify(resources.map(r=>[r.url,r.type,r.sha256]))).digest('hex')};
}
