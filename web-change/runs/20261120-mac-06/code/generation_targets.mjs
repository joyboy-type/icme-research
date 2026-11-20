import {plan} from '../../20261120-mac-05/code/mutations.mjs';
const number=x=>Number.parseFloat(x)||0;
export function targets(nodes,label){
 if(label==='position')return nodes.filter(n=>n.display!=='inline'&&n.width>=40&&n.height>=20&&n.width*n.height<250000&&((n.backgroundColor&&n.backgroundColor!=='transparent'&&!n.backgroundColor.startsWith('rgba(0, 0, 0, 0)'))||(number(n.borderTopWidth)>0&&n.borderTopStyle!=='none'))).sort((a,b)=>a.width*a.height-b.width*b.height||Number(a.id)-Number(b.id)).slice(0,3);
 if(label==='justify')return nodes.filter(n=>['flex','inline-flex'].includes(n.display)&&n.visibleChildren>=2&&n.flexDirection==='row'&&n.flexWrap==='nowrap'&&n.freeWidth>=8&&n.width>=100&&n.height<=200).sort((a,b)=>Number(a.id)-Number(b.id)).slice(0,3);
 throw new Error(label);
}
export function mutation(target,label,level){
 if(![0,1,2].includes(level))throw new Error('Invalid level');
 if(label==='justify')return plan([target],label,level);
 if(label==='position')return {target:target.id,property:'transform',value:(target.transform&&target.transform!=='none'?target.transform+' ':'')+`translateX(${[1,2,4][level]}px)`};
 throw new Error(label);
}
