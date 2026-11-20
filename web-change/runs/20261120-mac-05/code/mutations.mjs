export function plan(nodes,label,level){
 if(!Number.isInteger(level)||level<0||level>2)throw new Error('Invalid level');
 const number=x=>Number.parseFloat(x)||0;
 const texts=nodes.filter(n=>n.leaf&&n.text.trim().length>=8&&n.text.length<=250&&n.width>=35&&n.height>=10&&n.height<=140)
   .sort((a,b)=>(a.tag==='P'?0:1)-(b.tag==='P'?0:1)||a.y-b.y||Number(a.id)-Number(b.id));
 const filled=nodes.filter(n=>n.width>=40&&n.height>=20&&n.width*n.height<250000&&n.backgroundColor&&!n.backgroundColor.startsWith('rgba(0, 0, 0, 0)')&&n.backgroundColor!=='transparent')
   .sort((a,b)=>(a.tag==='BUTTON'?0:1)-(b.tag==='BUTTON'?0:1)||a.width*a.height-b.width*b.height);
 const bordered=nodes.filter(n=>number(n.borderTopWidth)>0&&n.borderTopStyle!=='none'&&n.width>=40&&n.height>=20).sort((a,b)=>a.width*a.height-b.width*b.height);
 let target=texts[0],property,value;
 if(label==='text'){
   target=texts.find(n=>n.text.trim().split(/\s+/).length>=level+2);if(!target)return null;
   const words=target.text.trim().split(/\s+/),i=level;[words[i],words[i+1]]=[words[i+1],words[i]];
   return {target:target.id,property:'textContent',value:words.join(' '),old:target.text};
 }
 if(label==='border'){target=bordered[0];property='border-top-width';if(target)value=`${number(target.borderTopWidth)+[1,2,3][level]}px`;}
 else if(label==='rounded'){target=filled[0]||bordered[0];property='border-radius';if(target)value=`${number(target.borderTopLeftRadius)+[2,4,8][level]}px`;}
 else if(label==='gradient'){
   target=filled[0];property='background-image';if(target){const rgb=(target.backgroundColor.match(/[\d.]+/g)||[]).slice(0,3).map(Number);const d=[8,16,24][level];const other=rgb.map(v=>Math.max(0,Math.min(255,v+(v>127?-d:d))));value=`linear-gradient(90deg, rgb(${rgb.join(',')}), rgb(${other.join(',')}))`;}
 }
 else if(label==='justify'){
   target=nodes.find(n=>['flex','inline-flex'].includes(n.display)&&n.children>=2&&n.width>=100&&n.height<=200);
   property='justify-content';if(target)value=['center','flex-end','space-between','flex-start'].filter(v=>v!==target.justifyContent)[level];
 }
 else if(target){
   if(label==='font_size'){property='font-size';value=`${number(target.fontSize)+[1,2,3][level]}px`;}
   else if(label==='font_weight'){property='font-weight';const w=number(target.fontWeight)||400;value=String(w+(w>=600?-1:1)*[100,200,300][level]);}
   else if(label==='letter_spacing'){property='letter-spacing';value=`${number(target.letterSpacing)+[.5,1,2][level]}px`;}
   else if(label==='line_height'){property='line-height';value=`${(number(target.lineHeight)||number(target.fontSize)*1.2)+[1,2,4][level]}px`;}
   else if(label==='spacing'){property='padding-left';value=`${number(target.paddingLeft)+[1,2,4][level]}px`;}
   else if(label==='position'){property='transform';value=(target.transform&&target.transform!=='none'?target.transform+' ':'')+`translateX(${[1,2,4][level]}px)`;}
   else if(label==='opacity'){property='opacity';value=String((number(target.opacity)||1)*[.9,.8,.65][level]);}
   else if(label==='color'){property='color';const rgb=(target.color.match(/[\d.]+/g)||[]).slice(0,3).map(Number);const d=[8,16,24][level];value=`rgb(${rgb.map(v=>Math.max(0,Math.min(255,v+(v>127?-d:d)))).join(',')})`;}
   else throw new Error('Unknown label '+label);
 }
 if(!target||!value)return null;
 return {target:target.id,property,value,old:target[property]??null};
}
