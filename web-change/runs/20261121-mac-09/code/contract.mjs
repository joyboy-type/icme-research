import {assessPair} from '../../20261121-mac-08/code/capture_contract.mjs';
export function validateRepeat(a,b){
 const base=assessPair(a,b),reasons=[...base.reasons];
 if(a.stateSignature!==b.stateSignature)reasons.push('different_control_state_or_font');
 if(a.targetMediaCount||b.targetMediaCount)reasons.push('media_inside_target');
 if(a.missingRequired.length||b.missingRequired.length)reasons.push('required_source_asset_missing');
 return {valid:reasons.length===0,reasons};
}
