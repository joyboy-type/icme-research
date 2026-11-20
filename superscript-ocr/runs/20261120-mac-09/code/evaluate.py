"""Evaluate saved frozen-gate decisions; reviewed truth never enters gate inference."""
from pathlib import Path
import json,math,hashlib
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parents[1];R=P.parent;O=P/'outputs'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def identity(a):return a['page_id'],a['method'],a['job_index'],a['text']
def marker(a):return a['page_id'],a['line_id'],a['offset'],a['text']

def main():
    decisions=json.loads((O/'decisions.json').read_text());prior=json.loads((R/'20261120-mac-08/outputs/insertion_audit.json').read_text());truth={identity(a):a for a in prior}
    audit=[]
    for d in decisions:
        t=truth[identity(d)];assert d['xy']==t['xy'] and d['left']==t['left'] and d['right']==t['right']
        audit.append(dict(d,reviewed_truth=t['pixel_review'],marker_scope=t['marker_scope'],prior_audit_index=t['index']))
    save(O/'decision_audit.json',audit)
    counts={}
    for method in ['baseline','isolate']:
        qs=[a for a in audit if a['method']==method];counts[method]={}
        for stage,subset in [('before',qs),('after',[a for a in qs if a['accepted']]),('rejected',[a for a in qs if not a['accepted']])]:
            counts[method][stage]=dict(total=len(subset),correct=sum(a['reviewed_truth']=='correct_number_and_word_anchor' for a in subset),wrong_number=sum(a['reviewed_truth']=='wrong_number' for a in subset),false_insertion=sum(a['reviewed_truth']=='false_insertion' for a in subset),correct_figure_footnotes=sum(a['reviewed_truth']=='correct_number_and_word_anchor' and a['marker_scope']=='figure_title_footnote_outside_selected90' for a in subset))
            counts[method][stage]['serious_errors']=counts[method][stage]['wrong_number']+counts[method][stage]['false_insertion']
    raw_baseline=counts['baseline']['before'];raw_new=counts['isolate']['before'];new=counts['isolate']['after']
    assert (raw_baseline['correct'],raw_baseline['serious_errors'])==(10,7)
    assert (raw_new['correct'],raw_new['serious_errors'])==(16,9)
    base_correct={marker(a) for a in audit if a['method']=='baseline' and a['reviewed_truth']=='correct_number_and_word_anchor'}
    new_correct={marker(a) for a in audit if a['method']=='isolate' and a['accepted'] and a['reviewed_truth']=='correct_number_and_word_anchor'}
    frozen_retention_required=math.ceil(raw_new['correct']*.8)
    criteria=dict(correct_net_gain=new['correct']>raw_baseline['correct'],serious_errors_nonincrease=new['serious_errors']<=raw_baseline['serious_errors'],frozen_80percent_retention=new['correct']>=frozen_retention_required)
    passed=all(criteria.values())
    summary=dict(scope='Development-only fixed-gate audit on already reviewed run08 candidates; all old17hard cases remain diagnostic. Includes all proposed insertions, with figure footnotes separate from90body-reference denominator.',
                 methods=counts,primary_reference='Ungated run08 baseline, not the weakened gated baseline',
                 primary_comparison=dict(correct_before=raw_baseline['correct'],correct_after=new['correct'],net_correct=new['correct']-raw_baseline['correct'],new_correct_gained=len(new_correct-base_correct),previous_correct_lost=len(base_correct-new_correct),serious_before=raw_baseline['serious_errors'],serious_after=new['serious_errors']),
                 retention=dict(correct_retained=new['correct'],correct_available=raw_new['correct'],fraction=new['correct']/raw_new['correct'],required_count=frozen_retention_required,required_fraction=.8),
                 secondary_gated_baseline_comparison=dict(baseline_correct=counts['baseline']['after']['correct'],isolate_correct=new['correct'],difference=new['correct']-counts['baseline']['after']['correct'],warning='This secondary+2comparison cannot replace the primary-5comparison against the original baseline.'),
                 criteria=criteria,development_passed=passed,automatic_insertion=False,decision='freeze_then_confirm_only_if_unseen_authorized_samples_exist' if passed else 'SUSPEND_AUTOMATIC_INSERTION; retain review candidates only; close this development line without expanding data or tuning thresholds',
                 review_policy='Every candidate remains auto_insert=false; prior repaired transcripts are archived diagnostic outputs, not authorized production edits.',
                 new_confirmation_run=False,threshold_search=False,limitations=['Zero observed accepted errors is not a population error-rate guarantee.','Gate uses word-level anchors and does not repair original punctuation or perform bibliography-graph checks.','Nine new local context OCR calls supplement exact cached evidence; this is not an equal-new-cost benchmark.'])
    save(O/'summary.json',summary)
    decision=dict(status='automatic_insertion_suspended' if not passed else 'development_passed_pending_confirmation',run_id=P.name,reason='Correct net gain and frozen80%retention requirements failed' if not passed else 'all preregistered development criteria passed',automatic_insertion=False,human_review_only=True,new_data_expansion=False,word_edit_allowed=False,thresholds_unchanged=True)
    save(P/'decision.json',decision)
    # Keep only the proposed method's queue; baseline controls remain in decision_audit.
    save(O/'review_candidates.json',[dict(a,review_required=True,auto_insert=False) for a in audit if a['method']=='isolate'])
    meta=json.loads((P/'run.json').read_text());meta.update(status='complete',development_passed=passed,automatic_insertion=False,decision=summary['decision'],summary='outputs/summary.json',review_queue='outputs/review_candidates.json');save(P/'run.json',meta)
    # Render source pixels only for the few candidates the unchanged gate admitted.
    accepted=[a for a in audit if a['accepted']];pages={p['page_id']:p for p in json.loads((R/'20261120-mac-08/outputs/pages.json').read_text())};font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    for start in range(0,len(accepted),6):
        sheet=Image.new('RGB',(1200,210*min(6,len(accepted)-start)),'#eee');d=ImageDraw.Draw(sheet)
        for row,a in enumerate(accepted[start:start+6]):
            pg=pages[a['page_id']];im=Image.open(pg['path']).convert('RGB');x,y=a['xy'];h=pg['lines'][a['line_id']]['box'][3];crop=im.crop((max(0,int(x-180)),max(0,int(y-h*.8)),min(im.width,int(x+150)),min(im.height,int(y+h*1.6))));crop=crop.resize((crop.width*2,crop.height*2));crop.thumbnail((1000,145));sheet.paste(crop,(5,row*210+53));d.text((5,row*210+4),f"case#{a['index']} {a['method']} -> {a['text']} | {a['reviewed_truth']}",font=font,fill='black');d.text((5,row*210+27),(a['left'][-65:]+' <'+a['text']+'> '+a['right'][:45])[:125],font=font,fill='black')
        sheet.save(O/f'accepted_review_{start//6:02}.png')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
