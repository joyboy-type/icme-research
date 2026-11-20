"""Evaluate only after pixel-only predictions have been saved."""
from pathlib import Path
import json, math, collections
from relations import admit

P=Path(__file__).resolve().parents[1];O=P/'outputs'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def correctness(candidate,gold,audit):
    if candidate is None:
        return dict(number_correct=False,anchor_correct=False,target_identity_correct=False,
                    target_boundary_clean=False,all_three_correct=False,strict_complete_relation=False,
                    wrong_number=False,wrong_link=False,unresolved_target=True)
    l,t,r,b=gold['box'];x,y=candidate['xy']
    anchor=l-18<=x<=r+18 and t-12<=y<=b+12
    number=candidate['numbers']==gold['numbers']
    statuses=[audit[candidate['source']].get(str(n),dict(identity_correct=False,boundary_clean=False,actual_leading_reference=None)) for n in candidate['numbers']]
    target=number and all(s['identity_correct'] for s in statuses) and all(candidate['targets'])
    clean=target and all(s['boundary_clean'] for s in statuses)
    wrong=any(s['actual_leading_reference'] is not None and s['actual_leading_reference']!=n for n,s in zip(candidate['numbers'],statuses))
    return dict(number_correct=number,anchor_correct=anchor,target_identity_correct=target,
                target_boundary_clean=clean,all_three_correct=number and anchor and target,
                strict_complete_relation=number and anchor and clean,wrong_number=not number,
                wrong_link=wrong,unresolved_target=not all(candidate['targets']))

def main():
    predictions=json.loads((O/'predictions.json').read_text())
    gold=json.loads((O/'gold.json').read_text())
    audit=json.loads((O/'bibliography_identity_audit.json').read_text())
    scored=[];counts={};outside=[]
    for method,rows in predictions.items():
        used=set();entries=[]
        for g in gold:
            l,t,r,b=g['box'];choices=[]
            for i,row in enumerate(rows):
                if i in used or row['page_id']!=g['page_id']:continue
                x,y=row['xy']
                if l-18<=x<=r+18 and t-12<=y<=b+12:
                    choices.append((math.hypot(x-(l+r)/2,y-(t+b)/2),i,row))
            choices.sort(key=lambda z:z[0]);candidate=choices[0][2] if choices else None
            if choices:used.add(choices[0][1])
            result=correctness(candidate,g,audit)
            entry=dict(method=method,case_id=g['case_id'],**result,predicted_numbers=candidate['numbers'] if candidate else [],expected_numbers=g['numbers'],prediction_origin=candidate['origin'] if candidate else None)
            entries.append(entry);scored.append(entry)
        extras=[row for i,row in enumerate(rows) if i not in used]
        # The only extra is the source-verified genuine [16-20] callout excluded by JATS/PDF context alignment.
        assert all(row['source']=='plos-0316191' and row['page_id'].endswith('p02') and row['numbers']==[16,17,18,19,20] for row in extras)
        outside.extend(dict(method=method,reason='true_visible_citation_but_not_in_aligned_JATS_scoring_set',numbers=row['numbers'],page_id=row['page_id']) for row in extras)
        keys=['number_correct','anchor_correct','target_identity_correct','target_boundary_clean','all_three_correct','strict_complete_relation','wrong_number','wrong_link','unresolved_target']
        counts[method]={key:sum(a[key] for a in entries) for key in keys}
        counts[method].update(scored_groups=len(gold),matched_groups=sum(a['anchor_correct'] for a in entries),false_insertions=0,unscored_true_citations=len(extras),total_emitted=len(rows))
    books=json.loads((O/'ocr_bibliography.json').read_text());locals=json.loads((O/'local_candidates.json').read_text());negative=[]
    for i,candidate in enumerate(locals):
        if not candidate['consistent']:continue
        book=books[candidate['source']];replacement=next(int(k) for k in sorted(book,key=int) if int(k) not in candidate['numbers'])
        changed=dict(candidate,numbers=[replacement])
        negative.append(dict(control_id=f'neg-{len(negative)+1:03}',source_candidate_index=i,original_numbers=candidate['numbers'],wrong_but_existing_numbers=[replacement],id_in_ocr_bibliography=True,same_body_anchor=True,simple_id_filter_accepts=admit(changed,{int(k):v for k,v in book.items()},candidate['anchor_before'],candidate['anchor_after'],False),anchor_consistency_filter_accepts=admit(changed,{int(k):v for k,v in book.items()},candidate['anchor_before'],candidate['anchor_after'],True),synthetic=True,excluded_from_natural_metrics=True))
    baseline={a['case_id'] for a in scored if a['method']=='whole' and a['all_three_correct']}
    local={a['case_id'] for a in scored if a['method']=='local_mapping' and a['all_three_correct']}
    advanced={a['case_id'] for a in scored if a['method']=='local_consistency' and a['all_three_correct']}
    summary=dict(scope='Two new PLOS documents; numeric bracket groups on first four physical pages; 9 aligned groups, 21 linked reference targets. Development feasibility only, not superscript or unseen-confirmation evidence.',
        methods=counts,relation_identity_net_gain_over_whole=len(advanced)-len(baseline),relation_identity_net_gain_over_local_mapping=len(advanced)-len(local),new_complete_relations_gained=sorted(advanced-baseline),previous_complete_relations_lost=sorted(baseline-advanced),
        natural_baseline_failures=dict(group_ocr_omission=1,delimiter_misrecognition=2,wrong_body_numbers_observed=0,missing_groups_after_local=2),
        recovered_body_number_but_wrong_target=dict(case_id='case-006',printed_callout=3,whole_ocr='[3)',local_ocr='[3]',ocr_bibliography_key=3,actual_reference_entry=23,correct_reference_entry=3,explanation='A real bibliography OCR error relabels printed23as3; local bracket repair is followed by a wrong link, which both consistency tests accept.'),
        negative_control=dict(cases=len(negative),simple_id_filter_accepts=sum(n['simple_id_filter_accepts'] for n in negative),anchor_filter_accepts=sum(n['anchor_consistency_filter_accepts'] for n in negative),not_mixed_with_natural_metrics=True),
        evaluation_notes=['Target identity and faithful entry boundaries are reported separately; merged entries can identify a leading target but do not count as clean complete relations.','Anchor correctness means the source callout location (all 9 manually reviewed); this pilot does not demonstrate sentence-level span reconstruction.','One real [16-20] callout per method is unscored because strict JATS/PDF context alignment failed; it is not counted as a false insertion.','All emitted predictions on the eight body pages were checked; there were no noncitation insertions, but the scoring denominator is only the aligned subset.','Reference numbers and rid mappings from PDF/JATS were never supplied to inference.','CEX/GROBID and OCR-EDR were not run; no strong-baseline superiority is claimed.'],
        criteria=dict(natural_errors_exist=True,positive_relation_gain_over_local_mapping=len(advanced)>len(local),serious_wrong_links_nonincrease_vs_local_mapping=counts['local_consistency']['wrong_link']<=counts['local_mapping']['wrong_link'],serious_wrong_links_nonincrease_vs_whole=counts['local_consistency']['wrong_link']<=counts['whole']['wrong_link']),
        decision='DO_NOT_INITIATE_FULL_PROJECT; close this frozen candidate version, no threshold search or data expansion',automatic_insertion=False,human_review_only=True,new_data_expansion=False,word_edit_allowed=False)
    save(O/'per_case.json',scored);save(O/'unscored_predictions.json',outside);save(O/'negative_controls.json',negative);save(O/'summary.json',summary)
    save(P/'decision.json',dict(status='feasibility_not_supported',automatic_insertion=False,human_review_only=True,new_data_expansion=False,word_edit_allowed=False,reason='No relation gain beyond ordinary local rereading; an actual reference23-to3OCR error creates a new wrong link; data is small and bracket-only'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
