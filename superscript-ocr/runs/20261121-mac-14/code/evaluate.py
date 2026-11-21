from pathlib import Path
import argparse,json,csv,collections,hashlib,datetime
from score import score_position

def main():
    parser=argparse.ArgumentParser(description='Score only source-verified development positions and reviewed change events.')
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'20261121-mac-13')
    parser.add_argument('--replay',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/replay.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/evaluation.json')
    args=parser.parse_args();p=Path(__file__).resolve().parents[1];assert not args.output.exists()
    replay=json.loads(args.replay.read_text());truth=json.loads((args.source/'outputs/manual_line_review.json').read_text());corrections={r['line_id']:r for r in json.loads((args.source/'outputs/annotation_corrections.json').read_text())};manual=json.loads((p/'outputs/manual_change_review.json').read_text());rows=[]
    for record in replay['records']:
        original=[r for r in record['lines'] if r['origin']=='receiver'];added=[r for r in record['lines'] if r['origin']=='donor']
        for g in truth:
            if g['case_id']!=record['case_id']:continue
            gold=dict(text=corrections[g['line_id']]['source_image_transcription'],box=g['box'])
            metrics=score_position(gold,original,record['lines'],added)
            rows.append(dict(line_id=g['line_id'],case_id=g['case_id'],method=record['method'],role='known_vision_omission' if g['line_id']=='line-0177' else 'intact_control',receiver_engine=record['receiver_engine'],original_body_edits=0,**metrics))
    methods=sorted({r['method'] for r in replay['records']});table=[]
    for method in methods:
        panel=[r for r in rows if r['method']==method];records=[r for r in replay['records'] if r['method']==method];changes=[r for r in manual['manual_reviews'] if r['method']==method]
        if method.startswith('position_neighbor'):
            review=manual['prototype_acceptance_review'];changes=[review] if review['method']==method else []
        additions=sum(len(r['insertions']) for r in records);assert len(changes)==additions
        table.append(dict(method=method,verified_primary_positions=len(panel),correct_at_primary_positions=sum(r['final_correct_at_source'] for r in panel),known_target_available=next(r['final_correct_at_source'] for r in panel if r['role']=='known_vision_omission'),primary_correct_recoveries=sum(r['correct_recoveries'] for r in panel),primary_false_insertions=sum(r['false_insertions'] for r in panel),primary_wrong_positions=sum(r['wrong_position_insertions'] for r in panel),primary_duplicates=sum(r['duplicate_copies'] for r in panel),accepted_change_events=additions,source_reviewed_change_events=len(changes),correct_missing_line_recoveries=sum(r['correct_missing_line_recovery'] for r in changes),false_insertions=sum(r['false_insertion'] for r in changes),duplicate_source_regions=sum(r['duplicate_source_region'] for r in changes),wrong_reading_order_positions=sum(r['wrong_reading_order_position'] for r in changes),original_body_edits=0,algorithm_seconds=sum(r['algorithm_seconds'] for r in records)))
    result=dict(id=p.name,scope='Known development replay, no independent confirmation',primary_position_denominator=4,missing_line_denominator=1,intact_control_denominator=3,primary_documents=1,whole_replay_pages=4,whole_replay_body_line_groups_not_used_as_accuracy_denominator=283,primary_per_position=rows,method_table=table,stronger_single_engine_on_verified_panel='Tesseract:4/4 vs Vision:3/4; not an overall four-page OCR ranking',secondary_change_event_denominator=17,secondary_change_events_across_both_text_only_orientations_not_independent_examples=True,review_coverage='Every accepted change from every method source-reviewed; all unmodified/unflagged page text remains uncertified',metrics_overlap='False insertion includes duplication; wrong reading order overlaps false insertion; original body edits exclude added bad text and are not a whole-output harmlessness measure',stop='Stronger single engine already matches target availability and primary-panel correctness; no expansion or threshold tuning',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(table,indent=2))

if __name__=='__main__':main()
