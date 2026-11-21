"""Scoring only for an explicitly source-verified line, never used for routing."""
import unicodedata

def score_position(gold,original,final,insertions):
    norm=lambda s:''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
    def at_source(row):
        a=gold['box'];b=row['box'];intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return intersection/max(1,(a[2]-a[0])*(a[3]-a[1]))>=.10
    expected=norm(gold['text']);correct=lambda r:norm(r['text'])==expected
    before_present=any(correct(r) and at_source(r) for r in original)
    copies=sum(correct(r) and at_source(r) for r in final)
    box=gold['box'];cy=lambda b:(b[1]+b[3])/2
    column=lambda b:max(0,min(box[2],b[2])-max(box[0],b[0]))/max(1,min(box[2]-box[0],b[2]-b[0]))>=.5 or abs(box[0]-b[0])<=40
    upper=[(i,r) for i,r in enumerate(original) if column(r['box']) and r['box'][3]<box[1]]
    lower=[(i,r) for i,r in enumerate(original) if column(r['box']) and r['box'][1]>box[3]]
    lo=max(upper,key=lambda x:cy(x[1]['box']))[0]+1 if upper else 0
    hi=min(lower,key=lambda x:cy(x[1]['box']))[0] if lower else len(original)
    def correct_order(r):return lo<=r.get('insertion_slot',-1)<=hi
    relevant=[r for r in insertions if correct(r) or at_source(r)]
    recovered=sum(correct(r) and at_source(r) and correct_order(r) for r in relevant) if not before_present else 0
    return dict(original_correct_at_source=before_present,final_correct_at_source=copies>0,correct_copies_at_source=copies,correct_recoveries=min(1,recovered),false_insertions=sum(at_source(r) and not correct(r) for r in relevant),wrong_position_insertions=sum(correct(r) and (not at_source(r) or not correct_order(r)) for r in relevant),duplicate_copies=max(0,copies-1),verified_reading_order_slot_interval=[lo,hi])
