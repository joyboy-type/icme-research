"""Cached-OCR, append-only line merging; no images or ground truth are inputs."""
import difflib,re,unicodedata

def merge(receiver,donor,width,height,position_check=True):
    norm=lambda s:''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
    sim=lambda a,b:difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
    cy=lambda b:(b[1]+b[3])/2
    def overlap(a,b):
        return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))/max(1,(a[2]-a[0])*(a[3]-a[1]))
    def same_column(a,b):
        fraction=max(0,min(a[2],b[2])-max(a[0],b[0]))/max(1,min(a[2]-a[0],b[2]-b[0]))
        return fraction>=.5 or abs(a[0]-b[0])<=40
    rn=[norm(r['text']) for r in receiver];dn=[norm(d['text']) for d in donor];joined=''.join(rn)
    matches=[]
    for n in dn:
        scores=[sim(n,r) if n and r else 0 for r in rn]
        matches.append([(i,s) for i,s in enumerate(scores) if s>=.90])
    decisions=[];accepted=[];seen=set();slots={}
    for j,d in enumerate(donor):
        n=dn[j];box=d['box'];record={'donor_index':j,'text':d['text'],'box':box,'accepted':False,'reason':'','novel_candidate':False}
        if len(n)<12 or len(re.findall(r'[a-z]+|[0-9]+',unicodedata.normalize('NFKC',d['text']).casefold()))<4:
            record['reason']='short_line_excluded'
        elif n in seen:
            record['reason']='donor_repeat'
        elif n in joined or matches[j]:
            record['reason']='text_already_represented'
        else:
            record['novel_candidate']=True;before=None;after=None;insert_at=None
            if position_check:
                if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):record['reason']='invalid_box'
                elif any(overlap(box,r['box'])>=.10 for r in receiver):record['reason']='receiver_row_occupied'
                else:
                    anchors=[]
                    for k,anchor in enumerate(donor):
                        ab=anchor['box'];distance=cy(ab)-cy(box)
                        if k==j or abs(distance)>180 or not same_column(box,ab):continue
                        for i,score in matches[k]:
                            rb=receiver[i]['box']
                            if abs(ab[0]-rb[0])<=20 and abs(cy(ab)-cy(rb))<=12:
                                anchors.append({'donor_index':k,'receiver_index':i,'similarity':score,'distance':distance,'receiver_cy':cy(rb)})
                    upper=[a for a in anchors if a['distance']<0 and a['receiver_cy']<cy(box)]
                    lower=[a for a in anchors if a['distance']>0 and a['receiver_cy']>cy(box)]
                    if upper:before=min(upper,key=lambda a:(abs(a['distance']),-a['similarity'],a['receiver_index']))
                    if lower:after=min(lower,key=lambda a:(abs(a['distance']),-a['similarity'],a['receiver_index']))
                    if before is None or after is None:record['reason']='no_two_sided_position_anchors'
                    elif before['receiver_index']>=after['receiver_index']:record['reason']='anchor_reading_order_conflict'
                    else:insert_at=before['receiver_index']+1
            else:
                for k in range(j-1,-1,-1):
                    if matches[k]:
                        i,score=max(matches[k],key=lambda x:(x[1],-x[0]));before={'donor_index':k,'receiver_index':i,'similarity':score};break
                if before is not None:insert_at=before['receiver_index']+1
                else:
                    for k in range(j+1,len(donor)):
                        if matches[k]:
                            i,score=max(matches[k],key=lambda x:(x[1],-x[0]));after={'donor_index':k,'receiver_index':i,'similarity':score};break
                    insert_at=after['receiver_index'] if after is not None else len(receiver)
            record.update(upper_anchor=before,lower_anchor=after,insertion_slot=insert_at)
            if insert_at is not None:
                record.update(accepted=True,reason='accepted_position_neighbors' if position_check else 'accepted_text_only')
                added=dict(d,origin='donor',donor_index=j,insertion_slot=insert_at)
                slots.setdefault(insert_at,[]).append(added);accepted.append(dict(record));seen.add(n)
        decisions.append(record)
    output=[]
    for i in range(len(receiver)+1):
        output.extend(slots.get(i,[]))
        if i<len(receiver):output.append(dict(receiver[i],origin='receiver',receiver_index=i))
    return {'lines':output,'insertions':accepted,'decisions':decisions}
