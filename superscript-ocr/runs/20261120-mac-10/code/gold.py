"""Evaluation-only JATS relationships. Never imported by OCR inference."""
import re
import xml.etree.ElementTree as ET
from relations import citations


def jats_relations(xml):
    root = ET.fromstring(xml)
    refs = {r.get('id'): r for r in root.findall('.//ref-list/ref')}
    labels = {}
    for rid, ref in refs.items():
        label = ''.join(ref.find('label').itertext()).strip().strip('.()[]') if ref.find('label') is not None else ''
        if label.isdigit():
            labels[int(label)] = rid
    rows, rejected = [], []
    for pi, paragraph in enumerate(root.findall('.//body//p')):
        chunks, spans = [], []
        def walk(node):
            if node.text:
                chunks.append(node.text)
            for child in node:
                start = sum(map(len, chunks))
                walk(child)
                end = sum(map(len, chunks))
                if child.tag == 'xref' and child.get('ref-type') == 'bibr':
                    spans.append(dict(start=start,end=end,rids=child.get('rid','').split()))
                if child.tail:
                    chunks.append(child.tail)
        walk(paragraph)
        text = ''.join(chunks)
        handled = set()
        groups = citations(text)
        # Single unbracketed callouts remain eligible for data discovery only.
        for si, span in enumerate(spans):
            if not any(g['start'] <= span['start'] and span['end'] <= g['end'] for g in groups):
                visible = text[span['start']:span['end']].strip()
                if visible.isdigit():
                    groups.append(dict(numbers=[int(visible)],start=span['start'],end=span['end'],left=text[max(0,span['start']-90):span['start']],right=text[span['end']:span['end']+90],unbracketed=True))
        for group in sorted(groups,key=lambda g:g['start']):
            inside = [(si,s) for si,s in enumerate(spans) if group['start'] <= s['start'] and s['end'] <= group['end']]
            if not inside:
                continue
            if any(si in handled for si,_ in inside):
                continue
            handled.update(si for si,_ in inside)
            explicit = [rid for _,s in inside for rid in s['rids']]
            if any(rid not in refs for rid in explicit):
                rejected.append(dict(paragraph=pi,reason='unresolved_rid'));continue
            if not all(n in labels for n in group['numbers']):
                rejected.append(dict(paragraph=pi,reason='missing_numeric_reference_label'));continue
            expected = [labels[n] for n in group['numbers']]
            if not set(explicit).issubset(expected):
                rejected.append(dict(paragraph=pi,reason='printed_number_rid_disagreement'));continue
            rows.append(dict(paragraph=pi,**group,rids=expected,visible=text[group['start']:group['end']]))
    return rows, rejected
