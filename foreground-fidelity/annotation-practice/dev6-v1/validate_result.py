#!/usr/bin/env python3
"""Validate AI diagnostic records; this never establishes annotation correctness."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def main(path):
    data = json.loads(Path(path).read_text())
    index = json.loads((ROOT / 'image_index.json').read_text())
    assert data['schema'] == 'ai-source-box-v1', 'Wrong schema'
    assert data['phase'] == 'development_practice', 'Development practice only'
    assert data['annotation_origin'] == 'AI_assisted_NOT_independent_human', 'Must disclose AI origin'
    assert isinstance(data['agent_id'], str) and data['agent_id'].strip(), 'Missing agent ID'
    assert isinstance(data['reviewed_other_agents_results'], bool), 'Missing exposure disclosure'
    assert data['image_index'] == index, 'Different image set'
    records = data['annotations']
    assert len(records) == 6 and [x['image_id'] for x in records] == [x['image_id'] for x in index], 'Missing or repeated image'
    for item in records:
        status, box = item['status'], item['box_xyxy']
        assert status in ('clear', 'no_clear', 'ambiguous'), 'All six must be reviewed; do not fabricate if unable to view'
        assert type(item['truncated']) is bool
        assert isinstance(item['note'], str)
        assert isinstance(item['reason'], str) and item['reason'].strip(), 'Give a short visible-image reason'
        if box is not None:
            assert isinstance(box, list) and len(box) == 4
            assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in box), 'Invalid normalized coordinates'
            assert box[0] < box[2] and box[1] < box[3], 'Empty or reversed box'
        if status == 'clear':
            assert box is not None, 'Clear subject needs a box'
        elif status == 'no_clear':
            assert box is None and item['truncated'] is False, 'No clear subject must not invent a box'
        else:
            assert item['note'].strip(), 'Ambiguity needs explanation'
    print('PASS: 6 AI diagnostic records are structurally valid; not human labels or a correctness verdict.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 validate_result.py results/agent_id.json')
    main(sys.argv[1])
