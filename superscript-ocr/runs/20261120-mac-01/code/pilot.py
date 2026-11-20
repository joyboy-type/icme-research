from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import unicodedata

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parents[1] / 'citation_marker_local_replay_bGc6kNc7'
METHODS = ('full', 'uniform', 'candidate')
cv2.setNumThreads(1)


def tile_scores(image, lines):
    height, width = image.shape
    w, h = width // 4, height // 4
    tiles = [(col * w, row * h, w, h) for row in range(4) for col in range(4)]
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)
    _, _, boxes, centers = cv2.connectedComponentsWithStats(mask)
    line_heights = [line['box'][3] * height for line in lines
                    if len(line['text']) >= 14 and 5 < line['box'][3] * height < height * .04]
    typical = np.median(line_heights) if line_heights else height * .014
    sizes = boxes[:, 3]
    shapes = ((sizes >= max(3, .23 * typical)) & (sizes <= .8 * typical)
              & (boxes[:, 2] >= 2) & (boxes[:, 2] <= sizes * 1.4)
              & (boxes[:, 4] >= 6))
    tiny = centers[shapes]
    votes = np.zeros(len(tiny), dtype=bool)
    for line in lines:
        if len(line['text']) < 14:
            continue
        x, y, bw, bh = np.array(line['box']) * [width, height, width, height]
        if not .6 * typical <= bh <= 1.65 * typical:
            continue
        # A compact mark just above or in the upper part of a detected text line.
        votes |= ((tiny[:, 0] >= x) & (tiny[:, 0] <= x + bw + typical)
                  & (tiny[:, 1] >= y - .50 * bh) & (tiny[:, 1] <= y + .25 * bh))
    xy = tiny[votes]
    score = [int(((xy[:, 0] >= x) & (xy[:, 0] < x + w)
                  & (xy[:, 1] >= y) & (xy[:, 1] < y + h)).sum())
             for x, y, w, h in tiles]
    return tiles, score


def prepare():
    pages = [json.loads(line) for line in
             (PREVIOUS / 'native_ocr/development_pages.jsonl').read_text().splitlines()]
    jobs = []
    for index, page in enumerate(pages):
        image_path = PREVIOUS / 'images' / (page['pageID'] + '.jpg')
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None
        height, width = image.shape
        assert (width, height) == (page['pixelWidth'], page['pixelHeight'])
        tiles, scores = tile_scores(image, page['lines'])
        jobs.append(make_job(page, 'full', image_path, (0, 0, width, height)))
        uniform = (1, 7, 8, 14)
        candidate = sorted(range(len(tiles)), key=lambda k: (-scores[k], k))[:4]
        for method, keys in [('uniform', uniform), ('candidate', candidate)]:
            for key in keys:
                x, y, w, h = tiles[key]
                file = ROOT / 'crops' / f'{index:02}_{method}_{key}.png'
                assert cv2.imwrite(str(file), image[y:y+h, x:x+w])
                jobs.append(make_job(page, method, file, tiles[key], scores[key]))
        assert len(uniform) == len(candidate) == 4
    return jobs


def make_job(page, method, path, box, score=None):
    return dict(page_id=page['pageID'], method=method, path=str(path),
                box=list(box), proposal_score=score)


def normalize(number):
    return unicodedata.normalize('NFKC', str(number)).translate(str.maketrans('–−—', '---'))


def assess(jobs, predictions):
    # Ground truth is loaded only after the inference outputs are saved.
    labels = json.loads((PREVIOUS / 'sample.json').read_text())
    labels = [row for row in labels if row['phase'] == 'discovery']
    assert len(labels) == 40
    index = {}
    for job, result in zip(jobs, predictions, strict=True):
        assert job['path'] == result['path']
        y = index.setdefault((job['page_id'], job['method']), [])
        x0, y0, w, h = job['box']
        digits = [(normalize(m['text']), x0 + m['x'] * w, y0 + m['y'] * h)
                  for m in result['matches']]
        y.append((job['box'], digits))
    rows = []
    for label in labels:
        page = label['page_id']
        xs, ys = label['span']['poly'][::2], label['span']['poly'][1::2]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        assert index[(page, 'full')][0][0][2:] == [label['page']['width'], label['page']['height']]
        center_x, center_y = (left+right)/2, (top+bottom)/2
        margin_x, margin_y = max(5, (right-left)*.2), max(5, (bottom-top)*.2)
        row = {'id': label['marker_id'], 'page_id': page, 'identity': label['identity']}
        for method in METHODS:
            values = index[(page, method)]
            row[method+'_covered'] = any(x <= center_x < x+w and y <= center_y < y+h
                                         for (x,y,w,h), _ in values)
            row[method+'_hit'] = any(token == normalize(label['identity'])
                                    and left-margin_x <= x <= right+margin_x
                                    and top-margin_y <= y <= bottom+margin_y
                                    for _, digits in values for token,x,y in digits)
        rows.append(row)
    return rows


def main():
    (ROOT / 'crops').mkdir(exist_ok=True)
    jobs = prepare()
    (ROOT / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    with (ROOT / 'tokens.jsonl').open('w') as handle:
        subprocess.run([str(ROOT/'run'), *(job['path'] for job in jobs)], stdout=handle, check=True)
    predictions = [json.loads(line) for line in (ROOT/'tokens.jsonl').read_text().splitlines()]
    rows = assess(jobs, predictions)
    with (ROOT/'per_marker.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'pages': 12, 'markers': len(rows), 'method': {},
               'scope': 'Exploratory discovery-only token localization. No text-anchor verification or insertion scoring.'}
    for method in METHODS:
        subset = [job for job in jobs if job['method'] == method]
        summary['method'][method] = {
            'calls': len(subset),
            'total_input_pixels': sum(job['box'][2]*job['box'][3] for job in subset),
            'markers_in_tile': sum(row[method+'_covered'] for row in rows),
            'digit_location_hits': sum(row[method+'_hit'] for row in rows),
            'additional_hits_over_full': sum(not row['full_hit'] and row[method+'_hit'] for row in rows),
        }
    assert summary['method']['uniform']['calls'] == summary['method']['candidate']['calls']
    assert summary['method']['uniform']['total_input_pixels'] == summary['method']['candidate']['total_input_pixels']
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
