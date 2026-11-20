#!/usr/bin/env python3
"""Fetch only six development originals and require pinned revision + exact hashes."""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent

def get(url, limit):
    request = urllib.request.Request(url, headers={'User-Agent': 'foreground-fidelity-dev6-practice/1'})
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError('Response exceeds small-input limit')
        return body, response.headers.get('x-revision')

def main():
    index = json.loads((ROOT / 'image_index.json').read_text())
    locators = json.loads((ROOT / 'source_fetch.json').read_text())
    assert locators['dataset'] == 'GEditBench-v2/VCReward-Bench'
    assert locators['revision'] == '4dfccf0bb4a4448aed140e8ff8ccb412e0f9b2d2'
    assert len(index) == len(locators['images']) == 6
    for image, locator in zip(index, locators['images']):
        assert image['image_id'] == locator['image_id']
        target = ROOT / image['file']
        assert target.parent == ROOT / 'images' and target.suffix == '.jpg'
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != image['sha256']:
                raise ValueError(f"{image['image_id']}: existing file hash mismatch; left untouched")
            print(f"{image['image_id']}: verified (already present)")
            continue
        query = urllib.parse.urlencode({'dataset': locators['dataset'], 'config': 'default',
                                         'split': 'train', 'offset': locator['row_index'], 'length': 1})
        metadata, revision = get('https://datasets-server.huggingface.co/rows?' + query, 2_000_000)
        if revision != locators['revision']:
            raise ValueError('Dataset revision changed or could not be verified; stop without substituting images')
        # Use only the source-image pointer. Never save metadata, expose preferences,
        # access edited-image pointers or retrieve A/B images.
        rows = json.loads(metadata)['rows']
        if len(rows) != 1 or rows[0]['row_idx'] != locator['row_index']:
            raise ValueError('Unexpected source row')
        url = rows[0]['row']['source_image']['src']
        address = urllib.parse.urlsplit(url)
        expected_path = (f"/cached-assets/{locators['dataset']}/--/{locators['revision']}"
                         f"/--/default/train/{locator['row_index']}/source_image/image.jpg")
        if address.scheme != 'https' or address.hostname != 'datasets-server.huggingface.co' or address.path != expected_path:
            raise ValueError('Source URL does not match fixed revision, development row and source-image field')
        content, _ = get(url, 25_000_000)
        if hashlib.sha256(content).hexdigest() != image['sha256']:
            raise ValueError(f"{image['image_id']}: downloaded bytes differ; stop without resizing or substituting")
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(target)
        print(f"{image['image_id']}: downloaded and verified")
    print('PASS: 6/6 development originals verified; no confirmation inputs or A/B images downloaded.')

if __name__ == '__main__':
    main()
