"""Curated source/data export; excludes API responses, private keys and media."""
from __future__ import annotations
import argparse
import ast
import json
import re
import shutil
import zipfile
from pathlib import Path
from local_backend import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--zip', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    engine = root / 'engine'
    sources = list((args.source / 'tools').glob('*.py')) + list((args.source / 'tools/event_decision').rglob('*.py'))
    sources += list((args.source / 'docs').glob('*.py')) + [args.source / 'run_effectiveness_v919.sh']
    source_hashes = {}
    for path in sorted(sources):
        text = path.read_text(encoding='utf-8-sig')
        if re.search(r'sk-[A-Za-z0-9_-]{20,}', text) or '-----BEGIN PRIVATE KEY-----' in text:
            raise ValueError('Possible embedded credential; export stopped: ' + str(path))
        relative = path.relative_to(args.source)
        target = engine / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[relative.as_posix()] = sha(path)
    fixture = root / 'frozen_cohort'
    fixture.mkdir(exist_ok=True)
    for name, source in {'protocol.json': 'protocol.json', 'graph_catalog.json': 'graph_catalog.json',
                         'inputs.jsonl': 'pilot/inputs.jsonl', 'pilot_labels.jsonl': 'private/pilot_labels.jsonl',
                         'enrollment.json': 'pilot/enrollment.json'}.items():
        shutil.copyfile(args.run / source, fixture / name)
    rows = [json.loads(l) for l in (fixture / 'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    # The receipt name is a canonical SHA256 of the video ID, not the raw filename.
    import sys
    sys.path.insert(0, str(engine / 'tools'))
    from event_decision.contracts import semantic_sha256
    expected = {}
    for vid in sorted({r['video_id'] for r in rows}):
        receipt = args.run / 'media' / semantic_sha256(vid)[:24] / 'receipt.json'
        if receipt.exists():
            obj = json.loads(receipt.read_text(encoding='utf-8'))
            expected[vid] = obj
    write(fixture / 'media_expectations.json', expected)
    write(fixture / 'inventory.json', {f.name: sha(f) for f in fixture.iterdir() if f.name != 'inventory.json'})
    write(root / 'source_export.json', {'source_files': source_hashes, 'cohort_videos': len({r['video_id'] for r in rows}),
        'cohort_windows': len(rows), 'API_video_receipts_available': len(expected),
        'API_caches_included': False, 'secrets_included': False, 'model_weights_included': False,
        'scope': 'current V919 matching + C1/C2 + fit/calibrate/evaluate, with historical Python dependencies',
        'inference_entry_point': 'run_local.py (network blocked; lazy local Transformers only)'})
    payload = sorted(f for f in root.rglob('*') if f.is_file() and not any(x in f.relative_to(root).parts for x in
                     ('runs', '__pycache__', '.pytest_cache')) and f.suffix not in ('.pyc', '.zip'))
    payload = [f for f in payload if f.name != 'PACKAGE_SHA256.json']
    write(root / 'PACKAGE_SHA256.json', {f.relative_to(root).as_posix(): sha(f) for f in payload})
    payload.append(root / 'PACKAGE_SHA256.json')
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for f in payload:
            archive.write(f, root.name + '/' + f.relative_to(root).as_posix())
    with zipfile.ZipFile(args.zip) as archive:
        assert archive.testzip() is None
    print(json.dumps({'zip': str(args.zip.resolve()), 'files': len(payload), 'bytes': args.zip.stat().st_size,
                      'sha256': sha(args.zip)}, indent=2))


if __name__ == '__main__':
    main()
