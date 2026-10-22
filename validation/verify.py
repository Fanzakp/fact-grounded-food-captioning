"""Verify bundled historical outputs without training, inference, or downloads.

Python 3.12 standard library suffices. --language additionally needs nltk and
pycocoevalcap already listed in the optional notebook requirements.
"""
import argparse
import ast
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_utils import TARGETS, fact_numbers, legacy_extract_numbers, numeric_metrics, read_indexed_csv

MODEL_NAMES = ['Baseline 1 (Template Fill)', 'Baseline 2 (T5-only)', 'ViT+T5 Freeze',
               'ViT+T5 E2E', 'Conventional Freeze', 'Conventional E2E']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collection_sha(items):
    text = ''.join(f'{name} {digest}\n' for name, digest in sorted(items))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def fact_string(row):
    cal, pro, fat, carb, mass = row['f']
    return (f"ingredients: {row['i']} | mass: {mass} g | calories: {cal} kcal | "
            f"protein: {pro} g | fat: {fat} g | carbohydrates: {carb} g")


def compare(observed, expected, label, tolerance=1e-9):
    require(observed is not None and math.isclose(observed, float(expected), rel_tol=0, abs_tol=tolerance),
            f'{label}: observed {observed}, expected {expected}')


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links.extend(attrs[k] for k in ('href', 'src') if k in attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])


def verify(language=False):
    checks = []
    record = lambda name, detail: checks.append({'status': 'PASS', 'check': name, 'detail': detail})
    evidence = json.loads((ROOT / 'validation/evidence/checkpoint-evidence.json').read_text(encoding='utf-8'))
    for rel, expected in evidence['bundled_files_sha256'].items():
        require(sha(ROOT / rel) == expected, f'Checksum mismatch: {rel}')
    record('Evidence hashes', 'Historical tables, archive payload, and retained inference tables match recorded SHA-256 values.')

    raw = (ROOT / 'docs/assets/js/archive-data.js').read_text(encoding='utf-8')
    data = json.loads(raw.split('=', 1)[1].strip().removesuffix(';'))
    rows = data['rows']
    require(len(rows) == len({r['id'] for r in rows}) == 507, 'Expected 507 unique dishes')
    require(len(data['models']) == 6 and sum(len(r['c']) for r in rows) == 3042, 'Incomplete model/caption coverage')
    photos = list((ROOT / 'docs/assets/images').glob('*.webp'))
    require(len(photos) == 507, 'Expected exactly 507 WebP photos')
    require(collection_sha((p.name, sha(p)) for p in photos) == evidence['webp_collection_sha256'], 'WebP collection changed')
    by_id = {r['id']: r for r in rows}
    for row in rows:
        require(row['photo'] == f"assets/images/{row['id']}.webp", f"Bad photo path: {row['id']}")
        image = ROOT / 'docs' / row['photo']
        with image.open('rb') as stream:
            header = stream.read(12)
        require(header[:4] == b'RIFF' and header[8:12] == b'WEBP', f'Invalid WebP header: {image.name}')
        require(len(row['c']) == len(row['n']) == 6 and len(row['f']) == len(row['gt']) == 5, 'Malformed dish record')
        for j, text in enumerate(row['c']):
            require(text and [legacy_extract_numbers(text)[c] for c in TARGETS] == row['n'][j], 'Stored parser extraction changed')
    record('Complete demo', '507 source-matching photos; 3,042 nonempty captions; six models; numeric extraction consistent.')

    cnn = read_indexed_csv(ROOT / 'validation/evidence/cnn-test.csv')
    require(set(cnn) == set(by_id), 'CNN evidence coverage differs')
    truth = {dish: {c: float(row[f'measured_{c}']) for c in TARGETS} for dish, row in cnn.items()}
    saved = read_indexed_csv(ROOT / 'results/eval_final.csv', 'Model')
    require(set(saved) == set(MODEL_NAMES), 'Historical model roster differs')
    comparisons = 0
    for j, name in enumerate(MODEL_NAMES):
        generated = [{'dish_id': r['id'], 'fact_string': fact_string(r), 'generated_caption': r['c'][j]} for r in rows]
        for metric, value in numeric_metrics(generated, truth)['metrics'].items():
            compare(value, saved[name][metric], f'{name}/{metric}')
            comparisons += 1
        for key, csv_key in [('nutacc', 'NutAcc_avg'), ('bleu', 'BLEU-4'), ('cider', 'CIDEr')]:
            compare(data['models'][j][key], saved[name][csv_key], f'{name}/demo {key}')
    record('Historical caption metrics', f'{comparisons} numerical scores recomputed from saved caption text; aggregate demo values match historical CSV.')

    historical_cnn = read_indexed_csv(ROOT / 'results/cnn_test_metrics.csv', 'Nutrient')
    for col in TARGETS:
        measured = [truth[d][col] for d in cnn]
        old = [float(cnn[d][f'archived_{col}']) for d in cnn]
        new = [float(cnn[d][f'checkpoint_{col}']) for d in cnn]
        for dish in cnn:
            r = by_id[dish]
            j = TARGETS.index(col)
            require(max(0, round(float(cnn[dish][f'archived_{col}']))) == r['f'][j], f'Historical fact mismatch: {dish}/{col}')
            compare(round(truth[dish][col], 2), r['gt'][j], f'{dish}/measured {col}')
        old_mae = mean(abs(a-b) for a,b in zip(old, measured))
        squared = sum((a-b)**2 for a,b in zip(old, measured))
        variance = sum((v-mean(measured))**2 for v in measured)
        for metric, value in {'MAE': round(old_mae, 2), 'RMSE': round(math.sqrt(squared/507), 2),
                              'R2': round(1-squared/variance, 3), 'Mean_GT': round(mean(measured), 2)}.items():
            compare(value, historical_cnn[col][metric], f'Historical CNN/{col}/{metric}')
        compare(mean(abs(a-b) for a,b in zip(new, measured)), evidence['runs']['cnn']['metrics'][col]['MAE'], f'Retained checkpoint/{col}/MAE')
    compare(float(historical_cnn['calories']['MAE']), 53.59, 'Historical calorie MAE')
    compare(round(evidence['runs']['cnn']['metrics']['calories']['MAE'], 2), 55.67, 'Checkpoint calorie MAE')
    record('CNN discrepancy', '20 historical regression scores match; retained checkpoint calorie MAE 55.67 vs historical 53.59 kcal.')

    freeze = read_indexed_csv(ROOT / 'validation/evidence/freeze-captions.csv')
    require(set(freeze) == set(by_id), 'Freeze evidence coverage differs')
    matches = sum(r['generated_caption'] == by_id[d]['c'][2] for d,r in freeze.items())
    require(matches == evidence['runs']['freeze']['exact_caption_matches'] == 507, 'Freeze text differs from historical archive')
    for name, item in evidence['freeze_metric_comparison'].items():
        require(item['matches'] is True, f'Recorded comparison mismatch: {name}')
        compare(item['checkpoint'], saved['ViT+T5 Freeze'][name], f'Recorded Freeze/{name}')
        compare(item['archived'], item['checkpoint'], f'Recorded comparison/{name}')
    record('Prior Freeze inference', '507/507 retained generated captions equal historical text; recorded BLEU-4/CIDEr/NutAcc agree.')

    pipeline = list(read_indexed_csv(ROOT / 'validation/evidence/pipeline-sample.csv').values())
    require(len(pipeline) == evidence['runs']['pipeline_sample']['rows'] == 8, 'Expected eight integration samples')
    for row in pipeline:
        supplied = fact_numbers(row['fact_string'])
        for col in TARGETS:
            compare(supplied[col], round(float(cnn[row['dish_id']][f'checkpoint_{col}'])), f'Fresh CNN fact/{col}')
    compare(numeric_metrics(pipeline, truth)['metrics']['NutAcc_avg'], 100, 'Prior integration NutAcc')
    require(sum(r['generated_caption'] == by_id[r['dish_id']]['c'][2] for r in pipeline) == evidence['runs']['pipeline_sample']['exact_caption_matches'] == 0, 'Unexpected integration text comparison')
    record('Prior integration inference', 'Eight saved samples follow fresh CNN facts (100.00% NutAcc); 0/8 identical historical captions, as recorded.')

    for file in ROOT.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'), filename=str(file.relative_to(ROOT)))
    notebook = json.loads((ROOT / 'notebooks/food_nutrition_captioning.ipynb').read_text(encoding='utf-8'))
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']), filename=f'notebook-cell-{i}')
            require(cell['outputs'] == [] and cell['execution_count'] is None, 'Notebook contains saved output')
    record('Python and notebook syntax', 'All packaged Python and notebook code cells parse; notebook outputs are clear.')

    html_file = ROOT / 'docs/index.html'
    html = html_file.read_text(encoding='utf-8')
    parser = References()
    parser.feed(html)
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        require(not url.path.startswith('/'), f'Root-relative link breaks project Pages paths: {link}')
        require((html_file.parent / unquote(url.path)).exists() if url.path else True, f'Broken HTML link: {link}')
        if not url.path and url.fragment:
            require(unquote(url.fragment) in parser.ids, f'Missing HTML anchor: {link}')
    markdown = [(p, p.read_text(encoding='utf-8')) for p in ROOT.rglob('*.md')]
    markdown.append((ROOT / 'notebooks/food_nutrition_captioning.ipynb', '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')))
    for file, text in markdown:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            url = urlsplit(link)
            if not url.scheme and not url.netloc and url.path:
                require((file.parent / unquote(url.path)).exists(), f'Broken Markdown link: {file.name} -> {link}')
    require((ROOT / 'docs/.nojekyll').is_file(), 'Missing static-host marker')
    require('https://creativecommons.org/licenses/by/4.0/' in html and 'converted to WebP' in html, 'Dataset attribution missing')
    record('Relative links and attribution', 'README/notebook links and project Pages asset paths resolve; Nutrition5k CC BY 4.0 attribution remains visible.')

    if language:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from pycocoevalcap.cider.cider import Cider
        refs = [r['ref'] for r in rows]
        for j, name in enumerate(MODEL_NAMES):
            hyps = [r['c'][j] for r in rows]
            bleu = corpus_bleu([[r.lower().split()] for r in refs], [h.lower().split() for h in hyps],
                               weights=(.25, .25, .25, .25), smoothing_function=SmoothingFunction().method1)
            cider = float(Cider().compute_score({r['id']: [r['ref']] for r in rows}, {r['id']: [r['c'][j]] for r in rows})[0])
            compare(round(bleu, 4), saved[name]['BLEU-4'], f'{name}/BLEU-4')
            compare(round(cider, 4), saved[name]['CIDEr'], f'{name}/CIDEr')
        record('Language metrics', 'All 12 BLEU-4/CIDEr values recomputed with nltk and pycocoevalcap match historical CSV.')
    else:
        checks.append({'status': 'SKIP', 'check': 'Fresh language score computation', 'detail': 'Use --language with optional evaluation libraries; retained comparison evidence was checked.'})
    checks.append({'status': 'SKIP', 'check': 'Fresh checkpoint inference', 'detail': 'No inference is performed by this verifier; see optional validate_checkpoints.py.'})
    return {'status': 'PASS', 'scope': 'Saved-output checks only; no model execution', 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', action='store_true')
    args = parser.parse_args()
    try:
        result = verify(args.language)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    for check in result['checks']:
        print(f"{check['status']}: {check['check']} — {check['detail']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
