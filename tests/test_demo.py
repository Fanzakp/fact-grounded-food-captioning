"""Static portfolio demo invariants (no third-party dependencies)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def read_data():
    raw = (DOCS/'assets/js/archive-data.js').read_text(encoding='utf-8')
    return json.loads(raw.split('window.DEMO_DATA = ', 1)[1].rstrip().removesuffix(';'))


class DemoIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read_data()

    def test_unique_507_test_dishes_and_six_models(self):
        rows = self.data['rows']
        self.assertEqual(len(rows), 507)
        self.assertEqual(len({x['id'] for x in rows}), 507)
        self.assertEqual(len(self.data['models']), 6)
        self.assertEqual(self.data['photos'],507)
        self.assertEqual(self.data['featuredPhotos'],12)
        for r in rows:
            self.assertEqual(len(r['c']), 6)
            self.assertEqual(len(r['n']), 6)
            self.assertEqual(len(r['f']), 5)
            self.assertEqual(len(r['gt']), 5)
            self.assertTrue(all(isinstance(text,str) and text.strip() for text in r['c']))
            self.assertTrue(all(len(group)==5 for group in r['n']))

    def test_photo_paths_really_exist(self):
        with_photos=[r for r in self.data['rows'] if r.get('photo')]
        self.assertEqual(len(with_photos), 507)
        for r in with_photos:
            self.assertEqual(r['photo'], f"assets/images/{r['id']}.webp")
            self.assertTrue((DOCS/r['photo']).is_file())

    def test_every_extracted_number_is_numeric_or_omitted(self):
        for r in self.data['rows']:
            for group in r['n']:
                for v in group:
                    self.assertTrue(v is None or isinstance(v,int))
                    if v is not None: self.assertGreaterEqual(v, 0)

    def test_compact_fallback_keeps_records_in_data_not_long_html(self):
        h = (DOCS/'index.html').read_text(encoding='utf-8')
        self.assertIn('class="no-js"',h)
        self.assertEqual(h.count('class="sample-card"'),12)
        self.assertNotIn('507 dishes without JavaScript',h)
        self.assertNotIn('class="archive-record"',h)
        self.assertEqual(h.count('class="gallery-item"'),12)
        self.assertIn('id="fullPhotoGallery"',h)
        self.assertIn('id="modelStats"',h)
        self.assertLess(len(h),100_000)

    def test_script_disables_fallback_only_after_render(self):
        j = (DOCS/'assets/js/explorer.js').read_text(encoding='utf-8')
        self.assertIn("document.body.classList.remove('no-js')",j)
        self.assertIn("document.body.classList.add('js-ready')",j)

    def test_primary_archived_metrics(self):
        models = {m['code']:m for m in self.data['models']}
        self.assertEqual(models['grounded']['bleu'],0.4062)
        self.assertEqual(models['grounded']['cider'],2.3827)
        self.assertEqual(models['grounded']['nutacc'],96.37)
        self.assertEqual(models['conventional']['nutacc'],16.57)

    def test_all_packaged_webp_headers_valid(self):
        # GitHub Actions has no Pillow dependency; verify file signatures quickly.
        for record in self.data['rows']:
            fp = DOCS / record['photo']
            with fp.open('rb') as f:
                head = f.read(12)
            self.assertEqual(head[0:4], b'RIFF', record['id'])
            self.assertEqual(head[8:12], b'WEBP', record['id'])

    def test_pages_assets_and_attribution_present(self):
        html = (DOCS/'index.html').read_text(encoding='utf-8')
        for local_asset in ('assets/css/explorer.css', 'assets/js/archive-data.js', 'assets/js/explorer.js'):
            self.assertTrue((DOCS/local_asset).is_file())
        self.assertTrue((DOCS/'.nojekyll').is_file())
        self.assertIn('https://creativecommons.org/licenses/by/4.0/', html)
        self.assertIn('Nutrition5k (Thames et al., CVPR 2021)', html)
        self.assertIn('converted to WebP', html)
        self.assertNotIn('href="research-data-provenance.json"', html)
        self.assertNotIn('Research code: © 2026 Fanza Khairan Pratama (MIT).', html)
        self.assertIn('code license not yet selected', html)

    def test_documentation_is_concise_and_local_links_resolve(self):
        import re
        md_files=sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.md'))
        for name in md_files:
            document=ROOT/name
            content=document.read_text(encoding='utf-8')
            for url in re.findall(r'\[[^\]]+\]\(([^)]+)\)', content):
                if url.startswith(('https://','http://','mailto:','#')): continue
                target=url.split('#',1)[0]
                if not target: continue
                self.assertTrue((document.parent/target).exists(),f'{name} -> {url}')

if __name__ == '__main__': unittest.main()
