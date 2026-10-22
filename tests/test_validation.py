"""Regression coverage for checksum preflight and retained evidence verification."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load('portfolio_verify', ROOT / 'validation/verify.py')
inference = load('portfolio_inference', ROOT / 'validation/validate_checkpoints.py')


class ValidationTests(unittest.TestCase):
    def test_bundled_evidence_recalculates(self):
        report = verifier.verify()
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(sum(c['status'] == 'PASS' for c in report['checks']), 8)

    def test_bad_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'observed'):
            verifier.compare(55.67, 53.59, 'separate CNN runs')

    def test_help_needs_no_optional_model_imports(self):
        completed = subprocess.run([sys.executable, '-B', str(ROOT / 'validation/validate_checkpoints.py'), '--help'],
                                   check=True, capture_output=True, text=True)
        self.assertIn('--check-inputs', completed.stdout)

    def test_missing_inference_input_stops_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Missing input'):
                inference.check_inputs(Path(tmp) / 'inputs', Path(tmp) / 'archive', 'freeze')

    def test_preflight_rejects_changed_weights_or_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inputs, archive = base / 'inputs', base / 'archive'
            archive.mkdir()
            ids = [f'dish_{i:010d}' for i in range(507)]
            id_file = archive / 'dish_ids_test.txt'
            id_file.write_text('\n'.join(ids) + '\n', encoding='utf-8')
            images = []
            for dish in ids:
                image = inputs / 'images' / dish / 'rgb.png'
                image.parent.mkdir(parents=True)
                image.write_bytes(b'fixture')
                images.append((dish, inference.sha(image)))
            checkpoint = inputs / 'cnn_best.pth'
            checkpoint.write_bytes(b'fixture weights; never loaded')
            evidence = {'external_inputs_sha256': {
                'archive/dish_ids_test.txt': inference.sha(id_file),
                'inputs/cnn_best.pth': inference.sha(checkpoint)},
                'source_png_collection_sha256': verifier.collection_sha(images)}
            self.assertEqual(inference.check_inputs(inputs, archive, 'cnn', evidence)['status'], 'PASS')
            checkpoint.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                inference.check_inputs(inputs, archive, 'cnn', evidence)
            checkpoint.write_bytes(b'fixture weights; never loaded')
            (inputs / 'images' / ids[-1] / 'rgb.png').write_bytes(b'changed image')
            with self.assertRaisesRegex(ValueError, 'PNG collection'):
                inference.check_inputs(inputs, archive, 'cnn', evidence)


if __name__ == '__main__':
    unittest.main()
