"""Guard the judge-facing reading path without changing the scientific evaluation."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


class ReadabilityTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads((ROOT / 'HypeCheck.ipynb').read_text())

    def test_no_embedded_source_or_data_walls(self):
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                self.assertLessEqual(len(source), 2400)
                self.assertLessEqual(len(source.splitlines()), 24)
                self.assertNotIn('EMBEDDED_INPUTS =', source)
                self.assertNotIn('SOURCES =', source)

    def test_reading_route_and_plain_language(self):
        text = '\n'.join(''.join(cell['source']) for cell in self.nb['cells'] if cell['cell_type'] == 'markdown')
        for phrase in ('Our user:', 'Finding:', 'Runtime → Run all', '## 1. Load',
                       '## 4. Baselines', '## 6. Where', '## 8. Do some kinds',
                       '## 9. The demo', '## 10. Results', '## 11. What should',
                       'Brier error', 'result is exploratory', 'not fine-tuned'):
            self.assertIn(phrase, text)
        self.assertTrue(self.nb['metadata']['colab']['toc_visible'])

    def test_verified_source_and_support_parity(self):
        import colab_setup
        metadata = self.nb['metadata']['hypecheck']
        for name, digest in metadata['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(colab_setup.FILE_DIGESTS[name], digest)
        for name, digest in metadata['bootstrap_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_rich_table_escapes_source_text(self):
        import pandas as pd
        from notebook_support import table
        with patch('IPython.display.display') as display:
            table(pd.DataFrame({'Post': ['<script>not trusted</script>']}), '<caption>')
        output = display.call_args.args[0].data
        self.assertIn('&lt;script&gt;', output)
        self.assertNotIn('<script>', output)
        self.assertIn('&lt;caption&gt;', output)

    def test_chart_uses_actual_scores_and_full_scale(self):
        from notebook_support import accuracy_chart
        report = json.loads((ROOT / 'docs/headline-report.json').read_text())
        with patch('IPython.display.display') as display:
            accuracy_chart(report, ['Always up', 'Sentiment + coin'])
        output = display.call_args.args[0].data
        self.assertIn(f"{report['metrics']['Sentiment + coin']['accuracy']:.2%}", output)
        self.assertIn('100%', output)
        self.assertIn('0%', output)
        self.assertIn('aria-label', output)

    def test_saved_outputs_include_tables_and_chart(self):
        outputs = [output for cell in self.nb['cells'] if cell['cell_type'] == 'code'
                   for output in cell.get('outputs', [])]
        tables = [output for output in outputs if 'text/html' in output.get('data', {})]
        charts = [output for output in outputs if 'image/svg+xml' in output.get('data', {})]
        self.assertGreaterEqual(len(tables), 8)
        self.assertEqual(len(charts), 1)
        self.assertFalse(any(output['output_type'] == 'error' for output in outputs))


if __name__ == '__main__':
    unittest.main()
