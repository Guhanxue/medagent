"""Check the usable release and the consistency of saved result records."""
import json
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_summary_runs_without_repo_working_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            run = subprocess.run([sys.executable, str(ROOT / 'summarize_results.py')], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        for expected in ('K=20: Dice 0.774; harmed 0/10', 'harmed 13/52', 'harmed 9/52', '7/7 verified packages'):
            self.assertIn(expected, run.stdout)

    def test_four_draw_aggregates_match_released_per_transfer_records(self):
        data = json.loads((ROOT / 'results/classification.json').read_text())['four_draw_selection']
        for name, field in [('audit_score_winner', 'audit_winner_full'), ('medadaptagent', 'shipped')]:
            means = [mean(row[field] for row in d['choices']) for d in data['per_draw']]
            self.assertAlmostEqual(mean(means), data[name]['mean'])
            self.assertAlmostEqual(pstdev(means), data[name]['population_sd'])
            harmed = sum(row[field] < row['source'] - 1e-9 for d in data['per_draw'] for row in d['choices'])
            self.assertEqual(harmed, data[name]['harmed_total'])

    def test_public_packages_match_saved_evidence(self):
        rows = json.loads((ROOT / 'results/public_model_attempts.json').read_text())
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertTrue(row['verified_package'])
            self.assertEqual(row['exact_reload_matches'], 10)
            self.assertIsNotNone(row['delivered_dice'])
            if row['evidence'].startswith('results/'):
                evidence = json.loads((ROOT / row['evidence']).read_text())
                self.assertEqual(evidence['decision'], row['decision'])
                self.assertEqual(evidence['selected_action'], row['selected_action'])
                self.assertAlmostEqual(evidence['delivered_hidden_test_dice'], row['delivered_dice'])
                self.assertEqual(evidence['reload_verification']['exact_matching_prediction_masks'], 10)

    def test_planner_results_use_matched_test_cases(self):
        data = json.loads((ROOT / 'results/llm_planner_comparison.json').read_text())
        self.assertEqual(len(data['rows']), 4)
        reference = data['rows'][0]
        for row in data['rows']:
            self.assertEqual(row['split'], reference['split'])
            self.assertEqual(row['checkpoint_sha256'], reference['checkpoint_sha256'])
            cases = row['delivered_hidden_test_per_case']
            self.assertEqual(len(cases), 10)
            self.assertAlmostEqual(mean(x['dice'] for x in cases), row['delivered_hidden_test_dice'], places=4)
            self.assertEqual(row['deployment_artifact']['exact_matching_prediction_masks'], 10)



if __name__ == '__main__':
    unittest.main()
