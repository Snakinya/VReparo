import argparse
import copy
import json
from pathlib import Path
import unittest
import tempfile

from vreparo_artifact.repeated import build_tables, validate_data, summarize
from vreparo_artifact.execution import workflows
from vreparo_artifact.execution.cli import add_arguments
from vreparo_artifact.execution.sampling import balanced_order
from vreparo_artifact.metrics import mean, paired_metrics

ROOT = Path(__file__).resolve().parents[1]


class RepeatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'data/repeated.json').read_text())

    def test_all_new_cells_match_independent_paper_transcription(self):
        expected = json.loads((ROOT / 'expected/repeated.json').read_text())
        self.assertEqual(build_tables(ROOT / 'data'), expected)
        metrics = summarize(validate_data(self.data))
        hb = mean(metrics[c][f'seed_{s}']['harmbench']['condition'] for c in metrics for s in range(5))
        sr = mean(metrics[c][f'seed_{s}']['strongreject']['condition'] for c in metrics for s in range(5))
        self.assertEqual(f'{hb*100:.2f}', '39.21')
        self.assertEqual(f'{sr:.4f}', '0.2163')

    def test_missing_run_and_changed_alpha_fail(self):
        changed = copy.deepcopy(self.data)
        del changed['cases']['base_direct']['harmbench']['seed_4']
        with self.assertRaises(ValueError):
            validate_data(changed)
        changed = copy.deepcopy(self.data)
        changed['cases']['base_direct']['calibration']['0']['selected_alpha'] = 1
        with self.assertRaises(ValueError):
            validate_data(changed)

    def test_paired_metrics_are_computed_before_averaging(self):
        # Opposite per-prompt changes must not cancel when computing SDV.
        baseline = [0.5, 0.5]
        runs = [[0, 1], [1, 0]]
        correct = mean(paired_metrics(baseline, r)['sdv'] for r in runs)
        incorrect = paired_metrics(baseline, [mean(v) for v in zip(*runs)])['sdv']
        self.assertEqual(correct, 0.5)
        self.assertEqual(incorrect, 0)

    def test_sampling_recreates_all_frozen_selections(self):
        from collections import Counter
        frozen = json.loads((ROOT / 'experiments/sampling.json').read_text())
        self.assertEqual(frozen['eligible_sizes'], {'qa': 143, 'safety': 330})
        for seed in range(5):
            for domain in ('qa', 'safety'):
                selected = sorted(balanced_order(frozen['pools'][domain], seed, domain)[:32])
                rows = frozen['subsets'][str(seed)][domain]
                self.assertEqual(selected, [r['source_index'] for r in rows])
                counts = Counter(r['stratum'] for r in rows)
                self.assertEqual(len(counts), 3 if domain == 'qa' else 11)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_fresh_plan_carries_each_seed_and_its_own_selection(self):
        args = workflows.parser('repeated_mitigation').parse_args(['--case', 'all'])
        recipe = workflows.plan(ROOT, 'repeated_mitigation', args)
        resources = json.loads((ROOT / 'experiments/resources.json').read_text())
        self.assertLessEqual(set(recipe['datasets']), set(resources['datasets']))
        self.assertLessEqual(set(recipe['models']), set(resources['models']))
        parser = argparse.ArgumentParser()
        add_arguments(parser.add_subparsers(dest='command', required=True))
        fit, select, evaluation = [], [], []
        for command in recipe['commands']:
            parsed = parser.parse_args(command[2:])
            if parsed.stage == 'fit':
                fit.append(parsed)
            elif parsed.stage == 'select':
                select.append(parsed)
            elif parsed.stage == 'generate' and parsed.experiment != 'calibration':
                evaluation.append(parsed)
                self.assertEqual(parsed.selection.parents[2], parsed.mapper.parents[1])
                self.assertEqual(parsed.dataset, 'qa_droid' if parsed.task == 'hotpotqa' else parsed.dataset)
        self.assertEqual(len(fit), 20)
        self.assertEqual(len(select), 20)
        self.assertEqual(len(evaluation), 80)
        self.assertEqual({(v.case, v.sampling_seed) for v in fit}, {(c, s) for c in recipe['targets'] for s in range(5)})

    def test_fresh_summary_requires_all_seeds_and_preserves_pairing(self):
        from vreparo_artifact.execution.inputs import digest, write
        from vreparo_artifact.execution.repeated import summarize as fresh_summary
        case = self.data['cases']['base_droid']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'experiments/datasets.json', json.loads((ROOT / 'experiments/datasets.json').read_text()))
            for seed in range(5):
                base = root / f'outputs/experiments/base_droid/repeated_mitigation/seed_{seed}'
                cal = case['calibration'][str(seed)]
                groups = {}
                for condition, values in cal['refusal'].items():
                    name = condition if condition in ('native', 'raw') else f'vreparo_a{float(condition):g}'
                    groups[name] = [{'prompt_id': key, 'prompt': 'Synthetic', 'label': label, 'type': subtype,
                                     'string_refusal': bool(refusal)} for key, label, subtype, refusal in
                                    zip(cal['prompt_ids'], cal['labels'], cal['subtypes'], values)]
                write(base / 'xstest/generation_0.json', {'state': 'completed', 'condition_records': groups})
                for dataset in ('harmbench', 'strongreject', 'orbench', 'hotpotqa'):
                    data = case[dataset]
                    target = base / ('qa_droid' if dataset == 'hotpotqa' else dataset)
                    groups, scores = {}, {}
                    for name in ('native', 'raw', 'vreparo'):
                        values = data[f'seed_{seed}' if name == 'vreparo' else name]
                        groups[name] = [{'prompt_id': key, 'prompt': 'Synthetic', **({'f1': value} if dataset == 'hotpotqa' else {})}
                                        for key, value in zip(data['prompt_ids'], values)]
                        scores[name] = [{'prompt_id': key, 'label' if dataset == 'orbench' else 'score': value}
                                        for key, value in zip(data['prompt_ids'], values)]
                    source = target / 'generation_0.json'
                    write(source, {'state': 'completed', 'fingerprint': {'config': {'alpha': cal['selected_alpha']}},
                                   'condition_records': groups})
                    if dataset != 'hotpotqa':
                        write(target / f'scored/{dataset}.json', {'state': 'completed', 'input_sha256': digest(source), 'conditions': scores})
            result = fresh_summary(root, ['base_droid'])
            output = json.loads((root / result['report'] / 'summary.json').read_text())
            self.assertEqual(result['runs'], 5)
            expected = summarize({'base_droid': case})['base_droid']
            for dataset in ('harmbench', 'strongreject'):
                for metric in ('condition', 'delta', 'sd', 'si', 'sdv'):
                    self.assertAlmostEqual(output['means'][0][f'{dataset}_{metric}'],
                                           mean(expected[f'seed_{s}'][dataset][metric] for s in range(5)))
            missing = root / 'outputs/experiments/base_droid/repeated_mitigation/seed_4/qa_droid/generation_0.json'
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                fresh_summary(root, ['base_droid'])


if __name__ == '__main__':
    unittest.main()
