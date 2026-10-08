"""Paired metrics and per-seed evidence for the paper's five sampling runs."""

import json
from pathlib import Path

from .evaluation import CASES, _ids, _keys, _numbers, _require, calibration_rows
from .metrics import format_number, mean, paired_metrics

SEEDS = tuple(range(5))
CONDITIONS = ('native', 'raw', *(f'seed_{s}' for s in SEEDS))


def validate_data(payload):
    _keys(payload, {'schema_version', 'seeds', 'budget_per_domain', 'sources', 'cases'}, 'five-run data')
    _require(payload['schema_version'] == 1 and payload['seeds'] == list(SEEDS)
             and payload['budget_per_domain'] == 32, 'Expected all five 32+32 runs')
    _keys(payload['cases'], {c for c, _, _ in CASES}, 'five-run cases')
    for case in payload['cases'].values():
        _keys(case, {'harmbench', 'strongreject', 'orbench', 'hotpotqa', 'calibration'}, 'five-run case')
        for dataset, n, population in [('harmbench', 127, 'harmbench'), ('strongreject', 313, 'strongreject'),
                                        ('orbench', 200, 'orbench'), ('hotpotqa', 248, 'hotpotqa_droid')]:
            group = case[dataset]
            _keys(group, {'prompt_ids', *CONDITIONS}, 'five-run observations')
            _ids(group['prompt_ids'], n, population)
            for name in CONDITIONS:
                if dataset == 'orbench':
                    _require(len(group[name]) == n and all(v in ('F', 'P', 'N', 'U') for v in group[name]),
                             'Invalid refusal labels')
                else:
                    _numbers(group[name], n, binary=dataset == 'harmbench')
        _keys(case['calibration'], {str(s) for s in SEEDS}, 'seed calibration')
        for calibration in case['calibration'].values():
            calibration_rows(calibration)
    return payload['cases']


def summarize(cases):
    """Compute each run's paired metrics before averaging, retaining all seeds."""
    result = {}
    for case_id, case in cases.items():
        result[case_id] = {}
        for name in CONDITIONS:
            result[case_id][name] = {
                dataset: paired_metrics(case[dataset]['native'], case[dataset][name])
                for dataset in ('harmbench', 'strongreject')}
            result[case_id][name].update(
                qa_f1=mean(case['hotpotqa'][name]),
                or_full_refusal=case['orbench'][name].count('F') / 200,
                or_unscorable=case['orbench'][name].count('U'))
    return result


def build_tables(data_dir: Path):
    cases = validate_data(json.loads((data_dir / 'repeated.json').read_text()))
    calculated = summarize(cases)
    safety, absolute, individual = [], [], []
    for case_id, model, config in CASES:
        metrics = calculated[case_id]
        for condition, names in [('Reuse', ['raw']), ('VReparo', [f'seed_{s}' for s in SEEDS])]:
            cells = []
            for dataset, scale, digits in [('harmbench', 100, 2), ('strongreject', 1, 4)]:
                cells.extend(format_number(mean(metrics[n][dataset][k] for n in names) * scale, digits)
                             for k in ('delta', 'sd', 'si', 'sdv'))
            safety.append([model, config, condition, *cells])
        for seed in SEEDS:
            row = metrics[f'seed_{seed}']
            individual.append([model, config, str(seed), f"{cases[case_id]['calibration'][str(seed)]['selected_alpha']:g}",
                               format_number(row['harmbench']['sdv'] * 100, 2),
                               format_number(row['strongreject']['sdv'], 4),
                               format_number(row['qa_f1'] * 100, 2),
                               format_number(row['or_full_refusal'] * 100, 1)])
    for prefix, model in [('code', 'LoRI-D'), ('base', 'Llama-3-8B')]:
        for condition, names in [('Native', ['native']), ('Reuse', ['raw']), ('VReparo', [f'seed_{s}' for s in SEEDS])]:
            cells = []
            for suffix in ('direct', 'droid'):
                metrics = calculated[f'{prefix}_{suffix}']
                cells.extend([format_number(mean(metrics[n]['harmbench']['condition'] for n in names) * 100, 2),
                              format_number(mean(metrics[n]['strongreject']['condition'] for n in names), 4),
                              format_number(mean(metrics[n]['or_full_refusal'] for n in names) * 100, 1)])
            absolute.append([model, condition, *cells])
    return {
        'table_03': {'title': 'Native-referenced safety metrics', 'columns': ['Model', 'Configuration', 'Condition', 'HB delta (pp)', 'HB SD (%)', 'HB SI (%)', 'HB SDV (%)', 'SR delta', 'SR SD', 'SR SI', 'SR SDV'], 'rows': safety},
        'table_10': {'title': 'Safety scores and benign refusal', 'columns': ['Model', 'Condition', 'Full KV ASR (%)', 'Full KV SR mean', 'Full KV refusal (%)', 'DroidSpeak ASR (%)', 'DroidSpeak SR mean', 'DroidSpeak refusal (%)'], 'rows': absolute},
        'table_11': {'title': 'Results for each sampling seed', 'columns': ['Model', 'Configuration', 'Seed', 'Alpha', 'HB SDV (%)', 'SR SDV', 'HotpotQA F1 (x100)', 'OR-Bench refusal (%)'], 'rows': individual},
    }
