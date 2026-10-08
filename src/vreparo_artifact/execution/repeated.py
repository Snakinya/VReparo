"""Fresh five-run workflow built from the shared fit, generation and judge stages."""

import csv
import math
from pathlib import Path

from .inputs import read, select_alpha, validate_groups, write
from .resources import model_paths
from .sampling import SAMPLING_DATASETS
from .reporting import scores_for
from ..metrics import mean, paired_metrics
from ..repeated import SEEDS


def plan(root, args):
    from .workflows import CASES, JUDGES

    assets = (args.assets or root / 'outputs/assets').absolute()
    python = (args.model_python or root / '.venv-models/bin/python').absolute()
    semantic = (args.semantic_python or root / '.venv-semantic/bin/python').absolute()
    selected = CASES if args.case == 'all' else (args.case,)
    cases = read(root / 'experiments/paper_cases.json')['cases']
    models = {JUDGES[k] for k in ('harmbench', 'strongreject', 'orbench')}
    models.add('models/evaluators/gemma-2b')
    commands = []

    def add(stage, *options, is_semantic=False):
        commands.append([str(semantic if is_semantic else python), str(root / 'run.py'),
                         'experiment', stage, *map(str, options)])

    for case in selected:
        models.update(model_paths(cases[case]))
        for seed in SEEDS:
            output = root / 'outputs/experiments' / case / 'repeated_mitigation' / f'seed_{seed}'
            fit = output / 'fit'
            mapper = fit / 'mapper.pt'
            add('fit', '--case', case, '--assets', assets, '--sampling-seed', seed,
                '--qa', assets / 'data/train_qa.jsonl', '--safety', assets / 'data/train_safety.json', '--output', fit)
            calibration = output / 'xstest'
            add('generate', '--case', case, '--assets', assets, '--experiment', 'calibration',
                '--dataset', 'xstest', '--input', assets / 'data/xstest.json', '--mapper', mapper,
                '--max-new-tokens', 512, '--output', calibration)
            add('select', '--input', calibration / 'generation_0.json', '--output', calibration / 'selected')
            selection = calibration / 'selected/selection.json'
            for dataset in ('harmbench', 'strongreject', 'orbench', 'qa_droid'):
                qa = dataset == 'qa_droid'
                folder = output / dataset
                generated = folder / 'generation_0.json'
                add('generate', '--case', case, '--assets', assets, '--experiment', 'quality' if qa else 'mitigation',
                    '--dataset', dataset, '--input', assets / 'data' / (dataset + '.json'),
                    '--mapper', mapper, '--selection', selection, '--task', 'hotpotqa' if qa else 'safety',
                    '--max-new-tokens', 32 if qa else 512, '--output', folder)
                if not qa:
                    extra = ['--base-model', assets / 'models/evaluators/gemma-2b'] if dataset == 'strongreject' else []
                    add('score', '--evaluator', dataset, '--model', assets / JUDGES[dataset],
                        '--input', generated, '--output', folder / 'scored', '--batch-size', 4,
                        *extra, is_semantic=dataset == 'orbench')
    return {'experiment': 'repeated_mitigation', 'targets': list(selected), 'seeds': list(SEEDS),
            'assets': str(assets), 'datasets': sorted(set(SAMPLING_DATASETS) | {'harmbench'}),
            'models': sorted(models), 'commands': commands}


def summarize(root, targets):
    """Require every seed and population, then write seed rows and means."""
    reference = read(root / 'experiments/datasets.json')['datasets']
    rows, averages = [], []
    for case in targets:
        for seed in SEEDS:
            folder = root / 'outputs/experiments' / case / 'repeated_mitigation' / f'seed_{seed}'
            calibration = read(folder / 'xstest/generation_0.json')
            if calibration['state'] != 'completed':
                raise ValueError('Incomplete calibration')
            alpha = select_alpha(calibration['condition_records'])['selected_alpha']
            if alpha is None:
                raise ValueError('Ineligible run cannot be omitted')
            row = {'case': case, 'seed': seed, 'alpha': alpha}
            for dataset in ('harmbench', 'strongreject', 'orbench', 'qa_droid'):
                generation = folder / dataset / 'generation_0.json'
                source = read(generation)
                if source['state'] != 'completed' or source['fingerprint']['config']['alpha'] != alpha:
                    raise ValueError('Incomplete generation or mismatched selected alpha')
                groups = source['condition_records']
                validate_groups(groups)
                if set(groups) != {'native', 'raw', 'vreparo'}:
                    raise ValueError('Missing comparison conditions')
                if [r['prompt_id'] for r in groups['native']] != reference[dataset]['ids']:
                    raise ValueError('Evaluation population differs from the paper')
                if dataset == 'qa_droid':
                    for condition in ('native', 'raw', 'vreparo'):
                        values = [r['f1'] for r in groups[condition]]
                        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                            raise ValueError('Invalid F1')
                        key = 'qa_f1' if condition == 'vreparo' else f'{condition}_qa_f1'
                        row[key] = mean(values)
                    continue
                scores = scores_for(generation.parent / 'scored' / (dataset + '.json'), generation)
                if dataset == 'orbench':
                    for condition in ('native', 'raw', 'vreparo'):
                        labels = [r['label'] for r in scores[condition]]
                        if any(label not in ('F', 'P', 'N', 'U') for label in labels):
                            raise ValueError('Invalid refusal judgment requires review')
                        prefix = '' if condition == 'vreparo' else condition + '_'
                        row[prefix + 'or_full_refusal'] = labels.count('F') / len(labels)
                        row[prefix + 'or_unscorable'] = labels.count('U')
                else:
                    values = {n: [r['score'] for r in series] for n, series in scores.items()}
                    if dataset == 'harmbench' and any(v not in (0, 1) for series in values.values() for v in series):
                        raise ValueError('HarmBench scores must be binary')
                    for condition in ('raw', 'vreparo'):
                        metrics = paired_metrics(values['native'], values[condition])
                        prefix = dataset if condition == 'vreparo' else f'raw_{dataset}'
                        row.update({f'{prefix}_{k}': metrics[k] for k in ('condition', 'delta', 'sd', 'si', 'sdv')})
                    row[f'native_{dataset}_condition'] = mean(values['native'])
            rows.append(row)
        records = [r for r in rows if r['case'] == case]
        averages.append({'case': case, 'runs': len(records),
                         **{k: mean(r[k] for r in records) for k in records[0] if k not in ('case', 'seed', 'alpha')}})
    destination = root / 'outputs/reports/repeated_mitigation' / ('all' if len(targets) > 1 else targets[0])
    result = {'status': 'completed', 'rows': rows, 'means': averages,
              'units': 'HB and OR are fractions, SR and F1 are 0--1. Multiply HB/OR/F1 by 100 for the paper.',
              'scope': 'Fresh five-run results. Timing and additional QA use separate fixed-map experiments.'}
    write(destination / 'summary.json', result)
    for name, records in [('individual', rows), ('means', averages)]:
        columns = list(records[0])
        with (destination / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        lines = [f'# {name.title()}', '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
        lines.extend('| ' + ' | '.join(str(r[k]) for k in columns) + ' |' for r in records)
        (destination / f'{name}.md').write_text('\n'.join(lines) + '\n')
    return {'status': 'completed', 'runs': len(rows), 'report': str(destination.relative_to(root))}
