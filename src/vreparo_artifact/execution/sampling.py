"""Reconstruct the five paper selections from pinned public dataset rows."""

from collections import defaultdict
import hashlib
import random

from .inputs import digest, identifier, read, rows


def norm(text):
    return ' '.join(text.casefold().split())


def qa_key(row):
    return norm(row['input']), norm(row['context'])


def balanced_order(pool, seed, domain):
    groups = defaultdict(list)
    for row in pool:
        groups[row['stratum']].append(row['source_index'])
    value = hashlib.sha256(f'vreparo-stratified-v1:{seed}:{domain}'.encode()).digest()
    rng = random.Random(int.from_bytes(value, 'big'))
    for values in groups.values():
        rng.shuffle(values)
    order, labels = [], sorted(groups)
    for depth in range(max(map(len, groups.values()))):
        cycle = list(labels)
        rng.shuffle(cycle)
        order.extend(groups[label][depth] for label in cycle if depth < len(groups[label]))
    return order


def build_pools(data):
    excluded = sum((data[k] for k in ('profile', 'qa_droid', '2wikimqa', 'multifieldqa_en')), [])
    ids, texts = {identifier(r) for r in excluded}, {qa_key(r) for r in excluded}
    qa = [{'source_index': i, 'id': identifier(r), 'length': int(r['length'])}
          for i, r in enumerate(data['train_qa']) if identifier(r) not in ids and qa_key(r) not in texts]
    ranked = sorted(qa, key=lambda r: (r['length'], r['id']))
    for i, row in enumerate(ranked):
        row['stratum'] = ('short', 'medium', 'long')[min(2, i * 3 // len(ranked))]
    excluded = {norm(r['prompt']) for k in ('xstest', 'harmbench159', 'strongreject', 'orbench') for r in data[k]}
    safety = [{'source_index': i, 'id': identifier(r), 'stratum': r['category']}
              for i, r in enumerate(data['train_safety']) if norm(r['prompt']) not in excluded]
    if len({qa_key(data['train_qa'][r['source_index']]) for r in qa}) != len(qa):
        raise ValueError('Duplicate QA candidates')
    if len({norm(data['train_safety'][r['source_index']]['prompt']) for r in safety}) != len(safety):
        raise ValueError('Duplicate safety candidates')
    return {'qa': qa, 'safety': safety}


SAMPLING_DATASETS = ('train_qa', 'train_safety', 'profile', 'qa_droid', '2wikimqa',
                     'multifieldqa_en', 'xstest', 'harmbench159', 'strongreject', 'orbench')


def selected_indices(root, assets, seed):
    if type(seed) is not int or seed not in range(5):
        raise ValueError('Sampling seed must be 0 through 4')
    resources = read(root / 'experiments/resources.json')['datasets']
    data = {}
    for name in SAMPLING_DATASETS:
        recipe = resources[name]
        path = assets / 'data' / recipe['filename']
        if digest(path) != recipe['sha256']:
            raise ValueError('Sampling input differs from the pinned dataset: ' + name)
        data[name] = rows(path)
    pools = build_pools(data)
    frozen = read(root / 'experiments/sampling.json')
    if pools != frozen['pools']:
        raise ValueError('Reconstructed sampling pools differ from the paper')
    result = {}
    for domain in ('qa', 'safety'):
        chosen = sorted(balanced_order(pools[domain], seed, domain)[:32])
        expected = frozen['subsets'][str(seed)][domain]
        if chosen != [r['source_index'] for r in expected]:
            raise ValueError('Reconstructed selection differs from the paper')
        result[domain] = chosen
    return result
