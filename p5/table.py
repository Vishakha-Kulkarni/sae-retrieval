"""Combined ablation table from row result JSONs (p5.evaluate): python -m p5.table --help

python -m p5.table --main B0.dev.json P1.dev.json L.dev.json --dl19 B0.dl19.json L.dl19.json \
    --dl20 B0.dl20.json L.dl20.json --output ablation       # writes ablation.md and ablation.tex

Rows keep their --main order, with B0 first. A piece that only has a run file evaluates it with
p5.evaluate (or p5.harness) first. † marks p < 0.05 against B0 (paired randomisation, from the row JSON,
uncorrected; P5's Holm-corrected claims come from p5.harness family and are reported in the text);
bold is the best value in each quality column. Missing values render as a dash. Rows within one split
must share a qrels file, and every p-value must come from a test against this B0's run.
"""
import argparse
import json
from pathlib import Path

BASELINE = 'B0'
# (header, split, metric view, metric); views come from p5.evaluate --subset names.
QUALITY = [('MRR@10', 'main', 'all', 'mrr_at_10'), ('R@1k', 'main', 'all', 'recall_at_1000'),
           ('Mis@1k MRR@10', 'main', 'mismatch_1000', 'mrr_at_10'), ('DL19 NDCG@10', 'dl19', 'all', 'ndcg_at_10'),
           ('DL20 NDCG@10', 'dl20', 'all', 'ndcg_at_10')]
COST = [('Concepts/q', lambda r: r.get('avg_query_concepts'), '{:.1f}'),
        ('Index GB', lambda r: r['index_size_bytes'] / 1e9 if 'index_size_bytes' in r else None, '{:.2f}'),
        ('ms/q', lambda r: r.get('latency_ms', {}).get('mean'), '{:.0f}')]
SIGNIFICANCE = .05


def check_split(rows, split):
    """Rows of one split by name; same qrels, and p-values against this split's B0 run."""
    by_name = {}
    for row in rows:
        if row['row'] in by_name:
            raise ValueError(f'{split}: row {row["row"]} given twice')
        by_name[row['row']] = row
    if len({r['input_sha256']['qrels'] for r in rows}) > 1:
        raise ValueError(f'{split}: rows were evaluated on different qrels files')
    baseline = by_name.get(BASELINE)
    for row in rows:
        tested = row.get('baseline_run_sha256')
        if 'p_value' in row and (baseline is None or tested is None or tested != baseline.get('run_sha256')):
            raise ValueError(f'{split}: {row["row"]} was tested against a baseline run other than {BASELINE}\'s')
    return by_name


def build_table(main, dl19=(), dl20=()):
    splits = {'main': check_split(main, 'main'), 'dl19': check_split(dl19, 'dl19'),
              'dl20': check_split(dl20, 'dl20')}
    names = [r['row'] for r in main]
    if BASELINE not in names:
        raise ValueError(f'the main split needs a {BASELINE} row')
    names.remove(BASELINE)
    table = []
    for name in [BASELINE, *names]:
        cells = []
        for _, split, view, metric in QUALITY:
            row = splits[split].get(name, {})
            value = row.get('metrics', {}).get(view, {}).get(metric)
            p = row.get('p_value', {}).get(view, {}).get(metric)
            cells.append({'value': value, 'significant': p is not None and p < SIGNIFICANCE})
        cells += [{'value': read(splits['main'][name])} for _, read, _ in COST]
        table.append({'row': name, 'cells': cells})
    for column in range(len(QUALITY)):
        values = [r['cells'][column]['value'] for r in table if r['cells'][column]['value'] is not None]
        for r in table:
            r['cells'][column]['best'] = bool(values) and r['cells'][column]['value'] == max(values)
    return table


def _render(table, bold, dagger, dash):
    formats = ['{:.3f}'] * len(QUALITY) + [f for _, _, f in COST]
    out = []
    for row in table:
        cells = []
        for cell, fmt in zip(row['cells'], formats):
            if cell['value'] is None:
                cells.append(dash)
                continue
            text = fmt.format(cell['value'])
            cells.append((bold(text) if cell.get('best') else text) + (dagger if cell.get('significant') else ''))
        out.append((row['row'], cells))
    return out


def render_markdown(table):
    headers = ['Row'] + [h for h, *_ in QUALITY] + [h for h, *_ in COST]
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for name, cells in _render(table, lambda t: f'**{t}**', '†', '-'):
        lines.append('| ' + ' | '.join([name, *cells]) + ' |')
    return '\n'.join(lines) + '\n'


def render_latex(table):
    headers = ['Row'] + [h for h, *_ in QUALITY] + [h for h, *_ in COST]
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{l' + 'r' * len(QUALITY) + '|' + 'r' * len(COST) + '}', r'\hline',
             ' & '.join(headers) + r' \\', r'\hline']
    for name, cells in _render(table, lambda t: rf'\textbf{{{t}}}', r'$^\dagger$', '--'):
        lines.append(' & '.join([name.replace('_', r'\_'), *cells]) + r' \\')
    lines += [r'\hline', r'\end{tabular}',
              r'\caption{Ablation on MS MARCO dev and TREC DL 2019/2020. Mis@1k: dev queries where BM25 finds no '
              r'relevant passage in its top 1000. $^\dagger$: $p<0.05$ against B0 (paired randomisation test, '
              r'uncorrected for multiple comparisons; corrected P5 tests in the text). '
              r'Bold: best per quality column.}', r'\label{tab:ablation}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main', nargs='+', required=True, help='Row JSONs on MS MARCO dev (or validation)')
    parser.add_argument('--dl19', nargs='*', default=[])
    parser.add_argument('--dl20', nargs='*', default=[])
    parser.add_argument('--output', required=True, help='Writes <output>.md and <output>.tex')
    args = parser.parse_args()
    load = lambda paths: [json.loads(Path(p).read_text()) for p in paths]
    table = build_table(load(args.main), load(args.dl19), load(args.dl20))
    for suffix, text in (('.md', render_markdown(table)), ('.tex', render_latex(table))):
        with open(args.output + suffix, 'x') as target:
            target.write(text)
    print(render_markdown(table), end='')


if __name__ == '__main__':
    main()
