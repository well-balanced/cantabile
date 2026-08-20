"""Fetch eval metrics from wandb and save per-run JSONs + aggregated table.

Running runs are included but saved with _INCOMPLETE suffix.
Usage: python fetch_wandb.py [--out-dir ../tmp/wandb_data]
"""

import json
import csv
import argparse
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
import wandb

SONGS = ['twinkle', 'clair', 'nocturne']
SEEDS = [42, 43, 44]

CONDITIONS = {
    'Baseline':       'base',
    'Vel-Aware':      'vel_aware',
    'Base+Res':       'base_residual',
    'Vel-Aware+Res':  'vel_aware_residual',
}

METRICS = [
    'eval/f1',
    'eval/onset_f1',
    'eval/onset_hit_rate',
    'eval/onset_precision',
    'eval/velocity_mae',
    'eval/velocity_correlation',
    'eval/match_rate',
]
SHORT = {m: m.split('/')[1] for m in METRICS}

VBAR = {'twinkle': 39.0, 'clair': 43.8, 'nocturne': 43.0}


def run_name(method: str, song: str, seed: int) -> str:
    if song == 'nocturne' and method in ('vel_aware', 'vel_aware_residual'):
        return f'{method}-{song}-s{seed}-v03-o10'
    return f'{method}-{song}-s{seed}'


def dynamics_score(onset_f1, mae, vbar):
    """D = F1_onset * max(0, 1 - MAE/vbar) -- main.tex Eq. dynamics-score.

    Requires eval/onset_precision to be computed correctly (counting every
    unmatched robot onset as FP, not just strikes on score-inactive keys --
    see env/robopianist/wrappers/evaluation.py). Runs logged before that fix
    have an inflated onset_f1 and therefore an inflated D.
    """
    return onset_f1 * max(0.0, 1.0 - mae / vbar)


def recall_weighted_mae(recall, mae_matched, vbar):
    """MAE_rw = R_onset * MAE_matched + (1 - R_onset) * vbar -- same raw

    MIDI-velocity units as matched-only MAE (unlike `dynamics_score`, which
    is a normalized [0,1] composite), so it's directly comparable to e.g. a
    human-human MAE anchor. See rl/human_mae/recall_weighted_mae.py.
    """
    return recall * mae_matched + (1 - recall) * vbar


def ci95(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    if len(v) == 1:
        return v[0], None
    mean = np.mean(v)
    se = scipy_stats.sem(v)
    t = scipy_stats.t.ppf(0.975, df=len(v) - 1)
    return mean, t * se


MIN_PROGRESS = 0.4
MAX_STEPS = 5_000_000


def is_complete(state: str) -> bool:
    return state == 'finished'


def is_usable(entry: dict) -> bool:
    if entry['complete']:
        return True
    step = entry.get('step') or 0
    return step / MAX_STEPS >= MIN_PROGRESS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='../tmp/wandb_data')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    print('Fetching runs from wandb...')
    all_runs = list(api.runs('cantabile/cantabile', order='-created_at'))
    run_map = {}
    for r in all_runs:
        if r.name not in run_map:
            run_map[r.name] = r
    print(f'  {len(run_map)} unique run names found.\n')

    # ── Collect per-run entries ───────────────────────────────────────────────
    data = {}
    for cond_name, tag in CONDITIONS.items():
        data[cond_name] = {}
        for song in SONGS:
            data[cond_name][song] = {}
            for seed in SEEDS:
                name = run_name(tag, song, seed)
                r = run_map.get(name)
                entry = {'run_name': name, 'state': 'NOT_FOUND', 'complete': False}
                if r is not None:
                    entry['state'] = r.state
                    entry['complete'] = is_complete(r.state)
                    entry['step'] = r.summary.get('_step', None)
                    for m in METRICS:
                        v = r.summary.get(m)
                        entry[SHORT[m]] = round(v, 4) if isinstance(v, float) else None
                else:
                    for m in METRICS:
                        entry[SHORT[m]] = None
                data[cond_name][song][seed] = entry

    # ── Save per-run JSONs ────────────────────────────────────────────────────
    for song in SONGS:
        for seed in SEEDS:
            seed_dir = out_dir / song / f's{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)
            for cond_name in CONDITIONS:
                entry = data[cond_name][song][seed]
                tag_safe = cond_name.replace(' ', '_').replace('+', 'p')
                suffix = '' if entry['complete'] else '_INCOMPLETE'
                fname = seed_dir / f'{tag_safe}{suffix}.json'
                for ext in ('', '_INCOMPLETE'):
                    old = seed_dir / f'{tag_safe}{ext}.json'
                    if old.exists() and str(old) != str(fname):
                        old.unlink()
                with open(fname, 'w') as f:
                    json.dump(entry, f, indent=2)
    print(f'Per-run JSONs saved to {out_dir}/{{song}}/s{{seed}}/\n')

    # ── Status grid ───────────────────────────────────────────────────────────
    print('=== RUN STATUS GRID ===')
    for cond_name, tag in CONDITIONS.items():
        print(f'\n{cond_name}:')
        for song in SONGS:
            parts = []
            for seed in SEEDS:
                e = data[cond_name][song][seed]
                step = e.get('step', '?')
                s = e['state']
                sym = {'finished': 'DONE', 'running': f'run@{step}',
                       'killed': 'KILLED', 'crashed': 'CRASH',
                       'NOT_FOUND': '????'}.get(s, s)
                parts.append(f's{seed}:{sym}')
            print(f'  {song:<12} ' + '  '.join(f'{p:<22}' for p in parts))

    # ── Aggregated table ──────────────────────────────────────────────────────
    metric_keys = [SHORT[m] for m in METRICS]
    # Derived metrics, computed per-run from the raw METRICS above:
    #   D      = Dynamics Score (main.tex Eq. dynamics-score), normalized [0,1]
    #   mae_rw = recall-weighted MAE, same raw MIDI-velocity units as `velocity_mae`
    #            (matched-only) -- report both side by side, not one or the other.
    derived_keys = ['D', 'mae_rw']

    def derived(e, vbar):
        out = {}
        if e.get('onset_f1') is not None and e.get('velocity_mae') is not None:
            out['D'] = dynamics_score(e['onset_f1'], e['velocity_mae'], vbar)
        if e.get('onset_hit_rate') is not None and e.get('velocity_mae') is not None:
            out['mae_rw'] = recall_weighted_mae(e['onset_hit_rate'], e['velocity_mae'], vbar)
        return out

    print('\n\n=== AGGREGATED TABLE (song-balanced, complete-preferred) ===')
    hdr = f"{'Condition':<20}  " + '  '.join(f'{k:<30}' for k in metric_keys + derived_keys)
    print(hdr)
    print('-' * (22 + 32 * (len(metric_keys) + len(derived_keys))))

    agg_rows = []
    for cond_name in CONDITIONS:
        song_means = {k: [] for k in metric_keys + derived_keys}
        song_labels = []

        for song in SONGS:
            all_usable = [
                data[cond_name][song][seed]
                for seed in SEEDS
                if is_usable(data[cond_name][song][seed])
                and data[cond_name][song][seed].get('f1') is not None
            ]

            if not all_usable:
                continue

            entries = all_usable
            n_complete = sum(1 for e in entries if e['complete'])
            n_running  = len(entries) - n_complete
            label = f'{n_complete}c' if n_running == 0 else f'{n_complete}c{n_running}r'
            song_labels.append(f'{song[:3]}:{label}')
            for k in metric_keys:
                vals = [e[k] for e in entries if e.get(k) is not None]
                if vals:
                    song_means[k].append(np.mean(vals))
            for dk in derived_keys:
                dvals = [derived(e, VBAR[song])[dk] for e in entries if dk in derived(e, VBAR[song])]
                if dvals:
                    song_means[dk].append(np.mean(dvals))

        cells = []
        row = {'Condition': cond_name}
        coverage = f'({",".join(song_labels)})' if song_labels else ''
        for k in metric_keys + derived_keys:
            mean, hw = ci95(song_means[k])
            if mean is None:
                cells.append(f'{"--":<28}')
            elif hw is None:
                cells.append(f'{mean:.3f} {coverage:<20}')
            else:
                cells.append(f'{mean:.3f} ± {hw:.3f} {coverage:<8}')
            row[k] = f'{mean:.3f}' if mean is not None else '--'
            row[f'{k}_ci'] = f'{hw:.3f}' if hw is not None else '--'
        print(f'{cond_name:<20}  ' + '  '.join(cells))
        agg_rows.append(row)

    # ── Per-song table ────────────────────────────────────────────────────────
    print('\n\n=== PER-SONG TABLE ===')
    for song in SONGS:
        print(f'\n-- {song.upper()} --')
        print(hdr)
        print('-' * (22 + 32 * (len(metric_keys) + len(derived_keys))))
        for cond_name in CONDITIONS:

            all_usable = [
                data[cond_name][song][seed]
                for seed in SEEDS
                if is_usable(data[cond_name][song][seed])
                and data[cond_name][song][seed].get('f1') is not None
            ]
            entries = all_usable
            n_c = sum(1 for e in entries if e['complete'])
            n_r = len(entries) - n_c
            if not entries:
                tag = ''
            elif n_r == 0 and n_c == len(SEEDS):
                tag = ''
            elif n_r == 0:
                tag = f'({n_c}c)'
            else:
                tag = f'({n_c}c{n_r}r)'

            cells = []
            for k in metric_keys:
                vals = [e[k] for e in entries if e.get(k) is not None]
                mean, hw = ci95(vals)
                if mean is None:
                    cells.append(f'{"--":<28}')
                elif hw is None:
                    cells.append(f'{mean:.3f}{tag:<23}')
                else:
                    cells.append(f'{mean:.3f} ± {hw:.3f}{tag:<15}')
            for dk in derived_keys:
                dvals = [derived(e, VBAR[song])[dk] for e in entries if dk in derived(e, VBAR[song])]
                d_mean, d_hw = ci95(dvals)
                if d_mean is None:
                    cells.append(f'{"--":<28}')
                elif d_hw is None:
                    cells.append(f'{d_mean:.3f}{tag:<23}')
                else:
                    cells.append(f'{d_mean:.3f} ± {d_hw:.3f}{tag:<15}')
            print(f'{cond_name:<20}  ' + '  '.join(cells))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / 'aggregated.csv'
    fieldnames = ['Condition'] + [f for k in metric_keys + derived_keys for f in (k, f'{k}_ci')]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)
    print(f'\nAggregated CSV → {csv_path}')


if __name__ == '__main__':
    main()
