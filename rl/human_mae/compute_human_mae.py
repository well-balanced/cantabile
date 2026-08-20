"""Computes an inter-performer (human-human) velocity MAE anchor from MAESTRO.

Motivation: the paper reports velocity MAE (26.4 -> 7.0) with no absolute
reference for what MAE value is musically meaningful (Reviewer vmh3). This
script estimates that reference empirically: how much do two different human
performances of the *same* piece naturally differ in per-note velocity? That
gap is a lower bound on what any GT-tracking system should be judged against.

Method
------
1. MAESTRO (github.com/magenta, CC BY-NC-SA per-piece) contains many pieces
   performed more than once (different years of the International
   Piano-e-Competition). MAESTRO does not expose performer identity, so we
   group by `canonical_title` and treat same-title/different-year pairs as
   a proxy for "different performer" (see Limitations in README.md).
2. For each pair of performances of the same title, we cannot compare notes
   by timestamp (tempo differs between performances), so we align by pitch
   *order* instead: both performances play (approximately) the same sequence
   of pitches, just at different speeds/dynamics. difflib.SequenceMatcher
   finds the longest common subsequence of pitches between the two note
   lists (sorted by onset time), which gives us a note-to-note correspondence
   without needing timing information.
3. MAE is computed over matched note pairs as |velocity_a - velocity_b|,
   on the raw MIDI velocity scale (0-127) -- the same scale used by
   `env/robopianist/wrappers/evaluation.py`'s `velocity_mae`.

Usage
-----
    .venv/bin/python rl/human_mae/compute_human_mae.py

Requires `pretty_midi` (already a project dependency). Downloads MAESTRO
MIDI-only archive (~58MB) into --cache-dir on first run (gitignored, not
committed).
"""

import argparse
import csv
import difflib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pretty_midi

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def ensure_maestro(cache_dir: Path) -> Path:
    """Downloads + extracts MAESTRO MIDI-only archive if not already cached."""
    maestro_dir = cache_dir / "maestro-v3.0.0"
    if (maestro_dir / "maestro-v3.0.0.csv").exists():
        return maestro_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "maestro-v3.0.0-midi.zip"
    print(f"Downloading MAESTRO to {zip_path} ...")
    urlretrieve(MAESTRO_URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    return maestro_dir


def load_notes(maestro_dir: Path, midi_filename: str):
    """Returns [(onset_time, pitch, velocity), ...] sorted by onset time."""
    pm = pretty_midi.PrettyMIDI(str(maestro_dir / midi_filename))
    notes = [(n.start, n.pitch, n.velocity) for inst in pm.instruments for n in inst.notes]
    notes.sort(key=lambda t: t[0])
    return notes


def match_and_diff(notes_a, notes_b):
    """Aligns two note lists by pitch order (difflib LCS) and returns
    |velocity_a - velocity_b| for each matched pair, plus both lengths."""
    pitches_a = [n[1] for n in notes_a]
    pitches_b = [n[1] for n in notes_b]
    sm = difflib.SequenceMatcher(None, pitches_a, pitches_b, autojunk=False)
    diffs = []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            va = notes_a[block.a + k][2]
            vb = notes_b[block.b + k][2]
            diffs.append(abs(va - vb))
    return diffs, len(pitches_a), len(pitches_b)


def build_pairs(rows, cap_per_title: int, seed: int):
    """Groups MAESTRO rows by canonical_title and selects candidate pairs,
    prioritizing different-year pairs (more likely different performers)."""
    by_title = defaultdict(list)
    for r in rows:
        by_title[r["canonical_title"]].append(r)
    multi = {t: v for t, v in by_title.items() if len(v) >= 2}

    rng = random.Random(seed)
    pairs = []
    for title, perfs in multi.items():
        idxs = list(range(len(perfs)))
        rng.shuffle(idxs)
        made = 0
        for i in range(len(perfs)):
            for j in range(i + 1, len(perfs)):
                if made >= cap_per_title:
                    break
                a, b = perfs[i], perfs[j]
                pairs.append((title, a, b, a["year"] != b["year"]))
                made += 1
            if made >= cap_per_title:
                break
    # Different-year pairs first (more likely distinct performers).
    pairs.sort(key=lambda p: not p[3])
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="rl/tmp/maestro",
                     help="Where to download/cache MAESTRO (gitignored).")
    ap.add_argument("--max-pairs", type=int, default=350)
    ap.add_argument("--cap-per-title", type=int, default=6,
                     help="Max pairs sampled from any single title (bounds combinatorics on "
                          "titles with many repeat performances).")
    ap.add_argument("--min-matched", type=int, default=10,
                     help="Discard pairs with fewer matched notes than this.")
    ap.add_argument("--min-match-frac", type=float, default=0.3,
                     help="Discard pairs where matched/min(len_a,len_b) is below this "
                          "(likely a different movement/excerpt, not a true repeat).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="rl/human_mae/results")
    args = ap.parse_args()

    maestro_dir = ensure_maestro(Path(args.cache_dir))

    with open(maestro_dir / "maestro-v3.0.0.csv") as f:
        rows = list(csv.DictReader(f))

    pairs = build_pairs(rows, args.cap_per_title, args.seed)[: args.max_pairs]
    print(f"Selected {len(pairs)} performance pairs across {len(set(p[0] for p in pairs))} titles")

    notes_cache = {}

    def get_notes(fn):
        if fn not in notes_cache:
            notes_cache[fn] = load_notes(maestro_dir, fn)
        return notes_cache[fn]

    pair_results = []
    all_diffs = []
    skipped = 0
    for title, a, b, diff_year in pairs:
        try:
            notes_a = get_notes(a["midi_filename"])
            notes_b = get_notes(b["midi_filename"])
        except Exception:
            skipped += 1
            continue
        if len(notes_a) < args.min_matched or len(notes_b) < args.min_matched:
            skipped += 1
            continue
        diffs, na, nb = match_and_diff(notes_a, notes_b)
        match_frac = len(diffs) / min(na, nb)
        if len(diffs) < args.min_matched or match_frac < args.min_match_frac:
            skipped += 1
            continue
        pair_results.append({
            "title": title,
            "year_a": a["year"], "year_b": b["year"],
            "midi_a": a["midi_filename"], "midi_b": b["midi_filename"],
            "diff_year": diff_year,
            "n_matched": len(diffs),
            "match_frac": round(match_frac, 4),
            "mae": round(float(np.mean(diffs)), 4),
        })
        all_diffs.extend(diffs)

    print(f"Usable pairs: {len(pair_results)} (skipped {skipped})")

    pair_maes = np.array([p["mae"] for p in pair_results])
    all_diffs = np.array(all_diffs)
    summary = {
        "n_pairs": len(pair_results),
        "n_titles": len(set(p["title"] for p in pair_results)),
        "n_skipped": skipped,
        "n_matched_notes_pooled": int(len(all_diffs)),
        "per_pair_mae_mean": round(float(pair_maes.mean()), 3),
        "per_pair_mae_median": round(float(np.median(pair_maes)), 3),
        "per_pair_mae_std": round(float(pair_maes.std()), 3),
        "per_pair_mae_p10": round(float(np.percentile(pair_maes, 10)), 3),
        "per_pair_mae_p90": round(float(np.percentile(pair_maes, 90)), 3),
        "pooled_mae": round(float(all_diffs.mean()), 3),
        "args": vars(args),
    }
    print(json.dumps(summary, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "human_mae_pairs.json", "w") as f:
        json.dump(pair_results, f, indent=2)
    with open(out_dir / "human_mae_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
