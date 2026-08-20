"""Checks whether the GT velocities for the 3 evaluated songs are real,
expressive per-note data or a quantized/constant placeholder.

Directly answers Reviewer vmh3's question ("How often are the per-note
velocity targets available in datasets for training?") for the specific
songs used in the paper's results, and is a prerequisite sanity check for
comparing against the MAESTRO human-human anchor (compute_human_mae.py):
if these were constant-velocity MIDI, an MAE anchor would be meaningless.

Usage: .venv/bin/python rl/human_mae/inspect_song_velocities.py
"""

import json
from pathlib import Path

import numpy as np
import pretty_midi
from note_seq import music_pb2

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "env" / "robopianist" / "music" / "data"

SONGS = {
    "TwinkleTwinkleRousseau": ("mid", DATA_DIR / "rousseau" / "twinkle-twinkle-trimmed.mid"),
    "NocturneRousseau": ("mid", DATA_DIR / "rousseau" / "nocturne-trimmed.mid"),
    "ClairDeLune": ("proto", DATA_DIR / "pig_single_finger" / "clair_de_lune-1.proto"),
}


def load_velocities(kind: str, path: Path):
    if kind == "mid":
        pm = pretty_midi.PrettyMIDI(str(path))
        return np.array([n.velocity for inst in pm.instruments for n in inst.notes])
    else:
        seq = music_pb2.NoteSequence()
        with open(path, "rb") as f:
            seq.ParseFromString(f.read())
        return np.array([n.velocity for n in seq.notes])


def main():
    results = {}
    for name, (kind, path) in SONGS.items():
        if not path.exists():
            print(f"{name}: MISSING at {path} (PIG dataset may need `robopianist preprocess` first)")
            continue
        vels = load_velocities(kind, path)
        stats = {
            "source": "rousseau (real YouTube performance, permission granted, see data/README.md)"
            if kind == "mid" else "PIG dataset (real performance-derived fingering corpus)",
            "n_notes": int(len(vels)),
            "n_unique_velocities": int(len(set(vels.tolist()))),
            "mean": round(float(vels.mean()), 2),
            "std": round(float(vels.std()), 2),
            "min": int(vels.min()),
            "max": int(vels.max()),
        }
        results[name] = stats
        print(f"{name}: {stats}")

    out_path = Path(__file__).parent / "results" / "song_velocity_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
