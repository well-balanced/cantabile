"""Render all MIDI_NAME_TO_CALLABLE songs to wav files organized by polyphony."""

import sys
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "env"))

from robopianist.music import library
import robopianist.music.constants as consts

OUT_DIR = Path(__file__).parent.parent / "env" / "audio" / "poly"


def max_polyphony(midi) -> int:
    times = sorted(
        [(n.start_time, 1) for n in midi.seq.notes]
        + [(n.end_time, -1) for n in midi.seq.notes]
    )
    cur = 0
    peak = 0
    for _, d in times:
        cur += d
        peak = max(peak, cur)
    return peak


def render(name: str, fn) -> None:
    print(f"  {name} ...", end=" ", flush=True)
    midi = fn()
    poly = max_polyphony(midi)
    out_path = OUT_DIR / str(poly) / f"{name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = midi.synthesize(sampling_rate=consts.SAMPLING_RATE)
    normalizer = float(np.iinfo(np.int16).max)
    waveform_int16 = np.array(np.asarray(waveform) * normalizer, dtype=np.int16)
    wavfile.write(str(out_path), consts.SAMPLING_RATE, waveform_int16)
    print(f"poly={poly}  {midi.seq.total_time:.1f}s  -> {out_path.relative_to(Path(__file__).parent.parent)}")


if __name__ == "__main__":
    print(f"Rendering {len(library.MIDI_NAME_TO_CALLABLE)} songs to {OUT_DIR}\n")
    for name, fn in library.MIDI_NAME_TO_CALLABLE.items():
        render(name, fn)
    print("\nDone.")
