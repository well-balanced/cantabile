"""CLI: render MIDI songs to WAV files.

Usage:
    python -m robopianist.music.render GymnopedieNo1 Berceuse
    python -m robopianist.music.render --list
    python -m robopianist.music.render --all -o /tmp/previews
"""

import argparse
import os
import numpy as np
import scipy.io.wavfile as wavfile

import robopianist.music as music
from robopianist.music import constants
from robopianist.music.midi_file import MidiFile


def flatten_velocity(mid: MidiFile, velocity: int = 64) -> MidiFile:
    from note_seq import music_pb2
    seq = music_pb2.NoteSequence()
    seq.CopyFrom(mid.seq)
    for note in seq.notes:
        note.velocity = velocity
    return MidiFile(seq=seq)


def style_velocity(mid: MidiFile, bias: int = 0, contrast: float = 1.0) -> MidiFile:
    from note_seq import music_pb2
    seq = music_pb2.NoteSequence()
    seq.CopyFrom(mid.seq)
    vels = np.array([n.velocity for n in seq.notes], dtype=float)
    mean = vels.mean()
    vels = mean + (vels - mean) * contrast + bias
    vels = np.clip(np.round(vels), 1, 127).astype(int)
    for note, v in zip(seq.notes, vels):
        note.velocity = int(v)
    return MidiFile(seq=seq)


def render(name: str, out_dir: str, flat: bool = False, styled: bool = False,
           bias: int = 0, contrast: float = 1.0) -> None:
    mid = music.load(name)
    if flat:
        mid = flatten_velocity(mid)
    elif styled:
        mid = style_velocity(mid, bias=bias, contrast=contrast)
    wav_data = mid.synthesize()
    arr = (wav_data / np.max(np.abs(wav_data)) * 32767).astype(np.int16)
    path = os.path.join(out_dir, f"{name}.wav")
    wavfile.write(path, constants.SAMPLING_RATE, arr)
    print(f"  {name:45s} {mid.duration:5.1f}s  {mid.n_notes:4d} notes  -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Render RoboPianist MIDIs to WAV")
    parser.add_argument("songs", nargs="*", help="Song names (e.g. GymnopedieNo1)")
    parser.add_argument("--all", action="store_true", help="Render all available songs")
    parser.add_argument("--list", action="store_true", help="List available song names")
    parser.add_argument("-o", "--out-dir", default="tmp/song_previews")
    parser.add_argument("--flat", action="store_true", help="Flatten all velocities to 64")
    parser.add_argument("--styled", action="store_true", help="Apply velocity bias and contrast")
    parser.add_argument("--bias", type=int, default=0, help="Velocity bias to add (default 0)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Velocity contrast around mean (default 1.0)")
    args = parser.parse_args()

    all_songs = music.PIG_MIDIS + music.ETUDE_MIDIS + music.DEBUG_MIDIS

    if args.list:
        for s in sorted(all_songs):
            print(s)
        return

    songs = all_songs if args.all else args.songs
    if not songs:
        parser.print_help()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Rendering {len(songs)} song(s) to {args.out_dir}/")
    for name in songs:
        try:
            render(name, args.out_dir, flat=args.flat, styled=args.styled,
                   bias=args.bias, contrast=args.contrast)
        except Exception as e:
            print(f"  ERR {name}: {e}")


if __name__ == "__main__":
    main()
