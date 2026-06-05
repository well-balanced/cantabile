# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piano sound module."""

from typing import Callable, List, Optional

import numpy as np
from dm_control import mjcf

from robopianist.models.piano import piano_constants
from robopianist.music import midi_file, midi_message

# ---------------------------------------------------------------------------
# MAX_KEY_VEL = 8.0 rad/s
#
# Summary: piano-intrinsic ceiling for key joint angular velocity, mapping
# linearly to MIDI velocity 127.  Derived from v_hammer_max = 6.0 m/s
# (upper bound of normal dynamic range, Russell & Rossing 1998) divided by
# key length 0.15 m and lever ratio ~5.0 (Askenfelt & Jansson 1990).
# Independent of robot hardware.
# ---------------------------------------------------------------------------
#
# This constant is an intrinsic property of the simulated piano, not of the
# robot. It maps piano key joint angular velocity (rad/s) to MIDI velocity
# (1–127) via a linear normalisation:
#
#   midi_velocity = clip(qvel / MAX_KEY_VEL * 126, 0, 126) + 1
#
# Being piano-intrinsic means: if a given robot cannot reach MAX_KEY_VEL, it
# simply cannot produce fff dynamics — analogous to a physically weak human
# pianist. That is a constraint of the embodiment, not of the mapping.
#
# Step 1 — Key-tip linear velocity from joint angular velocity
# ─────────────────────────────────────────────────────────────
#   The key rotates about its rear hinge.  The MuJoCo joint position is the
#   rotation angle θ (rad), so the joint velocity qvel (rad/s) gives a
#   linear tip velocity of (small-angle approximation; max key angle ~3.8°):
#
#     v_tip = qvel × L_key
#
#   where L_key = WHITE_KEY_LENGTH = 0.15 m  (piano_constants.py).
#
# Step 2 — Hammer velocity from key-tip velocity (lever ratio)
# ─────────────────────────────────────────────────────────────
#   Inside the piano action the key acts as a lever driving the hammer via
#   the wippen/repetition mechanism.  Typical upright piano regulation values
#   of ~10 mm key dip and ~44–45 mm hammer stroke (Reblitz, 1993, pp. 158–170)
#   imply a displacement amplification of approximately 4.5–5.  Experimental
#   studies of piano action kinematics further confirm this order-of-magnitude
#   estimate (Askenfelt & Jansson, 1990, 1991).  Under rigid-body kinematics
#   the displacement ratio equals the velocity ratio:
#
#     r_lever ≈ 5.0
#
#   References:
#     Reblitz, A. A. (1993). Piano Servicing, Tuning, and Rebuilding (2nd ed.).
#     Vestal Press, pp. 158–170.
#
#     Askenfelt, A. & Jansson, E. V. (1990). "From touch to string
#     vibrations: The initial course of the piano tone." Journal of the
#     Acoustical Society of America, 88(1), 52–63.
#
#   Therefore:
#     v_hammer = v_tip × r_lever = qvel × 0.15 × 5.0 = qvel × 0.75
#
# Step 3 — Maximum hammer velocity from acoustic measurements
# ─────────────────────────────────────────────────────────────
#   Russell & Rossing (1998) report "hammer speeds of 1 to 6 m/s span the
#   normal dynamic range of the piano"; Goebl, Bresin & Galembo (2005)
#   measured a minimum of ~0.1 m/s at the softest playable dynamic:
#
#     v_hammer ∈ [~0.1, ~6.0] m/s
#
#   References:
#     Russell, D. A. & Rossing, T. D. (1998). "Testing the nonlinearity of
#     piano hammers using residual shock spectra." Acta Acustica, 84, 967–975.
#
#     Goebl, W., Bresin, R. & Galembo, A. (2005). "Touch and temporal
#     behavior of grand piano actions." JASA 118(2), 1154–1165.
#
#   6.0 m/s is the upper bound of the measured normal dynamic range
#   (Russell & Rossing 1998), which we map to MIDI velocity 127 (fff per
#   MIDI 1.0 specification).
#
# Step 4 — Back-calculating MAX_KEY_VEL
# ──────────────────────────────────────
#   Setting v_hammer_max = 6.0 m/s and solving for qvel:
#
#     MAX_KEY_VEL = v_hammer_max / (L_key × r_lever)
#                = 6.0 / (0.15 × 5.0)
#                = 6.0 / 0.75
#                = 8.0 rad/s
#
# Empirical consistency check
# ────────────────────────────
#   RL agents trained with velocity reward (robopianist-expressive) show
#   onset qvel p90 ≈ 3.0–3.5 rad/s across 1M-step experiments, mapping to
#   MIDI velocity ~48–56 (mp–mf range).  This is consistent with the dynamic
#   distribution of the training MIDI data (GT notes are predominantly in the
#   mp–mf range) and indicates that the 8.0 rad/s ceiling does not cause
#   systematic saturation.
#
# Relationship to MIDI velocity specification
# ────────────────────────────────────────────
#   Commercial piano capture systems (e.g. Yamaha Disklavier, Bosendorfer
#   CEUS) measure final hammer velocity optically.  While the exact internal
#   mapping is proprietary (Goebl & Bresin, 2003, note no public disclosure),
#   a linear hammer-velocity-to-MIDI mapping is the standard physical
#   assumption in the literature, consistent with Palmer & Brown's (1991)
#   finding that acoustic amplitude is linearly proportional to hammer
#   velocity.  We adopt this linear convention; deviations at extreme
#   dynamics (compression reported by Goebl & Bresin, 2003) are noted as a
#   limitation.  A perceptually-weighted mapping — motivated by Dannenberg's
#   (2006) finding that synthesiser output amplitude scales as ~velocity² —
#   was considered but not adopted: that relationship describes the
#   synthesiser playback side (MIDI→amplitude), not the physical capture side
#   (hammer velocity→MIDI), and applying it here would introduce systematic
#   bias against Disklavier-captured GT velocities.
#
#   Additional references:
#     Goebl, W. & Bresin, R. (2003). "Measurement and reproduction accuracy
#     of computer-controlled grand pianos." JASA 114(4), 2273–2283.
#     Palmer, C. & Brown, J. C. (1991). "Investigations in the amplitude of
#     sounded piano tones." JASA 90(1), 60–66.
#     Dannenberg, R. B. (2006). "The interpretation of MIDI velocity." ICMC.
#
# Dead-zone offset (QVEL_MIN)
# ───────────────────────────
#   The RoboPianist piano model activates a key when its joint angle comes
#   within 0.5° of the fully-pressed position (_KEY_THRESHOLD in piano.py).
#   This threshold is intentionally kept unchanged from the original
#   RoboPianist codebase to preserve fair comparison with baseline models.
#
#   Due to the spring restoring force, reaching this threshold requires a
#   minimum onset qvel.  Measured via impulse rollout on the isolated piano
#   model (physics.data.qvel injection), the minimum onset qvel at the
#   activation moment is approximately 0.39 rad/s.  Keys pressed below this
#   speed simply do not activate regardless of intent.
#
#   The mapping therefore uses [QVEL_MIN, MAX_KEY_VEL] → [1, 127]:
#
#     midi_velocity = clip((qvel - QVEL_MIN) / (MAX_KEY_VEL - QVEL_MIN)
#                          * 126, 0, 126) + 1
#
#   This mirrors how a real piano maps its softest physically playable note
#   to MIDI 1, rather than leaving an unreachable dead zone at the bottom.
#
# Single source of truth — import both constants wherever the conversion is
# needed (piano_with_shadow_hands.py, wrappers/evaluation.py, etc.).
# ---------------------------------------------------------------------------
MAX_KEY_VEL: float = 8.0   # 6.0 m/s / (0.15 m × 5.0) = 8.0 rad/s
MIN_KEY_VEL: float = 0.39    # minimum onset qvel to activate a key (rad/s)


class MidiModule:
    """The piano sound module.

    It is responsible for tracking the state of the piano keys and generating
    corresponding MIDI messages. The MIDI messages can be used with a synthesizer
    to produce sound.
    """

    def __init__(self) -> None:
        self._note_on_callback: Optional[Callable[[int, int], None]] = None
        self._note_off_callback: Optional[Callable[[int], None]] = None
        self._sustain_on_callback: Optional[Callable[[], None]] = None
        self._sustain_off_callback: Optional[Callable[[], None]] = None

    def initialize_episode(self, physics: mjcf.Physics) -> None:
        del physics  # Unused.

        self._prev_activation = np.zeros(piano_constants.NUM_KEYS, dtype=bool)
        self._prev_sustain_activation = np.zeros(1, dtype=bool)
        self._midi_messages: List[List[midi_message.MidiMessage]] = []

    def after_substep(
        self,
        physics: mjcf.Physics,
        activation: np.ndarray,
        sustain_activation: np.ndarray,
        key_velocities: Optional[np.ndarray] = None,
    ) -> None:
        # Sanity check dtype since we use bitwise operators.
        assert activation.dtype == bool
        assert sustain_activation.dtype == bool

        timestep_events: List[midi_message.MidiMessage] = []
        message: midi_message.MidiMessage

        state_change = activation ^ self._prev_activation
        sustain_change = sustain_activation ^ self._prev_sustain_activation

        # Note on events.
        for key_id in np.flatnonzero(state_change & ~self._prev_activation):
            message = midi_message.NoteOn(
                note=midi_file.key_number_to_midi_number(key_id),
                velocity = 127 if key_velocities is None else int(np.clip(
                    (key_velocities[key_id] - MIN_KEY_VEL) / (MAX_KEY_VEL - MIN_KEY_VEL) * 126,
                    0, 126
                )) + 1,
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_on_callback is not None:
                self._note_on_callback(message.note, message.velocity)

        # Note off events.
        for key_id in np.flatnonzero(state_change & ~activation):
            message = midi_message.NoteOff(
                note=midi_file.key_number_to_midi_number(key_id),
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_off_callback is not None:
                self._note_off_callback(message.note)

        # Sustain pedal events.
        if sustain_change & ~self._prev_sustain_activation:
            timestep_events.append(midi_message.SustainOn(time=physics.data.time))
            if self._sustain_on_callback is not None:
                self._sustain_on_callback()
        if sustain_change & ~sustain_activation:
            timestep_events.append(midi_message.SustainOff(time=physics.data.time))
            if self._sustain_off_callback is not None:
                self._sustain_off_callback()

        self._midi_messages.append(timestep_events)
        self._prev_activation = activation.copy()
        self._prev_sustain_activation = sustain_activation.copy()

    def get_latest_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns the MIDI messages generated in the last substep."""
        return self._midi_messages[-1]

    def get_all_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns a list of all MIDI messages generated during the episode."""
        return [message for timestep in self._midi_messages for message in timestep]

    # Callbacks for synthesizer events.

    def register_synth_note_on_callback(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Registers a callback for note on events."""
        self._note_on_callback = callback

    def register_synth_note_off_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """Registers a callback for note off events."""
        self._note_off_callback = callback

    def register_synth_sustain_on_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Registers a callback for sustain pedal on events."""
        self._sustain_on_callback = callback

    def register_synth_sustain_off_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Registers a callback for sustain pedal off events."""
        self._sustain_off_callback = callback
