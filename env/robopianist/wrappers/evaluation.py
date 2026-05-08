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

"""A wrapper for tracking episode statistics pertaining to music performance.

TODO(kevin):
- Look into `mir_eval` for metrics.
- Should sustain be a separate metric or should it just be applied to the note sequence
    as a whole?
"""

from collections import deque
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support

from robopianist.models.piano.midi_module import MAX_KEY_VEL as _MAX_KEY_VEL
from robopianist.models.piano.midi_module import MIN_KEY_VEL as _MIN_KEY_VEL


class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class MidiEvaluationWrapper(EnvironmentWrapper):
    """Track metrics related to musical performance.

    This wrapper calculates the precision, recall, and F1 score of the last `deque_size`
    episodes. The mean precision, recall and F1 score can be retrieved using
    `get_musical_metrics()`.

    By default, `deque_size` is set to 1 which means that only the current episode's
    statistics are tracked.
    """

    def __init__(self, environment: dm_env.Environment, deque_size: int = 1) -> None:
        super().__init__(environment)

        self._key_presses: List[np.ndarray] = []
        self._sustain_presses: List[np.ndarray] = []

        # Key press metrics.
        self._key_press_precisions: Deque[float] = deque(maxlen=deque_size)
        self._key_press_recalls: Deque[float] = deque(maxlen=deque_size)
        self._key_press_f1s: Deque[float] = deque(maxlen=deque_size)

        # Sustain metrics.
        self._sustain_precisions: Deque[float] = deque(maxlen=deque_size)
        self._sustain_recalls: Deque[float] = deque(maxlen=deque_size)
        self._sustain_f1s: Deque[float] = deque(maxlen=deque_size)

        # Per-onset velocity/onset-accuracy trace.
        self._episode_onset_trace: List[dict] = []
        self._all_onset_traces: Deque[List[dict]] = deque(maxlen=deque_size)

        # Episode-level onset accuracy counts (for mean across episodes).
        self._ep_gt_onsets: int = 0
        self._ep_hits: int = 0
        self._ep_misses: int = 0
        self._ep_fp: int = 0          # robot onsets on non-score-active keys
        self._ep_hold_rehits: int = 0  # robot onsets on already-held keys
        self._all_onset_accuracy: Deque[Dict[str, float]] = deque(maxlen=deque_size)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        task = self._environment.task
        piano = task.piano
        key_activation = piano.activation
        self._key_presses.append(key_activation.astype(np.float64))
        sustain_activation = piano.sustain_activation
        self._sustain_presses.append(sustain_activation.astype(np.float64))

        # Per-step onset tracking.
        prev = getattr(task, "_prev_activation", None)
        if prev is not None:
            t = task._t_idx - 1
            if 0 <= t < len(task._notes):
                self._record_onset_step(task, piano, key_activation, prev, t)

        if timestep.last():
            key_press_metrics = self._compute_key_press_metrics()
            self._key_press_precisions.append(key_press_metrics.precision)
            self._key_press_recalls.append(key_press_metrics.recall)
            self._key_press_f1s.append(key_press_metrics.f1)

            sustain_metrics = self._compute_sustain_metrics()
            self._sustain_precisions.append(sustain_metrics.precision)
            self._sustain_recalls.append(sustain_metrics.recall)
            self._sustain_f1s.append(sustain_metrics.f1)

            self._all_onset_traces.append(list(self._episode_onset_trace))
            gt_onsets = max(self._ep_gt_onsets, 1)
            robot_onsets = max(
                sum(1 for r in self._episode_onset_trace if r["robot_new_onset"]), 1
            )
            self._all_onset_accuracy.append({
                "hit_rate": self._ep_hits / gt_onsets,
                "miss_rate": self._ep_misses / gt_onsets,
                "fp_rate": self._ep_fp / robot_onsets,
                "hold_rehit_rate": self._ep_hold_rehits / robot_onsets,
            })

            self._key_presses = []
            self._sustain_presses = []
            self._episode_onset_trace = []
            self._ep_gt_onsets = 0
            self._ep_hits = 0
            self._ep_misses = 0
            self._ep_fp = 0
            self._ep_hold_rehits = 0
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        self._episode_onset_trace = []
        self._ep_gt_onsets = 0
        self._ep_hits = 0
        self._ep_misses = 0
        self._ep_fp = 0
        self._ep_hold_rehits = 0
        return self._environment.reset()

    def _record_onset_step(
        self,
        task,
        piano,
        activation: np.ndarray,
        prev_activation: np.ndarray,
        t: int,
    ) -> None:
        new_onset_keys = set(int(k) for k in np.flatnonzero(activation & ~prev_activation))

        gt_active_map: Dict[int, int] = {}
        gt_onset_map: Dict[int, int] = {}
        if hasattr(task, "_score_active_velocity_map"):
            gt_active_map = task._score_active_velocity_map(t)
            gt_onset_map = task._score_true_onset_velocity_map(t)
        else:
            gt_active_map = {note.key: int(note.velocity) for note in task._notes[t]}
            prev_keys = {note.key for note in task._notes[t - 1]} if t > 0 else set()
            gt_onset_map = {k: v for k, v in gt_active_map.items() if k not in prev_keys}

        gt_onset_keys = set(gt_onset_map)
        gt_active_keys = set(gt_active_map)

        # Onset accuracy counts for this step.
        self._ep_gt_onsets += len(gt_onset_keys)
        self._ep_hits += len(new_onset_keys & gt_onset_keys)
        self._ep_misses += len(gt_onset_keys - new_onset_keys)
        self._ep_fp += len(new_onset_keys - gt_active_keys)
        self._ep_hold_rehits += len(new_onset_keys & (gt_active_keys - gt_onset_keys))

        score_sustain = bool(task._sustains[t]) if 0 <= t < len(task._sustains) else False

        for key in new_onset_keys:
            qvel = float(piano.onset_velocities[key])
            robot_midi_vel = int(np.clip(
                (qvel - _MIN_KEY_VEL) / (_MAX_KEY_VEL - _MIN_KEY_VEL) * 126, 0, 126
            )) + 1
            gt_vel: Optional[int] = gt_onset_map.get(key)
            self._episode_onset_trace.append({
                "t_idx": t,
                "key_id": key,
                "robot_qvel": round(qvel, 4),
                "robot_midi_vel": robot_midi_vel,
                "gt_midi_vel": gt_vel if gt_vel is not None else -1,
                "score_active_midi_vel": gt_active_map.get(key, -1),
                "needed_qvel": (
                    round((float(gt_vel) - 1) / 126.0 * (_MAX_KEY_VEL - _MIN_KEY_VEL) + _MIN_KEY_VEL, 4)
                    if gt_vel is not None else None
                ),
                "qvel_gap": (
                    round(qvel - ((float(gt_vel) - 1) / 126.0 * (_MAX_KEY_VEL - _MIN_KEY_VEL) + _MIN_KEY_VEL), 4)
                    if gt_vel is not None else None
                ),
                "error": (robot_midi_vel - gt_vel) if gt_vel is not None else None,
                "score_key_active": key in gt_active_keys,
                "gt_is_true_onset": key in gt_onset_keys,
                "score_sustain": score_sustain,
                "robot_new_onset": True,
                "matched": gt_vel is not None,
            })


    def get_velocity_metrics(self) -> Dict[str, float]:
        """Returns velocity and onset accuracy statistics over the last `deque_size` episodes."""
        trace = [row for ep in self._all_onset_traces for row in ep]
        matched = [row for row in trace if row["matched"]]

        result: Dict[str, float] = {}

        if matched:
            robot_arr = np.array([row["robot_midi_vel"] for row in matched], dtype=np.float64)
            gt_arr = np.array([row["gt_midi_vel"] for row in matched], dtype=np.float64)
            errors = robot_arr - gt_arr
            result.update({
                "velocity_mae": float(np.mean(np.abs(errors))),
                "velocity_bias": float(np.mean(errors)),
                "velocity_std": float(np.std(robot_arr)),
                "match_rate": len(matched) / max(len(trace), 1),
            })

        if self._all_onset_accuracy:
            for key in ("hit_rate", "miss_rate", "fp_rate", "hold_rehit_rate"):
                result[f"onset_{key}"] = float(
                    np.mean([ep[key] for ep in self._all_onset_accuracy])
                )

            recall = float(np.mean([ep["hit_rate"] for ep in self._all_onset_accuracy]))
            precision = 1.0 - float(np.mean([ep["fp_rate"] for ep in self._all_onset_accuracy]))
            onset_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            result["onset_f1"] = onset_f1

            vel_accuracy = 1.0 - result.get("velocity_mae", 127.0) / 127.0
            denom = onset_f1 + vel_accuracy
            result["expressive_f1"] = 2 * onset_f1 * vel_accuracy / denom if denom > 0 else 0.0

        return result

    def get_episode_velocity_trace(self) -> List[dict]:
        """Returns per-onset rows for the last episode(s), suitable for wandb.Table."""
        return [row for ep in self._all_onset_traces for row in ep]

    def get_musical_metrics(self) -> Dict[str, float]:
        """Returns the mean precision/recall/F1 over the last `deque_size` episodes."""
        if not self._key_press_precisions:
            raise ValueError("No episode metrics available yet.")

        def _mean(seq: Sequence[float]) -> float:
            return sum(seq) / len(seq)

        return {
            "precision": _mean(self._key_press_precisions),
            "recall": _mean(self._key_press_recalls),
            "f1": _mean(self._key_press_f1s),
            "sustain_precision": _mean(self._sustain_precisions),
            "sustain_recall": _mean(self._sustain_recalls),
            "sustain_f1": _mean(self._sustain_f1s),
        }

    # Helper methods.

    def _compute_key_press_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for key presses over the episode."""
        # Get the ground truth key presses.
        note_seq = self._environment.task._notes
        ground_truth = []
        for notes in note_seq:
            presses = np.zeros((self._environment.task.piano.n_keys,), dtype=np.float64)
            keys = [note.key for note in notes]
            presses[keys] = 1.0
            ground_truth.append(presses)

        # Deal with the case where the episode gets truncated due to a failure. In this
        # case, the length of the key presses will be less than or equal to the length
        # of the ground truth.
        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._key_presses)]

        assert len(ground_truth) == len(self._key_presses)

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._key_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)

    def _compute_sustain_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for sustain presses over the episode."""
        # Get the ground truth sustain presses.
        ground_truth = [
            np.atleast_1d(v).astype(float) for v in self._environment.task._sustains
        ]

        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._sustain_presses)]

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._sustain_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)
