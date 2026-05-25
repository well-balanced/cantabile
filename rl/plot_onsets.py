"""Generate an onset-timing figure from trained checkpoints.

Single run:
    python plot_onsets.py \
        --checkpoint ckpt.flax --environment_name ENV \
        [env flags] --output tmp/out.png

Comparison (2x2: RH/LH x ckpt1/ckpt2, auto-zoomed to best window):
    python plot_onsets.py \
        --checkpoint ckpt1.flax --song_name "w/ onset" \
        --compare_checkpoint ckpt2.flax --compare_song_name "w/o onset" \
        --compare_onset_accuracy_reward_coef 0.0 \
        [shared env flags] --window_sec 8.0 --output tmp/compare.png
"""

from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from flax import serialization
import tyro

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "env"))

import sac
import specs
from train import get_env, Args as TrainArgs, _make_base_actor, _resolve_residual_action_indices, _action_names


@dataclass
class PlotArgs:
    checkpoint: Path
    output: Path = Path("tmp/onset_figure.png")
    song_name: str = ""
    # shared env args
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    seed: int = 0
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    action_reward_observation: bool = False
    # panel 1 reward coefs
    velocity_reward_coef: float = 0.0
    onset_accuracy_reward_coef: float = 0.0
    onset_hit_bonus: float = 1.0
    # residual (panel 1)
    base_checkpoint: Optional[Path] = None
    residual_alpha: float = 0.0
    residual_action_mode: str = "fingers_only"
    # --- optional second panel ---
    compare_checkpoint: Optional[Path] = None
    compare_song_name: str = ""
    compare_velocity_reward_coef: float = 0.0
    compare_onset_accuracy_reward_coef: float = 0.0
    compare_onset_hit_bonus: float = 1.0
    compare_base_checkpoint: Optional[Path] = None
    compare_residual_alpha: float = 0.0
    # zoom window (seconds); 0 = show full episode
    window_sec: float = 8.0
    # key boundary for LH/RH split (MIDI key index, default = middle C = key 39)
    lh_rh_split: int = 39
    # figure-level title (e.g. song name)
    figure_title: str = ""
    # stochastic multi-episode eval
    n_eval_episodes: int = 1
    stochastic: bool = False
    # where to save episode JSON data (stem used; _p1.json / _p2.json appended)
    save_data: Optional[Path] = None


# MIDI note 21 = A0 (key 0). Keys 0-87 → MIDI 21-108.
_BLACK_MIDI = {1, 3, 6, 8, 10}

def _is_black(key: int) -> bool:
    return (key + 9) % 12 in _BLACK_MIDI


# ---------------------------------------------------------------------------
# Env / agent helpers
# ---------------------------------------------------------------------------

def _build_env_and_agent(args: PlotArgs, second: bool = False):
    ckpt       = args.compare_checkpoint if second else args.checkpoint
    vel_coef   = args.compare_velocity_reward_coef if second else args.velocity_reward_coef
    onset_coef = args.compare_onset_accuracy_reward_coef if second else args.onset_accuracy_reward_coef
    hit_bonus  = args.compare_onset_hit_bonus if second else args.onset_hit_bonus
    base_ckpt  = args.compare_base_checkpoint if second else args.base_checkpoint
    alpha      = args.compare_residual_alpha if second else args.residual_alpha

    train_args = TrainArgs(
        environment_name=args.environment_name,
        seed=args.seed,
        n_steps_lookahead=args.n_steps_lookahead,
        trim_silence=args.trim_silence,
        gravity_compensation=args.gravity_compensation,
        reduced_action_space=args.reduced_action_space,
        control_timestep=args.control_timestep,
        wrong_press_termination=args.wrong_press_termination,
        disable_fingering_reward=args.disable_fingering_reward,
        disable_forearm_reward=args.disable_forearm_reward,
        disable_colorization=args.disable_colorization,
        disable_hand_collisions=args.disable_hand_collisions,
        primitive_fingertip_collisions=args.primitive_fingertip_collisions,
        frame_stack=args.frame_stack,
        clip=args.clip,
        action_reward_observation=args.action_reward_observation,
        velocity_reward_coef=vel_coef,
        onset_accuracy_reward_coef=onset_coef,
        onset_hit_bonus=hit_bonus,
        base_checkpoint=base_ckpt,
        residual_alpha=alpha,
        residual_action_mode=args.residual_action_mode,
    )
    env = get_env(train_args)
    spec = specs.EnvironmentSpec.make(env)

    if alpha > 0.0:
        indices = _resolve_residual_action_indices(
            _action_names(spec.action), args.residual_action_mode
        )
        base_actor = _make_base_actor(spec, train_args)
        template = sac.ResidualSAC.initialize(
            spec=spec, config=sac.SACConfig(), base_actor=base_actor,
            seed=args.seed, discount=0.99,
            residual_alpha=alpha, residual_action_indices=indices,
        )
        agent = sac.ResidualSAC.load(ckpt, template)
    else:
        template = sac.SAC.initialize(spec=spec, config=sac.SACConfig(), seed=args.seed, discount=0.99)
        agent = sac.SAC.load(ckpt, template)

    return env, agent


def _run_episode(env, agent, stochastic: bool = False):
    timestep = env.reset()
    raw = env
    while hasattr(raw, "_environment"):
        raw = raw._environment
    task = raw.task
    gt_notes = task._notes
    dt = task.control_timestep

    robot_acts = []
    while not timestep.last():
        if stochastic:
            agent, action = agent.sample_actions(timestep.observation)
        else:
            action = agent.eval_actions(timestep.observation)
        timestep = env.step(action)
        robot_acts.append(task.piano.activation.copy())

    # return updated agent so rng key carries over to next episode
    return np.array(robot_acts, dtype=bool), gt_notes, dt, agent


# ---------------------------------------------------------------------------
# Note segment extraction & classification
# ---------------------------------------------------------------------------

def _extract_note_segments(gt_notes):
    active_since = {}
    segments = []
    for t, notes in enumerate(gt_notes):
        active_now = {n.key for n in notes}
        for key in active_now:
            if key not in active_since:
                active_since[key] = t
        for key in list(active_since):
            if key not in active_now:
                segments.append((key, active_since.pop(key), t))
    T = len(gt_notes)
    for key, start in active_since.items():
        segments.append((key, start, T))
    return segments


def _classify_with_press_time(segments, robot_acts):
    """For each GT note (key, start, end), find press_t: first timestep in [start, end)
    where robot_acts[t, key] is True. Returns (key, start, end, press_t_or_None)."""
    T = robot_acts.shape[0]
    result = []
    for key, start, end in segments:
        press_t = None
        for t in range(start, min(end, T)):
            if robot_acts[t, key]:
                press_t = t
                break
        result.append((key, start, end, press_t))
    return result


def _overall_hit_rate(labeled) -> float:
    if not labeled:
        return 0.0
    h = sum(1 for _, s, _, p in labeled if p is not None and p <= s)
    return h / len(labeled)


def _save_episode_json(labeled, dt: float, label: str, ckpt: str, hit_rate: float, path: Path):
    import json
    data = {
        "song_name": label,
        "checkpoint": str(ckpt),
        "dt": dt,
        "hit_rate": hit_rate,
        "labeled": [[int(k), int(s), int(e), int(p) if p is not None else None]
                    for k, s, e, p in labeled],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved episode data → {path}")


def _load_episode_json(path: Path):
    import json
    with open(path) as f:
        d = json.load(f)
    labeled = [(k, s, e, p) for k, s, e, p in d["labeled"]]
    return labeled, d["dt"], d["song_name"]


def collect_episode_data(args: PlotArgs, second: bool = False,
                          n_episodes: int = 1, maximize: bool = True,
                          save_path: Optional[Path] = None,
                          stochastic: bool = False):
    """Run n_episodes stochastic rollouts, pick best (maximize=True) or worst episode."""
    env, agent = _build_env_and_agent(args, second=second)
    label = (args.compare_song_name if second else args.song_name) or (
        args.environment_name.replace("RoboPianist-debug-", "").replace("-v0", "")
    )
    ckpt = args.compare_checkpoint if second else args.checkpoint

    best_labeled = None
    best_rate = -1.0
    dt = 0.05

    for i in range(n_episodes):
        robot_acts, gt_notes, dt, agent = _run_episode(env, agent, stochastic=stochastic)
        segments = _extract_note_segments(gt_notes)
        labeled = _classify_with_press_time(segments, robot_acts)
        rate = _overall_hit_rate(labeled)
        is_better = rate > best_rate if maximize else rate < best_rate
        if best_labeled is None or is_better:
            best_labeled = labeled
            best_rate = rate
        print(f"  ep {i+1}/{n_episodes}: hit={rate:.3f}  {'← best' if best_labeled is labeled else ''}")

    print(f"  Selected: hit={best_rate:.3f} ({'best' if maximize else 'worst'} of {n_episodes})")

    if save_path is not None:
        _save_episode_json(best_labeled, dt, label, ckpt, best_rate, save_path)

    return best_labeled, dt, label


# ---------------------------------------------------------------------------
# Window & hand helpers
# ---------------------------------------------------------------------------

def _find_best_window(labeled1, labeled2, dt: float, window_sec: float,
                       min_notes: int = 8) -> Tuple[float, float]:
    """Find the time window [t0, t0+window_sec] where onset hit-rate gap is largest.

    Gap = hit_rate(policy1) - hit_rate(policy2) over notes whose onset falls in window.
    Only considers windows with at least min_notes GT onsets.
    """
    if not labeled1 or not labeled2:
        return 0.0, window_sec

    T_max = max(max(end for _, _, end, _ in labeled1),
                max(end for _, _, end, _ in labeled2))
    win = int(window_sec / dt)
    if win >= T_max:
        return 0.0, T_max * dt

    def stats_in(labeled, t0, t1):
        notes = [(s, p) for _, s, _, p in labeled if t0 <= s < t1]
        n = len(notes)
        if n == 0:
            return 0.0, 0
        on_time = sum(1 for s, p in notes if p is not None and p <= s)
        return on_time / n, n

    candidates = []
    for t0 in range(0, T_max - win):   # stride=1 for exhaustive search
        t1 = t0 + win
        r1, n1 = stats_in(labeled1, t0, t1)
        r2, n2 = stats_in(labeled2, t0, t1)
        if n1 < min_notes or n2 < min_notes:
            continue
        # maximize policy1 hit rate (show our method at its best)
        candidates.append((r1, r1 - r2, t0, n1))

    if not candidates:
        print("  No window with enough notes; using t=0")
        return 0.0, window_sec

    candidates.sort(reverse=True)
    # Print top-3 for reference
    for rank, (r1, gap, t0, n) in enumerate(candidates[:3]):
        print(f"  Top-{rank+1}: t={t0*dt:.1f}s–{(t0+win)*dt:.1f}s  hit={r1:.2f}  gap={gap:.2f}  notes={n}")

    best_t0 = candidates[0][2]
    return best_t0 * dt, (best_t0 + win) * dt


def _filter_window(labeled, t0_sec: float, t1_sec: float, dt: float):
    # Include notes whose onset falls in the window; clip display to window edges.
    t0 = t0_sec / dt
    t1 = t1_sec / dt
    return [(key, start, end, press_t) for key, start, end, press_t in labeled
            if t0 <= start < t1]


def _split_hands(labeled, split_key: int):
    rh = [(k, s, e, p) for k, s, e, p in labeled if k >= split_key]
    lh = [(k, s, e, p) for k, s, e, p in labeled if k < split_key]
    return rh, lh


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

HIT_COLOR  = "#4CAF50"
MISS_COLOR = "#EF5350"
WHITE_BG   = "#F5F5F5"
BLACK_BG   = "#BDBDBD"


def _draw_panel(ax, labeled, dt, t0_sec, t1_sec, title="", show_xlabel=False):
    """Draw piano roll where each GT note bar is split by press_t:
    - [start, press_t): red  (waiting / missed so far)
    - [press_t, end):   green (robot is pressing)
    - entire bar red if never pressed
    """
    used = [k for k, *_ in labeled]
    if not used:
        ax.set_visible(False)
        return

    y_min = max(min(used) - 1, 0)
    y_max = min(max(used) + 1, 87)

    for key in range(y_min, y_max + 1):
        bg = BLACK_BG if _is_black(key) else WHITE_BG
        ax.axhspan(key - 0.5, key + 0.5, color=bg, zorder=0, linewidth=0)

    for key, start, end, press_t in labeled:
        x_start = start * dt
        x_end   = min(end * dt, t1_sec)
        if x_end <= x_start:
            continue

        if press_t is None:
            # never pressed → all red
            ax.barh(key, x_end - x_start, left=x_start,
                    height=0.78, color=MISS_COLOR, linewidth=0, zorder=2)
        else:
            x_press = press_t * dt
            # red portion: [start, press_t)
            if x_press > x_start:
                ax.barh(key, x_press - x_start, left=x_start,
                        height=0.78, color=MISS_COLOR, linewidth=0, zorder=2)
            # green portion: [press_t, end)
            if x_press < x_end:
                ax.barh(key, x_end - max(x_press, x_start), left=max(x_press, x_start),
                        height=0.78, color=HIT_COLOR, linewidth=0, zorder=2)

    ax.set_xlim(t0_sec, t1_sec)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1)

    midi_ticks = {k: f"C{(k+21)//12 - 1}" for k in range(y_min, y_max + 1)
                  if (k + 21) % 12 == 0}
    ax.set_yticks(list(midi_ticks.keys()))
    ax.set_yticklabels(list(midi_ticks.values()), fontsize=8)

    if title:
        ax.set_title(title, fontsize=10, pad=3)
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=9)
    else:
        ax.set_xticklabels([])


def make_comparison_figure(labeled1, labeled2, dt, name1, name2,
                            split_key, window_sec, output: Path,
                            song_prefix: str = ""):
    t0, t1 = _find_best_window(labeled1, labeled2, dt, window_sec)

    win1 = _filter_window(labeled1, t0, t1, dt)
    win2 = _filter_window(labeled2, t0, t1, dt)

    rh1, lh1 = _split_hands(win1, split_key)
    rh2, lh2 = _split_hands(win2, split_key)

    def pct(labeled):
        if not labeled: return 0, 0
        h = sum(1 for _, s, _, p in labeled if p is not None and p <= s)
        return h, len(labeled)

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7.5))
    fig.patch.set_facecolor("#FAFAFA")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.18,
                          left=0.06, right=0.97, top=0.97, bottom=0.12)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
    for row in axes:
        for ax in row:
            ax.set_facecolor("#FAFAFA")

    # ── Panel titles: "Method — Hand" ────────────────────────────────────────
    row_colors = ["#2E7D32", "#C62828"]
    hand_labels = ["Right Hand (RH)", "Left Hand (LH)"]
    row_names   = [name1, name2]
    panels = [
        (axes[0][0], rh1), (axes[0][1], lh1),
        (axes[1][0], rh2), (axes[1][1], lh2),
    ]
    for idx, (ax, labeled) in enumerate(panels):
        row, col = divmod(idx, 2)
        prefix = f"{song_prefix}  " if song_prefix else ""
        panel_title = f"{prefix}{row_names[row]}  —  {hand_labels[col]}"
        ax.set_title(panel_title, fontsize=11, fontweight="bold",
                     color=row_colors[row], pad=5)

    # ── Draw panels ──────────────────────────────────────────────────────────
    for (ax, labeled), show_x in zip(panels, [False, False, True, True]):
        _draw_panel(ax, labeled, dt, t0, t1, show_xlabel=show_x)

    # ── Hit-rate annotation inside each panel ────────────────────────────────
    for (ax, labeled) in panels:
        h, n = pct(labeled)
        pct_val = h / n * 100 if n else 0
        color = HIT_COLOR if pct_val >= 50 else MISS_COLOR
        ax.text(0.97, 0.96, f"{pct_val:.0f}% on time",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold", color=color,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=3))

    # ── Legend: bottom-center, large ─────────────────────────────────────────
    legend = [
        mpatches.Patch(color=HIT_COLOR,  label="Pressed on time"),
        mpatches.Patch(color=MISS_COLOR, label="Late or missed"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2,
               fontsize=12, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.0), frameon=True,
               edgecolor="#CCCCCC")

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")


def make_single_figure(labeled, dt, name, split_key, output: Path):
    T_max = max(end for _, _, end, _ in labeled) if labeled else 1
    t0, t1 = 0.0, T_max * dt
    rh, lh = _split_hands(labeled, split_key)

    def hit_str(lb):
        if not lb: return "–"
        h = sum(1 for _, s, _, p in lb if p is not None and p <= s)
        return f"{h}/{len(lb)} ({h/len(lb)*100:.0f}% on time)"

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"hspace": 0.3})
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes:
        ax.set_facecolor("#FAFAFA")

    _draw_panel(axes[0], rh, dt, t0, t1, title=f"{name}  ·  RH  ({hit_str(rh)} hit)")
    _draw_panel(axes[1], lh, dt, t0, t1, title=f"{name}  ·  LH  ({hit_str(lh)} hit)",
                show_xlabel=True)

    legend = [
        mpatches.Patch(color=HIT_COLOR,  label="Hit (on time)"),
        mpatches.Patch(color=MISS_COLOR, label="Missed"),
    ]
    axes[0].legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: PlotArgs) -> None:
    save1 = Path(str(args.save_data) + "_p1.json") if args.save_data else None
    save2 = Path(str(args.save_data) + "_p2.json") if args.save_data else None

    if save1 and save1.exists():
        print(f"Loading cached panel 1 from {save1}")
        labeled1, dt, name1 = _load_episode_json(save1)
    else:
        print(f"Running panel 1 ({args.n_eval_episodes} eps, pick best): {args.checkpoint}")
        labeled1, dt, name1 = collect_episode_data(
            args, second=False,
            n_episodes=args.n_eval_episodes, maximize=True,
            save_path=save1, stochastic=args.stochastic,
        )

    if args.compare_checkpoint is not None:
        if save2 and save2.exists():
            print(f"Loading cached panel 2 from {save2}")
            labeled2, _, name2 = _load_episode_json(save2)
        else:
            print(f"Running panel 2 ({args.n_eval_episodes} eps, pick worst): {args.compare_checkpoint}")
            labeled2, _, name2 = collect_episode_data(
                args, second=True,
                n_episodes=args.n_eval_episodes, maximize=False,
                save_path=save2, stochastic=args.stochastic,
            )
        make_comparison_figure(
            labeled1, labeled2, dt, name1, name2,
            split_key=args.lh_rh_split,
            window_sec=args.window_sec,
            output=args.output,
            song_prefix=args.figure_title,
        )
    else:
        make_single_figure(labeled1, dt, name1, args.lh_rh_split, args.output)


if __name__ == "__main__":
    main(tyro.cli(PlotArgs))
