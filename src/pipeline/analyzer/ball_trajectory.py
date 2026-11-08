"""Offline ball trajectory selection, smoothing and outlier handling.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

from typing import Dict, List

import numpy as np

from src.tracking.ball_trajectory_dp import BallTrajectoryDP, interpolate_gaps
from src.tracking.ball_static_sources import (
    find_static_sources_in_candidates, suppress_static_candidates,
)
from src.tracking.ball_trajectory_smoother import jitter_metric, smooth_ball_annotations
from src.tracking.ball_gap_reconstructor import BallGapReconstructor
from src.utils.logger import get_logger

log = get_logger("Analyzer")


class BallTrajectoryMixin:

    def _apply_trajectory_dp(self, stats: "GameStats") -> bool:
        """Replace the gate's per-frame picks with a globally chosen path.

        Rewrites ``ball`` on every annotation, so everything downstream — the
        FSM re-run, the heatmap, the viewer — reads the same trajectory.
        ``ball_new`` is rebuilt from physically disconnected solver segments.
        A restart is a change of hypothesis, not a ball flight, so the FSM,
        heatmap and Viewer must not measure or draw across it.

        Returns whether the trajectory was actually replaced. A run with no
        candidates at all (an empty clip, or every frame below the confidence
        floor) leaves the gate's output in place, and the caller must then keep
        the legacy clean-up passes that output still needs.
        """
        if not self._ball_candidates:
            return False

        # A fixed object in the gym — a spare ball on the floor, a bag, a lamp —
        # is detected confidently on every frame, and a confident candidate has
        # *negative* cost in the solver. A long stationary run therefore beats a
        # short correct flight on total cost, which is how the chosen path ends
        # up alternating between the real ball and a piece of furniture. Take
        # the bait off the table before the solver ever sees it.
        if self.suppress_static_ball_sources:
            sources = find_static_sources_in_candidates(
                self._ball_candidates,
                fps=float(getattr(self, "_current_fps", 30.0) or 30.0),
                px_scale=self._ball_px_scale,
            )
            removed = suppress_static_candidates(
                self._ball_candidates, sources, px_scale=self._ball_px_scale,
            )
            if sources:
                log.info(
                    f"Static ball sources: {len(sources)} fixed object(s), "
                    f"{removed} candidates dropped "
                    f"(longest {max(s.seconds for s in sources):.1f} s)"
                )
            self._static_source_report = {
                "candidate_sources": [s.as_dict() for s in sources],
                "candidates_removed": removed,
            }

        solver = BallTrajectoryDP(px_scale=self._ball_px_scale)
        solved = solver.solve(self._ball_candidates)
        chosen = (
            interpolate_gaps(solved, max_gap=4, max_step_px=solver.max_step_px)
            if self.ball_reconstruction_mode == "off"
            else solved
        )

        before = sum(1 for a in stats.frame_annotations if a.get("ball"))
        for ann in stats.frame_annotations:
            cand = chosen.get(ann["f"])
            ann["ball"] = [round(cand.x, 1), round(cand.y, 1)] if cand else None
            ann["ball_confidence"] = round(float(cand.confidence), 4) if cand else 0.0
            ann["ball_new"] = (
                True if cand is not None and ann["f"] in solver.track_start_frames
                else None
            )
            ann["ball_interp"] = (
                True if cand is not None and cand.confidence == 0.0 else None
            )

        if self.ball_reconstruction_mode != "off":
            active_mode = (
                "conservative"
                if self.ball_reconstruction_mode == "compare"
                else self.ball_reconstruction_mode
            )
            calibration = self.calibration
            reconstructor = BallGapReconstructor(
                int(calibration.frame_width if calibration else 1280),
                int(calibration.frame_height if calibration else 720),
                net_tape_top=calibration.net_tape_top if calibration else None,
                net_tape_bottom=calibration.net_tape_bottom if calibration else None,
                net_floor_left=calibration.net_floor_left if calibration else None,
                net_floor_right=calibration.net_floor_right if calibration else None,
            )
            summaries = reconstructor.apply(
                stats.frame_annotations,
                active_mode=active_mode,
                contact_hints=self.reconstruction_contact_hints,
                keep_variants=True,
            )
            stats.ball_reconstruction = {
                "requested_mode": self.ball_reconstruction_mode,
                "active_mode": active_mode,
                "affects_fsm": self.reconstruction_affects_fsm,
                "variants": {
                    name: vars(summary) for name, summary in summaries.items()
                },
            }
        after = sum(1 for a in stats.frame_annotations if a.get("ball"))

        stats.ball_detected_frames = after
        log.info(
            f"Trajectory solver: {after} frames with a ball (gate had {before}), "
            f"from {solver.stats['frames_with_candidates']} frames with candidates "
            f"in {solver.stats['segments']} segments / "
            f"{len(solver.track_start_frames)} physical tracks"
        )
        self._ball_candidates.clear()
        return True

    def _smooth_ball_trajectory(self, stats: "GameStats") -> None:
        """Fit per-frame detector wobble out of the chosen trajectory.

        Records the jitter (mean absolute second difference of the path) either
        side of the pass, because "looks smoother" is not a measurement and the
        window is a tuning knob that has to be judged on numbers as well as on
        the video.
        """
        window = getattr(self, "ball_smooth_window", 0)
        if not window:
            stats.ball_smoothing = {"window": 0}
            return

        before = jitter_metric(stats.frame_annotations)
        summary = smooth_ball_annotations(
            stats.frame_annotations,
            window=window,
            px_scale=getattr(self, "_ball_px_scale", 1.0),
        )
        summary["jitter_before"] = before
        summary["jitter_after"] = jitter_metric(stats.frame_annotations)
        stats.ball_smoothing = summary
        log.info(
            f"Ball smoothing (window {summary['window']}): "
            f"{summary['moved']}/{summary['points']} points moved, "
            f"mean {summary['mean_shift_px']} px, "
            f"{summary['contacts_preserved']} contacts kept sharp, "
            f"jitter {before['mean']} -> {summary['jitter_after']['mean']} px/frame"
        )

    def _filter_ball_outliers(
        self,
        annotations: List[Dict],
        max_gap: int = 4,
        detour_ratio: float = 2.5,
        min_jump_px: float = 30.0,
    ) -> int:
        """
        Remove ball detections that are inconsistent with neighbouring frames.

        A detection at frame N is a false positive if:
        - It is within `max_gap` frames of both a previous and a next detection
        - The detour distance (prev→curr + curr→next) exceeds `detour_ratio` × the
          direct distance (prev→next) — i.e. the ball "jumps out and back"
        - The jump size is at least `min_jump_px` (ignore tiny wiggles)

        Removed frames are cleared (ball=None) so interpolation can fill them later.
        Returns the number of frames removed.
        """
        detected = [i for i, ann in enumerate(annotations)
                    if ann.get("ball") and not ann.get("ball_interp")]
        removed = 0

        for k in range(1, len(detected) - 1):
            i_prev, i_curr, i_next = detected[k - 1], detected[k], detected[k + 1]

            # Only consider closely-spaced detections
            if i_curr - i_prev > max_gap or i_next - i_curr > max_gap:
                continue

            # Skip if any neighbour was already filtered this pass
            bp = annotations[i_prev]["ball"]
            bc = annotations[i_curr]["ball"]
            bn = annotations[i_next]["ball"]
            if bp is None or bc is None or bn is None:
                continue

            xp, yp = bp
            xc, yc = bc
            xn, yn = bn

            d_in   = np.hypot(xc - xp, yc - yp)   # prev → curr
            d_out  = np.hypot(xn - xc, yn - yc)   # curr → next
            d_skip = np.hypot(xn - xp, yn - yp)   # prev → next (direct)

            if d_skip < 1.0:
                d_skip = 1.0

            jump = max(d_in, d_out)
            if jump < min_jump_px:
                continue  # tiny movement, not an outlier

            if (d_in + d_out) > detour_ratio * d_skip:
                annotations[i_curr]["ball"] = None
                annotations[i_curr]["ball_filtered"] = True
                removed += 1

        return removed

    def _interpolate_ball_positions(self, annotations: List[Dict], max_gap: int = 10) -> int:
        """
        Linearly interpolate ball positions for short detection gaps.

        Only fills gaps of at most `max_gap` consecutive missing frames — longer gaps
        mean the ball is genuinely out of sight.  Interpolated frames are tagged with
        ``"ball_interp": True`` so the viewer can style them differently.

        Returns the number of frames filled.
        """
        if not annotations:
            return 0

        # Indices of annotation entries that have a confirmed ball detection
        detected = [i for i, ann in enumerate(annotations) if ann.get("ball") and not ann.get("ball_interp")]
        filled = 0

        for k in range(len(detected) - 1):
            i0, i1 = detected[k], detected[k + 1]
            gap = i1 - i0 - 1
            if gap <= 0 or gap > max_gap:
                continue
            x0, y0 = annotations[i0]["ball"]
            x1, y1 = annotations[i1]["ball"]
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                annotations[i0 + step]["ball"] = [
                    round(x0 + t * (x1 - x0), 1),
                    round(y0 + t * (y1 - y0), 1),
                ]
                annotations[i0 + step]["ball_interp"] = True
                filled += 1

        return filled
