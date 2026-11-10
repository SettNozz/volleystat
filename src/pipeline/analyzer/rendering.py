"""Frame drawing and explicit annotated-video rendering.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.player_pose import PlayerPose, draw_skeleton
from src.utils.logger import get_logger

log = get_logger("Analyzer")


class RenderingMixin:

    def _draw_frame(
        self,
        frame: np.ndarray,
        ball_pos: Optional[Tuple[float, float]],
        player_teams: Dict[str, List[Tuple[float, float]]],
        fsm_status: Dict,
        event_display: Optional[str] = None,
        player_poses: Optional[List[PlayerPose]] = None,
        ball_source: str = "detected",
        ball_confidence: Optional[float] = None,
    ) -> np.ndarray:
        out = frame.copy()

        # Court outline from homography (thin) — shows the on-court filter area
        projector = self._get_projector()
        if projector is not None:
            hw, hl = projector.half_width, projector.half_length
            quad = np.array([
                projector.world_to_image(w) for w in
                [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
            ], dtype=np.int32)
            cv2.polylines(out, [quad], isClosed=True, color=(200, 200, 200), thickness=2)
            # Line under the net (world y=0) — the actual team split reference
            floor_line = projector.net_floor_points()
            if floor_line is not None:
                p0, p1 = floor_line
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                         (255, 0, 255), 3)
                for label, point in (("A", p0), ("B", p1)):
                    px, py = int(point[0]), int(point[1])
                    cv2.circle(out, (px, py), 12, (255, 0, 255), -1)
                    cv2.putText(out, label, (px + 14, py - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            else:
                p0 = projector.world_to_image((-hw, 0.0))
                p1 = projector.world_to_image((hw, 0.0))
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                         (200, 200, 200), 2)

        # Net: SAM-detected tape band if available, else calibration net_y line
        if self.calibration:
            c = self.calibration
            x0, x1 = int(c.net_x_left), int(c.net_x_right)
            # Automatic tape detections have only a height, so their extent
            # comes from the posts. Explicitly drawn tape endpoints retain
            # their own X coordinates, even if a stored floor line differs.
            if c.net_tape_endpoint_y is None and c.net_floor_left and c.net_floor_right:
                x0, x1 = sorted((int(c.net_floor_left[0]), int(c.net_floor_right[0])))
            if c.net_tape_top is not None and c.net_tape_bottom is not None:
                left_y, right_y = c.net_tape_endpoint_y or (c.net_tape_top, c.net_tape_top)
                depth = c.net_tape_bottom - c.net_tape_top
                top_left, top_right = (x0, int(left_y)), (x1, int(right_y))
                bottom_left, bottom_right = (x0, int(left_y + depth)), (x1, int(right_y + depth))
                band = out.copy()
                polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
                cv2.fillConvexPoly(band, polygon, (0, 180, 255))
                cv2.addWeighted(band, 0.25, out, 0.75, 0, out)
                cv2.line(out, top_left, top_right, (0, 140, 255), 2)
                cv2.line(out, bottom_left, bottom_right, (0, 90, 220), 2)
            else:
                ny = int(c.net_y)
                cv2.line(out, (x0, ny), (x1, ny), (255, 220, 0), 4)

        # Ball
        if ball_pos:
            cx, cy = int(ball_pos[0]), int(ball_pos[1])
            ball_colors = {
                "detected": (0, 255, 0),
                "interpolated_short": (0, 200, 255),
                "reconstructed_conservative": (255, 212, 0),
                "reconstructed_contact_aware": (216, 79, 255),
                "reconstructed_occlusion": (250, 139, 167),
            }
            color = ball_colors.get(ball_source, (0, 255, 0))
            cv2.circle(out, (cx, cy), 18, color, 3)
            cv2.circle(out, (cx, cy), 5, color, -1)
            if ball_source != "detected":
                label = ball_source.replace("reconstructed_", "")
                if ball_confidence is not None:
                    label += f" {float(ball_confidence):.2f}"
                cv2.putText(
                    out, label, (cx + 22, max(24, cy - 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )

        # Players — skeletons when keypoints are available, dots as fallback
        if player_poses:
            for pose in player_poses:
                st = self._SKELETON_STYLE[pose.team]
                draw_skeleton(out, pose, st["color"],
                              bone_thickness=st["bone"], kp_radius=st["kp"],
                              draw_id=st["draw_id"] or pose.player_identity_processed)
                if pose.team_prior:
                    x1, y1 = int(pose.bbox[0]), int(pose.bbox[1])
                    cv2.putText(
                        out, "BACK/NET prior", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
                    )
        else:
            colors = {"team_a": (255, 80, 0), "team_b": (0, 80, 255)}
            for team, pts in player_teams.items():
                for px, py in pts:
                    cv2.circle(out, (int(px), int(py)), 10, colors[team], -1)

        # Confirmed lifecycle overlay (line 1). The old raw FSM tuple often
        # displayed internally valid but user-meaningless combinations such as
        # ``serving | | team_b [0]``.
        lifecycle = fsm_status.get("lifecycle") or {}
        lifecycle_status = lifecycle.get("status", "unconfirmed")
        lifecycle_labels = {
            "unconfirmed": "Waiting for serve",
            "serve_candidate": "Serve candidate",
            "serve_confirmed": "Serve confirmed",
            "reception_pending": "Reception confirmed",
            "playable_rally": "Playable rally",
            "ace_no_reception": "Ace / no reception",
            "reception_failed": "Reception failed",
            "serve_fault": "Serve fault",
        }
        team  = fsm_status.get("current_team", "?")
        contacts = fsm_status.get("team_contacts", 0)
        text = lifecycle_labels.get(lifecycle_status, lifecycle_status)
        if lifecycle_status in {"reception_pending", "playable_rally"}:
            text += f" | {team} [{contacts}]"
        cv2.putText(out, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

        # Event model overlay (line 2): sliding-window Transformer v3 prediction
        if event_display is not None and event_display != "-":
            ev_text = f"Event: {event_display}"
            cv2.putText(out, ev_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 200, 0), 4)

        return out

    def _draw_saved_annotation(self, frame: np.ndarray, ann: Dict) -> np.ndarray:
        teams = {"team_a": [], "team_b": []}
        for saved in ann.get("poses") or []:
            box = saved.get("bb") or []
            if len(box) == 4:
                team = "team_a" if saved.get("team") in ("a", "team_a") else "team_b"
                teams[team].append(((box[0]+box[2])/2., (box[1]+box[3])/2.))
        ball = ann.get("ball")
        drawn = self._draw_frame(
            frame, (float(ball[0]), float(ball[1])) if ball else None, teams,
            {"game_state": ann.get("state"), "rally_phase": ann.get("phase"),
             "lifecycle": ann.get("lifecycle"), "current_team": ann.get("current_team"),
             "team_contacts": ann.get("contacts", 0)},
            event_display=ann.get("event_model"), player_poses=self._poses_from_annotation(ann),
            ball_source=str(ann.get("ball_source") or "detected"),
            ball_confidence=float(ann.get("ball_confidence") or 0.),
        )
        if not ball and ann.get("ball_out_of_frame"):
            cv2.putText(drawn, "BALL OUT OF FRAME", (20, frame.shape[0]-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        return drawn

    def _render_final_video(
        self,
        video_path: Path,
        output_path: Path,
        annotations: List[Dict],
        fps: float,
        frame_width: int,
        frame_height: int,
        max_frames: int,
    ) -> bool:
        """Render final trajectories and roster IDs from the exported annotations."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.warning(f"Final render cannot open source video: {video_path}")
            return False

        writer = None
        for codec in ("mp4v", "avc1", "H264"):
            candidate = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*codec), fps,
                (frame_width, frame_height),
            )
            if candidate.isOpened():
                writer = candidate
                log.info(f"Final VideoWriter codec: {codec}")
                break
            candidate.release()
        if writer is None:
            cap.release()
            log.warning("Final VideoWriter failed — annotated video not saved")
            return False

        by_frame = {int(ann["f"]): ann for ann in annotations}
        rendered = 0
        try:
            while rendered < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                ann = by_frame.get(rendered)
                if ann is None:
                    writer.write(frame)
                    rendered += 1
                    continue

                drawn = self._draw_saved_annotation(frame, ann)
                writer.write(drawn)
                rendered += 1
                if rendered % max(1, int(fps * 2)) == 0:
                    self._emit_progress(
                        phase="rendering",
                        progress=98.0 + rendered / max(max_frames, 1),
                        frame_current=rendered,
                        frame_total=max_frames,
                        video_percent=round(100.0 * rendered / max(max_frames, 1), 2),
                        message=f"Rendering final annotated video {rendered:,}/{max_frames:,}",
                    )
        finally:
            cap.release()
            writer.release()
        log.info(f"Final annotated video: {output_path} ({rendered:,} frames)")
        return rendered > 0
