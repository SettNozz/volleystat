"""Final-trajectory FSM replay, event-model features and contact spotting.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.player_pose import PlayerPose
from src.tracking.pose_feature_extractor import PoseFeatureExtractor
from src.tracking.match_analytics import filter_non_game_records
from src.utils.logger import get_logger
from src.models.inference_cache import CacheMiss
from src.pipeline.analyzer.types import _EventOverlayState

log = get_logger("Analyzer")


class EventsMixin:

    def _recompute_features_and_predictions(self, annotations: List[Dict], fps: float) -> None:
        """Replay saved poses against final ball coordinates and clean FSM states.

        Features use the exported ``ball`` trajectory, including accepted
        reconstruction. The separate reconstruction-affects-FSM policy remains
        enforced by the clean FSM replay. Raw observations/provenance stay intact.
        """
        pose_fx = PoseFeatureExtractor()
        predictor = self._event_predictor
        if predictor is not None:
            predictor.reset()
        overlay = _EventOverlayState(fps)
        calibration = self.calibration
        pose_net_y = (
            calibration.net_tape_top
            if calibration is not None and calibration.net_tape_top is not None
            else (calibration.net_y if calibration is not None else 830.0)
        )
        for ann in annotations:
            frame_idx = int(ann["f"])
            poses = self._poses_from_annotation(ann)
            ball = ann.get("ball")
            ball_pos = tuple(float(value) for value in ball) if ball is not None else None
            if ann.get("ball_out_of_frame"):
                ball_pos = None
            ann["pose_f"] = pose_fx.update(frame_idx, poses, ball_pos, pose_net_y)
            ann.pop("event_model", None)
            ann.pop('event_model_decision', None)
            if predictor is None or calibration is None:
                continue
            teams = {"team_a": [], "team_b": []}
            for pose in poses:
                teams[pose.team].append(pose.center)
            # ``p`` stores the exact rounded centers used by offline feature
            # builders, grouped team_a then team_b. Avoid bbox-rounding drift.
            points = ann.get("p")
            if points is not None and len(points) == len(poses):
                split = len(teams["team_a"])
                teams = {"team_a": points[:split], "team_b": points[split:]}
            state = ann.get("state") or ""
            raw_display = predictor.add_frame(
                ball_pos, teams, calibration.net_y, fps,
                fsm_state=state,
                video_time=frame_idx / fps,
                pose_features=ann["pose_f"],
                frame_size=(calibration.frame_width, calibration.frame_height),
                ball_new=bool(ann.get("ball_new")),
            )
            display = overlay.update(raw_display, state, frame_idx)
            guard = getattr(predictor, 'serve_guard', None)
            if guard is not None and guard.reason(display):
                display = None
            if getattr(predictor, 'last_decision', None) is not None:
                ann['event_model_decision'] = dict(predictor.last_decision)
            if display is not None:
                ann["event_model"] = display

    def _spot_contacts(self, stats: "GameStats", fps: float) -> None:
        """Replace the FSM's contact events with the dense spotter's.

        Measured leave-one-video-out on the three gold videos, the spotter puts
        a typed contact on the right frame far more often than the FSM
        (0.79 against 0.156 four-class macro F1 at +/-5 frames). The contact
        events are replaced, and so are rallies and service attempts: they are
        rebuilt from the spotter's contacts, each rally opening on its serve
        (see ``rally_segmentation``). Score and match analytics stay
        FSM-derived; they were computed before this point and need team sides
        the spotter does not produce.

        The FSM's events, rallies, attempts and counters are kept in their
        ``*_fsm`` fields so enabling this never loses what the previous path
        produced.
        """
        # Read defensively: tests build an analyzer with ``__new__`` and set
        # only the attributes the path under test needs, as elsewhere here.
        checkpoint = getattr(self, "contact_spotter_path", None)
        if not checkpoint:
            return
        from action_recognition.rally_spotting.contact_spotter import (
            ContactSpotter, phase_segments,
        )
        from action_recognition.rally_spotting import dense_contacts as dc
        from action_recognition.rally_spotting.rally_segmentation import (
            segment_rallies,
        )

        if getattr(self, "_contact_spotter", None) is None:
            self._contact_spotter = ContactSpotter.load(checkpoint)
            log.info(
                f"Contact spotter: {checkpoint} "
                f"({len(self._contact_spotter.members)} member(s))"
            )
        self._cache_dense_probabilities(self._contact_spotter, checkpoint)
        calib = self.calibration
        payload = {
            "fps": fps,
            "frame_width": calib.frame_width if calib else 1280,
            "frame_height": calib.frame_height if calib else 720,
            "net_y": calib.net_y if calib else 370.0,
            "frames": stats.frame_annotations,
        }
        try:
            spotted = self._contact_spotter.run(payload)
        except Exception as error:
            if isinstance(error, CacheMiss):
                raise
            # A failed spotter must not cost the whole analysis; the FSM
            # events already in ``all_events`` stay as they are.
            log.warning(f"Contact spotter failed, keeping FSM events: {error}")
            stats.contact_spotter = {"enabled": True, "error": str(error)}
            return

        stats.all_events_fsm = list(stats.all_events)
        events = []
        for event in spotted.events:
            events.append({
                "frame": event["frame"],
                "time": round(event["frame"] / fps, 2),
                "type": event["type"],
                "core_type": event["core_type"],
                "confidence": event["confidence"],
                "phase_before": event["phase_before"],
                "phase_after": event["phase_after"],
                "source": event["source"],
                # The spotter does not attribute a contact to a side; these
                # stay empty rather than carrying a guess.
                "team": None,
                "game_state": None,
                "rally_phase": None,
            })
        serve_probability = (
            spotted.probabilities[:, dc.LABEL_TYPES.index("serve")]
            if spotted.probabilities is not None else None
        )
        segmented = segment_rallies(
            events, fps,
            serve_probability=serve_probability,
            phase_labels=spotted.phase_labels,
            frame_numbers=spotted.frame_numbers,
        )
        # A missed serve recovered from the serve probability is still the
        # model's evidence, so it joins the event list, marked ``inferred``
        # with its own source; the review queue reads only decoded contacts.
        events = sorted(
            events + segmented["inferred_serves"],
            key=lambda event: int(event["frame"]),
        )
        stats.all_events = events
        stats.rallies_fsm = list(stats.rallies)
        stats.service_attempts_fsm = list(stats.service_attempts)
        stats.rally_lifecycle_fsm = dict(stats.rally_lifecycle)
        attempts = segmented["service_attempts"]
        match_segments = getattr(self, "match_segments", None)
        if match_segments:
            attempts = filter_non_game_records(
                attempts, match_segments, frame_field="start_frame",
            )
        kept_ids = {attempt["service_attempt_id"] for attempt in attempts}
        stats.service_attempts = attempts
        stats.rallies = [
            rally for rally in segmented["rallies"]
            if rally["service_attempt_id"] in kept_ids
        ]
        lifecycle = dict(segmented["rally_lifecycle"])
        if match_segments:
            lifecycle["service_attempts"] = len(attempts)
            lifecycle["playable_rallies"] = len(stats.rallies)
            lifecycle["serves_not_played"] = len(attempts) - len(stats.rallies)
            lifecycle["excluded_as_non_game"] = (
                len(segmented["service_attempts"]) - len(attempts)
            )
        stats.rally_lifecycle = lifecycle
        by_frame: Dict[int, List[str]] = {}
        for event in events:
            by_frame.setdefault(int(event["frame"]), []).append(event["type"])
        for ann in stats.frame_annotations:
            frame_index = int(ann["f"])
            ann["ev_fsm"] = list(ann.get("ev") or [])
            ann["ev"] = by_frame.get(frame_index, [])
        phases = phase_segments(spotted.phase_labels, spotted.frame_numbers)
        stats.contact_spotter = {
            "enabled": True,
            "checkpoint": str(checkpoint),
            "diagnostics": spotted.diagnostics,
            "phase_segments": phases,
            "rally_segmentation": {
                "rules": segmented["rules"],
                "inferred_serves": len(segmented["inferred_serves"]),
                "groups_without_serve": segmented["unanchored_groups"],
            },
            "replaced": {
                "all_events": "dense_contact_spotter",
                "fsm_events_kept_in": "all_events_fsm",
                "per_frame_fsm_events_kept_in": "ev_fsm",
                "rallies": "spotter_rallies_v1",
                "fsm_rallies_kept_in": [
                    "rallies_fsm", "service_attempts_fsm", "rally_lifecycle_fsm",
                ],
            },
            "not_replaced": ["match_analytics", "attack_heatmap"],
        }
        log.info(
            f"Contact spotter: {len(events)} events "
            f"({spotted.diagnostics.get('per_type')}, "
            f"{len(segmented['inferred_serves'])} inferred serves), "
            f"{len(stats.rallies)} rallies from {len(attempts)} serves, "
            f"{len(phases)} phase segments, FSM had "
            f"{len(stats.all_events_fsm)} events / {len(stats.rallies_fsm)} rallies"
        )

    @staticmethod
    def _poses_from_annotation(ann: Dict) -> List[PlayerPose]:
        poses = []
        for saved in ann.get("poses") or []:
            bbox = tuple(float(value) for value in saved.get("bb", ()))
            keypoints = np.asarray(saved.get("kp") or [], dtype=np.float32)
            if len(bbox) != 4 or keypoints.shape != (17, 3):
                continue
            poses.append(PlayerPose(
                team="team_a" if saved.get("team") in ("a", "team_a") else "team_b",
                center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                bbox=bbox,
                keypoints=keypoints,
                track_id=saved.get("id"),
                back_facing_score=saved.get("back"),
                source=saved.get("source", "yolo_pose"),
                mask_instance_id=saved.get("mask_instance_id"),
                team_prior=saved.get("team_prior"),
                player_id=saved.get("player_id"),
                player_identity_processed="player_id" in saved,
                team_id=saved.get('team_id'), team_source=saved.get('team_source'),
                team_geometry=saved.get('team_geometry'),
            ))
        return poses

    def _assign_contact_sides(self, stats: "GameStats", fps: float) -> None:
        """Propose which side of the net made each final contact.

        Runs on whatever ``all_events`` holds once the spotter has had its
        turn, reading only saved box sizes and the calibration: no inference.
        ``team`` is left as the event's producer wrote it; the proposal goes
        into separate ``side*`` fields and abstains when the geometry cannot
        carry it (see ``src/tracking/contact_side.py``).
        """
        if not getattr(self, "enable_contact_side", True):
            stats.contact_side = {"status": "disabled"}
            return
        from src.tracking.contact_side import assign_contact_sides
        try:
            report = assign_contact_sides(
                stats.all_events, stats.frame_annotations, fps,
                getattr(self, "calibration", None),
            )
        except Exception as error:
            # Additive output: a failure here must not cost the analysis.
            log.warning(f"Contact side failed, events keep no side: {error}")
            for event in stats.all_events:
                for key in ("side", "side_score", "side_source", "side_confident", "side_ball_px"):
                    event.pop(key, None)
            stats.contact_side = {"status": "error", "error": str(error)}
            return
        stats.contact_side = report
        if report["status"] == "completed":
            log.info(
                f"Contact side: {report['assigned']}/{report['contacts']} contacts "
                f"(far {report['sides']['a']}, near {report['sides']['b']}), "
                f"net-plane ball {report['net_ball_px']:.1f} px"
            )
        else:
            log.info(f"Contact side: not assigned ({report['status']})")

    def _recompute_events_on_clean_trajectory(
        self, annotations: List[Dict], fps: float,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Re-run the FSM over the cleaned trajectory.

        Per-frame ``state``/``phase``/``ev`` fields are rewritten in place so the
        viewer overlay agrees with the saved rally list. Returns all events,
        confirmed playable rallies, and every terminal service attempt.
        """
        fsm = self._make_fsm(fps)
        events: List[Dict] = []
        rallies: List[Dict] = []
        service_attempts: List[Dict] = []
        cur_events: List[Dict] = []
        prev_state: Optional[str] = None
        frames_since_ball = 0
        last_frame_idx = 0
        reconstruction_mode = getattr(self, "ball_reconstruction_mode", "off")
        reconstruction_affects_fsm = bool(
            getattr(self, "reconstruction_affects_fsm", False)
        )

        def event_dict(ev) -> Dict:
            return {
                "frame": ev.frame_idx,
                "time": round(ev.timestamp, 2),
                "type": ev.event_type,
                "game_state": str(ev.game_state.value),
                "rally_phase": (
                    str(ev.rally_phase.value) if ev.rally_phase else None
                ),
                "team": ev.team_controlling,
                "contact_seq": ev.contacts_count,
                "lifecycle": dict(ev.lifecycle) if ev.lifecycle else None,
            }

        def finish_attempt(end_frame: int, lifecycle: Dict) -> None:
            nonlocal cur_events
            if lifecycle.get("status") not in {
                "playable_rally",
                "ace_no_reception",
                "reception_failed",
                "serve_fault",
            }:
                cur_events = []
                return

            cur_events.sort(key=lambda e: e["frame"])
            start_frame = lifecycle.get("start_frame")
            if start_frame is None:
                start_frame = lifecycle.get("serve_frame")
            if start_frame is None and cur_events:
                start_frame = cur_events[0]["frame"]
            if start_frame is None:
                cur_events = []
                return

            duration = max(0, int(end_frame) - int(start_frame))
            attempt = {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "duration_frames": duration,
                "duration_sec": round(duration / fps, 2),
                "events": list(cur_events),
                "contacts_team_a": sum(
                    1 for e in cur_events if e["team"] == "team_a"
                ),
                "contacts_team_b": sum(
                    1 for e in cur_events if e["team"] == "team_b"
                ),
                "lifecycle": dict(lifecycle),
            }
            service_attempts.append(attempt)
            if lifecycle.get("is_playable"):
                rallies.append(attempt)
            cur_events = []

        for ann in annotations:
            frame_idx = int(ann["f"])
            last_frame_idx = frame_idx
            ball = ann.get("ball")
            if reconstruction_mode != "off":
                if not reconstruction_affects_fsm:
                    ball = ann.get("ball_raw")
                elif ann.get("ball_source") != "detected" \
                        and float(ann.get("ball_confidence") or 0.0) < 0.8:
                    ball = None
            ball_pos = (float(ball[0]), float(ball[1])) if ball else None
            frames_since_ball = 0 if ball_pos else frames_since_ball + 1

            teams: Dict[str, List[Tuple[float, float]]] = {"team_a": [], "team_b": []}
            for pose in ann.get("poses") or []:
                key = "team_a" if pose.get("team") in ("a", "team_a") else "team_b"
                bb = pose.get("bb")
                if bb:
                    teams[key].append(((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0))

            pre_state = fsm.get_current_status().get("game_state", "")
            if (frames_since_ball >= self.NO_BALL_RALLY_TIMEOUT
                    and pre_state in ("rally_active", "serving")):
                fsm._end_rally(frame_idx, "no_ball_timeout")
                frames_since_ball = 0

            fsm_events = fsm.update(
                frame_idx=frame_idx,
                ball_pos=ball_pos,
                player_positions=teams,
                timestamp=frame_idx / fps,
                # Both an interpolated fill and a track restart make the local
                # trajectory unusable for measuring a turn.
                ball_interp=bool(
                    ann.get("ball_new")
                    or (
                        ann.get("ball_interp")
                        and not (
                            reconstruction_affects_fsm
                            and ann.get("ball_contact_reconstructed")
                            and float(ann.get("ball_confidence") or 0.0) >= 0.8
                        )
                    )
                ),
            )
            status = fsm.get_current_status()
            state = status.get("game_state")

            ann["state"] = state
            ann["phase"] = status.get("rally_phase")
            ann["ev"] = [e.event_type for e in fsm_events]
            ann["lifecycle"] = status.get("lifecycle")
            ann["current_team"] = status.get("current_team")
            ann["contacts"] = status.get("team_contacts", 0)

            for ev in fsm_events:
                d = event_dict(ev)
                events.append(d)
                cur_events.append(d)

            if state == "rally_end" and prev_state != "rally_end":
                finish_attempt(frame_idx, status.get("lifecycle") or {})
            prev_state = state

        # Preserve an attempt cut off by the end of the source video.
        tail_status = fsm.get_current_status()
        tail_lifecycle = tail_status.get("lifecycle") or {}
        if (
            tail_status.get("game_state") in ("serving", "rally_active")
            and tail_lifecycle.get("serve_frame") is not None
        ):
            fsm._end_rally(last_frame_idx, "video_end")
            for ev in fsm.update(
                frame_idx=last_frame_idx + 1,
                ball_pos=None,
                player_positions={"team_a": [], "team_b": []},
                timestamp=(last_frame_idx + 1) / fps,
            ):
                d = event_dict(ev)
                events.append(d)
                cur_events.append(d)
            finish_attempt(
                last_frame_idx,
                fsm.get_current_status().get("lifecycle") or {},
            )

        events.sort(key=lambda e: e["frame"])
        log.info(
            "Clean-trajectory pass: "
            f"{len(service_attempts)} service attempts, "
            f"{len(rallies)} playable rallies, {len(events)} events"
        )
        return events, rallies, service_attempts
