"""Court calibration loading/scaling, projector and net detection.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np

from src.detection.player_pose import CourtProjector
from src.utils.logger import get_logger
from src.models.inference_cache import file_digest
from src.pipeline.analyzer.defaults import _ROOT
from src.pipeline.analyzer.types import CalibrationData

log = get_logger("Analyzer")


class CalibrationMixin:

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_calibration(self, path: Optional[str]) -> Optional[CalibrationData]:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        if not p.exists():
            log.warning(f"Calibration not found: {p} — using defaults")
            return None

        with open(p) as f:
            data = json.load(f)

        net_pts   = data.get("net_line_points", [])
        back_pts  = data.get("back_line_points", [])
        left_pts  = data.get("left_sideline_points", [])
        right_pts = data.get("right_sideline_points", [])

        net_y      = float(np.mean([p[1] for p in net_pts]))   if net_pts   else 800.0
        net_xs     = [p[0] for p in net_pts]
        court_bot  = float(np.mean([p[1] for p in back_pts]))  if back_pts  else 1500.0
        court_left = float(min(p[0] for p in left_pts))        if left_pts  else 0.0
        court_right= float(max(p[0] for p in right_pts))       if right_pts else float(data.get("frame_width", 3840))

        # Load SAM-detected net tape positions if available.
        # net_*_y_720p are stored in 720p space — convert immediately to the
        # native calibration resolution so all CalibrationData fields share
        # one coordinate system (then _scale_calibration scales uniformly).
        sam = data.get("net_detection_sam", {})
        native_h = float(data.get("frame_height", 2160))
        sam_sy = native_h / 720.0
        net_tape_top    = float(sam["net_top_y_720p"])    * sam_sy if sam.get("net_top_y_720p")    else None
        net_tape_bottom = float(sam["net_bottom_y_720p"]) * sam_sy if sam.get("net_bottom_y_720p") else None
        floor_left_720 = sam.get("net_floor_left_720p")
        floor_right_720 = sam.get("net_floor_right_720p")
        native_w = float(data.get("frame_width", 3840))
        net_floor_left = (
            (float(floor_left_720[0]) * native_w / 1280.0,
             float(floor_left_720[1]) * sam_sy)
            if floor_left_720 else None
        )
        net_floor_right = (
            (float(floor_right_720[0]) * native_w / 1280.0,
             float(floor_right_720[1]) * sam_sy)
            if floor_right_720 else None
        )
        net_floor_method = str(sam.get("net_floor_method")) if floor_left_720 and floor_right_720 else None
        stored_net_height = sam.get("net_height_m")
        if stored_net_height is None:
            stored_net_height = (
                data.get("metadata", {}).get("court_geometry", {}).get("net_height", 2.43)
            )
        net_height_m = float(stored_net_height)
        net_post_spacing_m = (
            float(sam["net_post_spacing_m"])
            if sam.get("net_post_spacing_m") is not None else None
        )
        raw_corners = data.get("court_corners")
        court_corners = (
            [(float(point[0]), float(point[1])) for point in raw_corners]
            if raw_corners and len(raw_corners) == 4 else None
        )
        if net_floor_left and net_floor_right:
            # Calibrations persisted before the floor line was levelled still
            # carry a noise tilt, so re-level on load instead of requiring a
            # fresh detection run.
            from src.detection.net_detector import (
                anchor_net_top_to_calibration,
                infer_net_bottom_y_from_floor,
                level_floor_line,
            )
            if net_floor_method != "manual_court_midline":
                net_floor_left, net_floor_right = level_floor_line(net_floor_left, net_floor_right)
            if net_tape_top is not None:
                anchored_top = anchor_net_top_to_calibration(
                    net_tape_top,
                    net_y,
                    net_height_hint=60.0 * sam_sy,
                )
                if anchored_top != net_tape_top:
                    log.warning(
                        f"Stored SAM net top {net_tape_top:.0f}px is inconsistent "
                        f"with the calibrated tape {net_y:.0f}px; repairing it"
                    )
                net_tape_top = anchored_top
                floor_y = (net_floor_left[1] + net_floor_right[1]) / 2.0
                net_tape_bottom = infer_net_bottom_y_from_floor(
                    net_tape_top,
                    floor_y,
                    net_height_m,
                )

        return self._validated_net_floor(CalibrationData(
            net_y=net_y,
            net_x_left=float(min(net_xs)) if net_xs else court_left,
            net_x_right=float(max(net_xs)) if net_xs else court_right,
            # court_top is set very high in the frame (low y) so that balls in the air
            # (above the net) are NOT flagged as "out of bounds".
            court_top=50.0,
            court_bottom=court_bot,
            court_left=court_left,
            court_right=court_right,
            frame_width=data.get("frame_width", 3840),
            frame_height=data.get("frame_height", 2160),
            net_tape_top=net_tape_top,
            net_tape_bottom=net_tape_bottom,
            net_tape_endpoint_y=tuple(float(point[1]) for point in sorted(net_pts))
            if net_floor_method == "manual_court_midline" and len(net_pts) == 2 else None,
            net_floor_left=net_floor_left,
            net_floor_right=net_floor_right,
            net_floor_method=net_floor_method,
            net_height_m=net_height_m,
            net_post_spacing_m=net_post_spacing_m,
            homography=data.get("perspective_matrix"),
            court_corners=court_corners,
        ))

    @staticmethod
    def _validated_net_floor(calibration: Optional[CalibrationData]) -> Optional[CalibrationData]:
        """Reject a geometrically impossible detected divider, in memory only.

        An invalid Hough/SAM floor line can put both teams on one side; the
        six-player cap then drops real players. Use the calibrated court's
        centre line at the detected post X coordinates, preserving its slope.
        """
        c = calibration
        if c is None or c.homography is None or not (c.net_floor_left and c.net_floor_right):
            return c
        if c.net_floor_method in {"calibrated_court_midline_fallback", "manual_court_midline"}:
            return c
        from src.detection.net_detector import NetDetector, infer_net_bottom_y_from_floor
        spacing = c.net_post_spacing_m
        if spacing is None:
            spacing = NetDetector.estimate_post_spacing_m(
                (c.net_floor_left[0] * 1280 / c.frame_width,
                 c.net_floor_left[1] * 720 / c.frame_height),
                (c.net_floor_right[0] * 1280 / c.frame_width,
                 c.net_floor_right[1] * 720 / c.frame_height),
                c.homography, c.frame_width, c.frame_height,
            )
        if NetDetector.post_spacing_is_plausible(spacing):
            return c
        try:
            projector = CourtProjector(c.homography)
            a = projector.world_to_image((-projector.half_width, 0.0))
            b = projector.world_to_image((projector.half_width, 0.0))
            if not np.isfinite([*a, *b]).all() or abs(b[0] - a[0]) < 1e-6:
                return c
            def on_midline(x):
                return (float(x), float(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0])))
            left, right = on_midline(c.net_floor_left[0]), on_midline(c.net_floor_right[0])
            bottom = c.net_tape_bottom
            if c.net_tape_top is not None:
                bottom = infer_net_bottom_y_from_floor(c.net_tape_top, (left[1] + right[1]) / 2, c.net_height_m)
        except (ValueError, TypeError, ZeroDivisionError, np.linalg.LinAlgError):
            return c
        log.warning(
            f"Rejecting {c.net_floor_method} team divider (post spacing {spacing:.2f} m); "
            "using calibrated court midline. Source calibration is unchanged."
        )
        return replace(c, net_floor_left=left, net_floor_right=right,
                       net_floor_method="calibrated_court_midline_fallback",
                       net_post_spacing_m=None, net_tape_bottom=bottom)

    def _scale_calibration(self, actual_w: int, actual_h: int) -> "CalibrationData":
        """Return calibration scaled to actual frame size, with optional net_y override."""
        c = self._validated_net_floor(self.calibration)
        if c is None:
            # No calibration: build defaults with override if provided
            net_y = self.net_y_override if self.net_y_override else actual_h * 0.5
            return CalibrationData(
                net_y=net_y, net_x_left=0.0, net_x_right=float(actual_w),
                court_top=10.0, court_bottom=float(actual_h),
                court_left=0.0, court_right=float(actual_w),
                frame_width=actual_w, frame_height=actual_h,
            )
        if c.frame_width != actual_w or c.frame_height != actual_h:
            sx = actual_w / c.frame_width
            sy = actual_h / c.frame_height
            log.info(f"Scaling calibration {c.frame_width}x{c.frame_height} "
                     f"→ {actual_w}x{actual_h}  (sx={sx:.3f} sy={sy:.3f})")
            # Homography maps world → calibration pixels; compose with the
            # pixel scaling so it maps world → actual-resolution pixels.
            homography_scaled = None
            if c.homography is not None:
                S = np.diag([sx, sy, 1.0])
                homography_scaled = (S @ np.asarray(c.homography, dtype=np.float64)).tolist()
            c = CalibrationData(
                net_y=c.net_y * sy,
                net_x_left=c.net_x_left * sx,
                net_x_right=c.net_x_right * sx,
                court_top=c.court_top * sy,
                court_bottom=c.court_bottom * sy,
                court_left=c.court_left * sx,
                court_right=c.court_right * sx,
                frame_width=actual_w,
                frame_height=actual_h,
                net_tape_top=c.net_tape_top * sy if c.net_tape_top else None,
                net_tape_bottom=c.net_tape_bottom * sy if c.net_tape_bottom else None,
                net_tape_endpoint_y=tuple(y * sy for y in c.net_tape_endpoint_y)
                if c.net_tape_endpoint_y is not None else None,
                net_floor_left=(c.net_floor_left[0] * sx, c.net_floor_left[1] * sy)
                    if c.net_floor_left else None,
                net_floor_right=(c.net_floor_right[0] * sx, c.net_floor_right[1] * sy)
                    if c.net_floor_right else None,
                net_floor_method=c.net_floor_method,
                net_height_m=c.net_height_m,
                net_post_spacing_m=c.net_post_spacing_m,
                homography=homography_scaled,
                court_corners=[
                    (point[0] * sx, point[1] * sy)
                    for point in c.court_corners
                ] if c.court_corners else None,
            )
        if self.net_y_override is not None:
            log.info(f"net_y override: {c.net_y:.1f} -> {self.net_y_override}")
            c = CalibrationData(
                net_y=self.net_y_override,
                net_x_left=c.net_x_left, net_x_right=c.net_x_right,
                court_top=c.court_top, court_bottom=c.court_bottom,
                court_left=c.court_left, court_right=c.court_right,
                frame_width=c.frame_width, frame_height=c.frame_height,
                net_tape_top=c.net_tape_top, net_tape_bottom=c.net_tape_bottom,
                net_tape_endpoint_y=c.net_tape_endpoint_y,
                net_floor_left=c.net_floor_left, net_floor_right=c.net_floor_right,
                net_floor_method=c.net_floor_method,
                net_height_m=c.net_height_m,
                net_post_spacing_m=c.net_post_spacing_m,
                homography=c.homography,
                court_corners=c.court_corners,
            )
        return c

    def _get_projector(self) -> Optional[CourtProjector]:
        """CourtProjector for the current (scaled) calibration, cached."""
        c = self.calibration
        if c is None or c.homography is None:
            return None
        if self._projector_calib is not c:  # calibration object changed (rescaled)
            try:
                floor_line = None
                net_tape_band = None
                if c.net_floor_left and c.net_floor_right:
                    floor_line = (c.net_floor_left, c.net_floor_right)
                if (
                    c.net_floor_left and c.net_floor_right
                    and c.net_tape_top is not None and c.net_tape_bottom is not None
                ):
                    x1, x2 = sorted((c.net_floor_left[0], c.net_floor_right[0]))
                    net_tape_band = (x1, x2, c.net_tape_top, c.net_tape_bottom)
                self._projector = CourtProjector(
                    c.homography,
                    net_floor_line=floor_line,
                    net_tape_band=net_tape_band,
                )
            except Exception as e:
                log.warning(f"Court projector unavailable ({e}) — using x-range player filter")
                self._projector = None
            self._projector_calib = c
        return self._projector

    def _maybe_run_sam_net_detection(self, video_path, frame_w, frame_h, refresh_floor=False):
        cache = getattr(self, '_model_cache', None)
        if cache is None or not self.enable_net_detection:
            return self._run_net_detection_uncached(video_path, frame_w, frame_h, refresh_floor)
        import os
        resolved = self._resolve('net_sam')
        worker = _ROOT/'scripts/detection/detect_net_posts_sam3.py'
        python = Path(os.environ.get('SAM3_PYTHON') or (_ROOT/'.sam3_env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')))
        sam3 = {'python':str(python),'available':python.is_file(),
                'worker_sha256':file_digest(worker) if worker.is_file() else None}
        if python.is_file():
            try:
                from huggingface_hub import try_to_load_from_cache
                for name in ('sam3.pt','config.json'):
                    path = try_to_load_from_cache('facebook/sam3',name)
                    sam3[name] = file_digest(path) if isinstance(path,str) and Path(path).is_file() else None
            except ImportError:
                sam3['checkpoint_resolution']='unavailable'
        base = {'weights': file_digest(resolved.path) if resolved.path.is_file() else None,
                'backend': file_digest(_ROOT/'src/detection/net_detector.py'),
                'size': [frame_w,frame_h], 'net_height_m': self.net_height_m,
                'refresh_floor': refresh_floor,
                'sam3_worker':sam3}
        def stage():
            calibration = asdict(self.calibration) if self.calibration else None
            if calibration is not None and calibration["net_tape_endpoint_y"] is None:
                calibration.pop("net_tape_endpoint_y")
            return cache.stage('net_geometry', {**base, 'calibration': calibration})
        key = stage()
        def infer():
            self._run_net_detection_uncached(video_path, frame_w, frame_h, refresh_floor)
            return {'calibration': asdict(self.calibration) if self.calibration else None,
                    'models': {k:v for k,v in self.model_provenance.items() if k.startswith('net_')}}
        result = cache.compute(key, 'result', infer)
        self.calibration = CalibrationData(**result['calibration']) if result['calibration'] else None
        self.model_provenance.update(result['models'])
        cache.put(stage(), 'result', result)  # Calibration write-back must not force another SAM pass.

    def _run_net_detection_uncached(
        self,
        video_path: str,
        frame_w: float,
        frame_h: float,
        refresh_floor: bool = False,
    ):
        """Run SAM net-tape segmentation if not already in calibration.

        Results (720p space) are persisted into the calibration JSON so this
        runs once per camera setup; in-memory calibration gets the values in
        native calibration resolution.
        """
        if not self.enable_net_detection:
            log.info("Net detection disabled for this analysis")
            return

        c = self.calibration
        if (c is not None and c.net_tape_top is not None
                and c.net_floor_left is not None and c.net_floor_right is not None
                and c.net_floor_method == "sam3_post_x_physical_height"
                and abs(c.net_height_m - self.net_height_m) < 1e-6
                and not refresh_floor):
            log.info(f"Net tape already calibrated: top={c.net_tape_top:.0f}px "
                     f"bottom={c.net_tape_bottom:.0f}px (calibration px)")
            return

        log.info("Net tape not in calibration — running SAM net segmentation on first 5 s …")
        try:
            from src.detection.net_detector import (
                NetDetector,
                infer_net_bottom_y_from_floor,
                save_sam_net_to_calibration,
            )

            net_y = c.net_y if c else frame_h * 0.5
            net_x_left = c.net_x_left if c else 0.0
            net_x_right = c.net_x_right if c else frame_w

            detector = NetDetector(net_height_m=self.net_height_m,
                                   use_model_registry=self.use_model_registry)
            if detector.model_provenance:
                self.model_provenance["net_sam"] = detector.model_provenance
            if c is not None and c.net_tape_top is not None and c.net_tape_bottom is not None:
                top = c.net_tape_top * 720.0 / c.frame_height
                bot = c.net_tape_bottom * 720.0 / c.frame_height
                log.info("Net tape already calibrated; detecting only post bases")
            else:
                top, bot = detector.detect_from_video(
                    video_path=video_path,
                    net_y_hint=net_y,
                    net_x_left=net_x_left,
                    net_x_right=net_x_right,
                    frame_h=c.frame_height if c else frame_h,
                    frame_w=c.frame_width if c else frame_w,
                )
            if top is None:
                log.warning("SAM net detection returned no result — net tape bounds unavailable")
                return

            # In-memory: convert 720p → native calibration resolution
            floor_720 = detector.detect_floor_line_sam3(
                video_path=video_path,
                calibration_path=str(self._calibration_path) if self._calibration_path else "",
                net_tape_top=top,
                net_tape_bottom=bot,
            )
            floor_method = "sam3_post_x_physical_height" if floor_720 else None
            floor_spacing_m = None
            if floor_720 is not None and c is not None:
                floor_spacing_m = detector.estimate_post_spacing_m(
                    floor_720[0], floor_720[1], c.homography,
                    c.frame_width, c.frame_height,
                )
                if not detector.post_spacing_is_plausible(floor_spacing_m):
                    log.warning(
                        f"SAM3 post spacing {floor_spacing_m:.2f} m is outside "
                        "the expected 10-11 m; trying Hough fallback"
                    )
                    floor_720 = None
                    floor_method = None
            if floor_720 is None:
                floor_x_priors = detector.project_net_floor_x_priors(
                    c.homography if c else None,
                    c.frame_width if c else frame_w,
                    c.frame_height if c else frame_h,
                )
                floor_720 = detector.detect_floor_line_from_video(
                    video_path=video_path,
                    net_x_left=net_x_left,
                    net_x_right=net_x_right,
                    frame_h=c.frame_height if c else frame_h,
                    frame_w=c.frame_width if c else frame_w,
                    net_tape_top=top * (c.frame_height if c else frame_h) / 720.0,
                    net_tape_bottom=bot * (c.frame_height if c else frame_h) / 720.0,
                    floor_x_priors=floor_x_priors,
                )
                floor_method = "hough_post_x_physical_height" if floor_720 else None
                if floor_720 is not None and c is not None:
                    floor_spacing_m = detector.estimate_post_spacing_m(
                        floor_720[0], floor_720[1], c.homography,
                        c.frame_width, c.frame_height,
                    )
                    if not detector.post_spacing_is_plausible(floor_spacing_m):
                        log.warning(
                            f"Hough post spacing {floor_spacing_m:.2f} m is outside "
                            "the expected 10-11 m; keep it only as a low-confidence fallback"
                        )
            if floor_spacing_m is not None:
                log.info(f"Net post spacing: {floor_spacing_m:.2f} m (expected 10-11 m)")

            if floor_720 is not None:
                floor_y_720 = (floor_720[0][1] + floor_720[1][1]) / 2.0
                physics_bot = infer_net_bottom_y_from_floor(
                    top,
                    floor_y_720,
                    self.net_height_m,
                )
                log.info(
                    f"Net mesh bottom from physical proportions: {bot:.1f}px -> "
                    f"{physics_bot:.1f}px (1.00m / {self.net_height_m:.2f}m)"
                )
                bot = physics_bot

            if c is not None:
                sam_sy = c.frame_height / 720.0
                self.calibration = CalibrationData(
                    net_y=c.net_y, net_x_left=c.net_x_left, net_x_right=c.net_x_right,
                    court_top=c.court_top, court_bottom=c.court_bottom,
                    court_left=c.court_left, court_right=c.court_right,
                    frame_width=c.frame_width, frame_height=c.frame_height,
                    net_tape_top=top * sam_sy, net_tape_bottom=bot * sam_sy,
                    net_floor_left=(floor_720[0][0] * c.frame_width / 1280.0,
                                    floor_720[0][1] * sam_sy) if floor_720 else c.net_floor_left,
                    net_floor_right=(floor_720[1][0] * c.frame_width / 1280.0,
                                     floor_720[1][1] * sam_sy) if floor_720 else c.net_floor_right,
                    net_floor_method=floor_method or c.net_floor_method,
                    net_height_m=self.net_height_m,
                    net_post_spacing_m=floor_spacing_m,
                    homography=c.homography,
                    court_corners=c.court_corners,
                )
            # Persist original 720p values for future runs
            if self._calibration_path and self._calibration_path.exists():
                save_sam_net_to_calibration(
                    str(self._calibration_path), top, bot, floor_720, floor_method,
                    net_height_m=self.net_height_m,
                    net_post_spacing_m=floor_spacing_m,
                )

        except Exception as exc:
            log.warning(f"SAM net detection failed ({exc}) — continuing without net tape bounds")
