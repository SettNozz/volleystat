"""Offline OSNet roster identity and contact-author attribution.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logger import get_logger
from src.pipeline.model_prefetch import BackgroundEncoder
from src.models.inference_cache import ModelOutputCache, FrozenOSNet, osnet_feature_store
from src.pipeline.analyzer.defaults import _ROOT

log = get_logger("Analyzer")


class IdentityMixin:

    def _identity_reid(self):
        """Pinned OSNet encoder plus cache-backed crop features, built once per analysis.

        Shared by the in-loop embedding prefetch and the identity pass so the
        weights are resolved and loaded once. Raises FileNotFoundError when the
        bundle is not installed.
        """
        built = getattr(self, '_identity_reid_built', None)
        cache = getattr(self, '_model_cache', None)
        if built is not None and built[0] is cache:
            return built[1:]
        from src.models.player_reid_encoder import DEFAULT_ASSETS, OSNetEncoder
        assets = Path(self.player_reid_assets) if self.player_reid_assets else DEFAULT_ASSETS
        if not assets.is_absolute():
            assets = _ROOT / assets
        override = str(assets / "osnet_msmt17.pth") if self.player_reid_assets else None
        resolved = self._resolve("player_reid", override)
        feature_cache = None
        if cache is None:
            encoder = OSNetEncoder(assets, resolved.path, self.player_reid_device)
        else:
            encoder = FrozenOSNet(assets, resolved.path, self.player_reid_device)
            size = getattr(self, '_cache_frame_size', None)
            if size is None:
                size = (self.calibration.frame_width,self.calibration.frame_height)
            width,height = size
            feature_cache = osnet_feature_store(cache, assets, resolved.path, width,height)
        self._identity_reid_built = (cache, resolved, encoder, feature_cache)
        return resolved, encoder, feature_cache

    def _start_embedding_prefetch(self) -> None:
        """Encode every saved pose crop while the loop still holds its frame.

        The identity pass (and the contact-author runtime) read OSNet features
        by frame + clamped rounded box from the model-output cache. Computing
        them here, from the exact ``ann['poses'][i]['bb']`` they will look up,
        means a cold run no longer decodes the whole video a second time for
        them. Same encoder, crop and cache key: a value is identical to one the
        identity pass would compute (up to OSNet batch composition, <=1.3e-6
        cosine distance). Anything missed still falls back to decoding.
        """
        self._embedding_prefetch = None
        cache = getattr(self, '_model_cache', None)
        if (not getattr(self, 'enable_player_identity', False) or not getattr(self, 'model_prefetch', False)
                or cache is None or cache.require):
            return
        try:
            _, encoder, feature_cache = self._identity_reid()
        except Exception as exc:  # the identity pass reports it properly later
            log.info(f"OSNet embedding prefetch unavailable: {exc}")
            return
        if feature_cache is None:
            return
        from src.models.inference_cache import OSNET_BATCH_SIZE

        def encode(crops):
            # Serialisation (JSON, zlib, sha256) also runs here, off the loop.
            return [feature_cache.encode(value) for value in encoder.encode(crops)]
        known = feature_cache.cache.items(feature_cache.stage)
        self._embedding_prefetch = (BackgroundEncoder(encode, OSNET_BATCH_SIZE), encoder, feature_cache, known)
        log.info("OSNet crop embeddings computed during the detection pass")

    def _prefetch_embeddings(self, ann: Dict[str, Any], frame: Optional[np.ndarray]) -> None:
        state = getattr(self, '_embedding_prefetch', None)
        if state is None or frame is None:
            return
        worker, encoder, store, known = state
        frame_index = int(ann['f'])
        for pose in ann.get('poses') or []:
            box = pose.get('bb')
            if not box or len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            key = store.key(frame_index, box)
            if key in known:
                continue
            try:
                crop = encoder.crop(frame, box).copy()
            except ValueError:
                continue
            known.add(key)
            worker.add((frame_index, tuple(box)), crop)
        self._store_prefetched_embeddings()

    def _store_prefetched_embeddings(self, wait: bool = False) -> None:
        state = getattr(self, '_embedding_prefetch', None)
        if state is None:
            return
        worker, _, store, _ = state
        ready = worker.drain(wait=wait)
        if ready:
            store.cache.computed['osnet_embeddings'] += len(ready)
            store.put_encoded([(frame_index, list(box), body) for (frame_index, box), body in ready])

    def _finish_embedding_prefetch(self, store_results: bool = True) -> None:
        state = getattr(self, '_embedding_prefetch', None)
        if state is None:
            return
        try:
            if store_results:
                self._store_prefetched_embeddings(wait=True)
        finally:
            self._embedding_prefetch = None
            state[0].close()

    def _apply_player_identity(self, stats: "GameStats", images, fps: float,
                               video_path: Optional[Path] = None) -> None:
        """Resolve roster membership before final features, FSM and statistics."""
        if not getattr(self, "enable_player_identity", False):
            stats.player_identity = {"enabled": False, "status": "disabled"}
            return
        from src.tracking.player_identity_workflow import METHOD, run_identity_pass
        started = time.time()
        # Filled by the whole-video roster pass; read by _apply_player_substitutions
        # once the final rallies exist, then released.
        self._identity_context = {}
        side = self.player_identity_side
        cache = getattr(self, '_model_cache', None)
        if cache is None and video_path is not None and hasattr(self, 'model_cache_dir'):
            cache = self._model_cache = ModelOutputCache(video_path, self.model_cache_dir, self.require_model_cache)
        try:
            resolved, encoder, feature_cache = self._identity_reid()
        except FileNotFoundError as exc:
            if self.player_reid_assets or self.player_identity_seed:
                raise
            log.warning(f"Player identity unavailable: {exc}. Run scripts/models/install_player_reid.py")
            stats.player_identity = {"enabled": True, "status": "model_unavailable", "method": METHOD,
                                     "side": side, "error": str(exc)}
            for ann in stats.frame_annotations:
                for pose in ann.get("poses") or []:
                    if pose.get("team") in (side, f"team_{side}"):
                        pose["player_id"] = None
                        pose["identity"] = {"method": METHOD, "reason": "model_unavailable",
                                            "label_source": "model", "verified": False}
        else:
            def progress(done, total):
                if done % max(1, int(fps * 2)) == 0 or done == total:
                    elapsed = max(time.time() - started, .001)
                    self._emit_progress(phase="player_identity", progress=97.0 + done/max(total, 1),
                                        frame_current=done, frame_total=total,
                                        processing_fps=round(done/elapsed, 2),
                                        eta_sec=round(elapsed/max(done, 1)*(total-done), 1),
                                        message=f"Assigning stable player IDs {done:,}/{total:,}")
            seed = Path(self.player_identity_seed) if self.player_identity_seed else None
            if seed is not None and not seed.is_absolute():
                seed = _ROOT / seed
            stats.player_identity = run_identity_pass(
                stats.frame_annotations, images, fps, encoder, side=side,
                roster_size=self.max_players_per_team or 6, seed_path=seed,
                video_path=video_path, progress=progress,
                feature_cache=feature_cache,
                jersey_reads=getattr(self, "player_jersey_reads", None),
                method=getattr(self, "player_identity_method", "roster-discovery"),
                preserve_membership=True, identity_map=getattr(self, 'team_identity_map', None),
                segments=getattr(self, 'match_segments', ()),
                context=self._identity_context,
            )
            stats.player_identity["model_resolution"] = resolved.as_dict()
            self.model_provenance["player_reid"].update(encoder.provenance)
            log.info(f"Player identity: {stats.player_identity['status']}, "
                     f"{stats.player_identity.get('assigned', 0)} assigned observations")
        stats.models = dict(self.model_provenance)
        elapsed = time.time() - started
        stats.player_identity["processing_time_sec"] = round(elapsed, 3)
        stats.processing_time_sec += elapsed

    def _apply_player_substitutions(self, stats: "GameStats", fps: float) -> None:
        """Extend the roster with detected substitutes; needs the final rallies.

        Runs after the offline passes (substitutions happen only between rallies)
        and before contact authors, which must see the substitutes' IDs.
        """
        context = getattr(self, '_identity_context', None) or {}
        self._identity_context = None
        report = stats.player_identity or {}
        if not getattr(self, 'enable_player_substitutions', True):
            if report.get('status') == 'completed':
                report['substitutions'] = {'status': 'disabled'}
            return
        if report.get('status') != 'completed' or 'roster' not in context:
            if report.get('status') == 'completed':
                report['substitutions'] = {'status': 'not_applicable',
                                           'reason': 'needs the whole-video roster method'}
            return
        from src.tracking.player_substitutions import apply_substitutions
        started = time.time()
        rallies = stats.service_attempts or stats.rallies
        apply_substitutions(stats.frame_annotations, report, context, rallies, fps)
        block = report['substitutions']
        block['processing_time_sec'] = round(time.time() - started, 3)
        log.info(f"Player substitutions: {block['status']}, {len(block.get('events') or [])} detected")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _apply_contact_authors(self, stats, video, width, height, images=None):
        from src.tracking.contact_author_runtime import attribute_contacts
        payload = {"frames": stats.frame_annotations, "fps": stats.fps, "video": stats.video_name,
            "original_video_path": video, "frame_width": width, "frame_height": height,
            "player_identity": stats.player_identity}
        stats.player_contacts = attribute_contacts(payload, stats.all_events,
            enabled=getattr(self, "enable_contact_author", False),
            model_path=getattr(self, "contact_author_model", None), roster=getattr(self, "contact_author_roster", None),
            cache=getattr(self, "_model_cache", None), cache_dir=getattr(self, "model_cache_dir", None),
            assets=getattr(self, "player_reid_assets", None), device=getattr(self, "player_reid_device", "auto"),
            require_cache=getattr(self, "require_model_cache", False), use_registry=getattr(self, "use_model_registry", True), images=images)
        if stats.player_identity.get('team_id'):
            stats.player_contacts['team_id'] = stats.player_identity['team_id']
        provenance = stats.player_contacts.get("attribution", {}).get("model_resolution")
        if provenance:
            self.model_provenance["contact_author"] = provenance
            stats.models = dict(self.model_provenance)
        # Two independent side estimates now exist for a contact. Record where
        # they disagree for review; neither report is rewritten.
        if (getattr(stats, "contact_side", None) or {}).get("status") == "completed":
            from src.tracking.contact_side import cross_check_authors
            stats.contact_side["author_check"] = cross_check_authors(stats.all_events, stats.player_contacts)
