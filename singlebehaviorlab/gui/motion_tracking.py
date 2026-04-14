"""
Motion-Aware Tracking Enhancements for SAM2.

Includes:
- Kalman filter-based motion prediction (SAMURAI-style)
- OC-SORT drift correction (virtual trajectory + ORU)
- Mask quality scoring and temporal consistency checks
"""

import numpy as np
import torch
import logging


logger = logging.getLogger(__name__)


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes with OC-SORT enhancements.
    State: [x_center, y_center, scale, aspect_ratio, dx, dy, d_scale]
    
    OC-SORT features:
    - Virtual trajectory: During occlusion, generate virtual observations from velocity
    - ORU (Observation-Centric Re-Update): When object reappears, correct past states
    """
    def __init__(self, bbox, delta_t=1, inertia=0.2):
        """
        Initialize tracker with bounding box [x1, y1, x2, y2].
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            delta_t: Frame interval for virtual trajectory (OC-SORT paper: 1)
            inertia: Velocity smoothing factor for ORU (OC-SORT paper: 0.2)
        """
        from filterpy.kalman import KalmanFilter
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.delta_t = delta_t
        self.inertia = inertia
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Measurement noise covariance
        self.kf.R *= 10.0
        self.kf.R[2, 2] *= 10.0  # Scale measurement has higher noise
        
        # Process noise covariance
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocities initially
        self.kf.P *= 10.0
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox).reshape(4, 1)
        
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.age = 0
        
        # OC-SORT: Store observations for ORU
        self.observations = {}  # frame_idx -> observation (z)
        self.last_observation_frame = 0
        self.last_observation = self._bbox_to_z(bbox)
        self.frozen_velocity = None  # Velocity frozen at occlusion start
    
    def _bbox_to_z(self, bbox):
        """Convert [x1, y1, x2, y2] to [x_center, y_center, scale, aspect_ratio]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + w / 2.0
        y_c = y1 + h / 2.0
        s = w * h  # scale (area)
        r = w / (h + 1e-6)  # aspect ratio
        return np.array([x_c, y_c, s, r], dtype=np.float32)
    
    def _z_to_bbox(self, z):
        """Convert [x_center, y_center, scale, aspect_ratio] to [x1, y1, x2, y2]."""
        x_c, y_c, s, r = z.flatten()[:4]
        w = np.sqrt(max(s * r, 1.0))
        h = s / (w + 1e-6)
        return np.array([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2], dtype=np.float32)
    
    def predict(self):
        """Predict next state and return predicted bbox."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]
    
    def predict_with_virtual_trajectory(self, frame_idx):
        """
        OC-SORT: Predict with virtual trajectory during occlusion.
        Uses last known velocity to generate virtual observations.
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            Predicted bbox
        """
        if self.time_since_update == 0:
            # Not occluded, use normal prediction
            return self.predict()
        
        # Freeze velocity at start of occlusion
        if self.frozen_velocity is None:
            self.frozen_velocity = self.kf.x[4:7].copy()
        
        # Generate virtual observation using frozen velocity
        virtual_z = self.last_observation.copy()
        dt = frame_idx - self.last_observation_frame
        if dt > 0:
            virtual_z[0] += self.frozen_velocity[0, 0] * dt  # x_center
            virtual_z[1] += self.frozen_velocity[1, 0] * dt  # y_center
            virtual_z[2] += self.frozen_velocity[2, 0] * dt  # scale
            virtual_z[2] = max(virtual_z[2], 1.0)  # scale must stay positive

        # Fold the virtual observation into the Kalman state so predicted and
        # observed trajectories stay in sync during occlusions.
        self.kf.predict()
        self.kf.update(virtual_z)
        self.age += 1
        self.time_since_update += 1
        
        bbox = self._z_to_bbox(self.kf.x)
        self.history.append(bbox)
        return bbox
    
    def update(self, bbox, frame_idx=None):
        """Update state with observed bbox."""
        z = self._bbox_to_z(bbox)
        
        # OC-SORT: Apply ORU if recovering from occlusion
        if self.time_since_update > 0 and frame_idx is not None:
            self._apply_oru(z, frame_idx)
        
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.kf.update(z)
        
        # Store observation for future ORU
        if frame_idx is not None:
            self.observations[frame_idx] = z.copy()
            self.last_observation_frame = frame_idx
        self.last_observation = z.copy()
        self.frozen_velocity = None  # Reset frozen velocity
    
    def _apply_oru(self, new_z, frame_idx):
        """
        OC-SORT: Observation-Centric Re-Update.
        When object reappears after occlusion, interpolate backwards
        to correct past Kalman state drift.
        
        Args:
            new_z: New observation [x_c, y_c, s, r]
            frame_idx: Current frame index
        """
        if self.time_since_update <= 1:
            return  # ORU only matters after more than one missed frame

        dt = frame_idx - self.last_observation_frame
        if dt <= 0:
            return

        velocity = (new_z - self.last_observation) / dt
        
        # Apply velocity smoothing with inertia
        # Mix new velocity with old frozen velocity
        if self.frozen_velocity is not None:
            old_vel = self.frozen_velocity.flatten()[:3]
            new_vel = velocity[:3]
            smoothed_vel = self.inertia * old_vel + (1 - self.inertia) * new_vel
        else:
            smoothed_vel = velocity[:3]
        
        # Re-update state with corrected velocity
        self.kf.x[4, 0] = smoothed_vel[0]  # dx
        self.kf.x[5, 0] = smoothed_vel[1]  # dy
        self.kf.x[6, 0] = smoothed_vel[2]  # d_scale
        
        # Reduce uncertainty since we now have a good observation
        self.kf.P[4:, 4:] *= 0.5
    
    def get_state(self):
        """Get current bbox estimate."""
        return self._z_to_bbox(self.kf.x)
    
    def get_velocity(self):
        """Get current velocity estimate [dx, dy, d_scale]."""
        return self.kf.x[4:7].flatten().copy()


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x1, y1, x2, y2]."""
    if mask is None or mask.max() == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    # Use [x1, y1, x2, y2) convention (exclusive max corner) for consistent area math.
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def compute_mask_score(mask_logit, predicted_bbox=None, actual_bbox=None, 
                       use_multiplicative=True):
    """
    Compute quality score for a mask prediction (SAMURAI-style).
    
    SAMURAI formula: score = sigmoid(obj_score) * sigmoid(IoU)
    This multiplicative approach is stricter - if either component is low,
    the whole score drops significantly.
    
    Args:
        mask_logit: Raw mask logits from SAM2
        predicted_bbox: Predicted bbox from Kalman filter
        actual_bbox: Actual bbox from mask
        use_multiplicative: If True, use SAMURAI's multiplicative scoring.
                           If False, use additive (average).
    
    Returns:
        score: Quality score between 0 and 1
        obj_score: Objectness component
        motion_iou: Motion consistency component
    """
    # Objectness score: max confidence in the mask (already sigmoid for logits)
    if isinstance(mask_logit, torch.Tensor):
        obj_score = torch.sigmoid(mask_logit).max().item()
    else:
        obj_score = 1.0 / (1.0 + np.exp(-mask_logit.max()))
    
    # Motion consistency score (IoU between predicted and actual bbox)
    if predicted_bbox is not None and actual_bbox is not None:
        motion_iou = compute_iou(predicted_bbox, actual_bbox)
    else:
        motion_iou = 1.0  # No motion prediction available
    
    # SAMURAI applies sigmoid to IoU as well for normalization
    # sigmoid(IoU) maps [0,1] -> [0.5, 0.73] roughly, so we use a scaled sigmoid
    # that maps 0->0, 0.5->0.5, 1->1 for better range
    def soft_sigmoid(x, k=5.0):
        """Soft sigmoid that maps [0,1] to [0,1] with 0.5->0.5"""
        return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    
    if use_multiplicative:
        # SAMURAI-style: multiplicative scoring (stricter)
        # Both components must be high for good score
        normalized_iou = soft_sigmoid(motion_iou)
        score = obj_score * normalized_iou
    else:
        # Additive scoring (more lenient)
        score = 0.5 * obj_score + 0.5 * motion_iou
    
    return score, obj_score, motion_iou


class AppearanceMemoryBank:
    """
    Long-term appearance memory that stores high-quality "golden" masks per object.
    When an object is lost for many frames (long occlusion) and starts reappearing,
    the best stored mask is used to re-seed SAM2 so appearance is recovered.
    """
    def __init__(self, max_snapshots=5, min_score_to_store=0.6,
                 occlusion_enter_frames=10, recovery_area_ratio=0.05,
                 reseed_debounce_frames=10, shape_area_ratio_range=(0.7, 1.3),
                 max_aspect_ratio_change=0.35):
        """
        Args:
            max_snapshots: max golden masks kept per object (best-scoring kept)
            min_score_to_store: minimum quality score to consider a frame "golden"
            occlusion_enter_frames: consecutive zero/low frames before entering occlusion state
            recovery_area_ratio: when current area / golden area >= this, consider partial
                recovery. Set very low (0.05 = 5%) so even a sliver of tail triggers reseed.
            reseed_debounce_frames: min frames between consecutive re-seeds for same object
            shape_area_ratio_range: accepted candidate area ratio range vs learned shape prior
            max_aspect_ratio_change: max relative aspect-ratio change vs learned shape prior
        """
        self.max_snapshots = max_snapshots
        self.min_score_to_store = min_score_to_store
        self.occlusion_enter_frames = occlusion_enter_frames
        self.recovery_area_ratio = recovery_area_ratio
        self.reseed_debounce_frames = reseed_debounce_frames
        self.shape_area_ratio_range = shape_area_ratio_range
        self.max_aspect_ratio_change = max_aspect_ratio_change

        # Per-object storage
        self.snapshots = {}          # obj_id -> [(score, frame_idx, mask, bbox, area)]
        self.golden_area = {}        # obj_id -> median area from golden masks
        self.golden_aspect = {}      # obj_id -> median aspect ratio from golden masks
        self.zero_streak = {}        # obj_id -> consecutive frames with empty/tiny mask
        self.occluded = {}           # obj_id -> bool
        self.recovery_pending = {}   # obj_id -> bool (just left occlusion, needs re-seed)
        self.last_reseed_frame = {}  # obj_id -> frame_idx of last re-seed (debounce)

    @staticmethod
    def _bbox_aspect_ratio(bbox):
        if bbox is None:
            return None
        w = max(float(bbox[2] - bbox[0]), 1.0)
        h = max(float(bbox[3] - bbox[1]), 1.0)
        return w / h

    def _passes_shape_guard(self, obj_id, area, bbox):
        """Keep golden masks close to the learned prompt-shape profile."""
        ref_area = self.golden_area.get(obj_id)
        if ref_area is not None and ref_area > 0:
            ratio = float(area) / float(ref_area)
            lo, hi = self.shape_area_ratio_range
            if ratio < lo or ratio > hi:
                return False

        ref_aspect = self.golden_aspect.get(obj_id)
        cur_aspect = self._bbox_aspect_ratio(bbox)
        if ref_aspect is not None and ref_aspect > 0 and cur_aspect is not None:
            rel_change = abs(cur_aspect - ref_aspect) / ref_aspect
            if rel_change > self.max_aspect_ratio_change:
                return False
        return True

    def store_if_golden(self, obj_id, mask, bbox, area, score, frame_idx):
        """Store mask snapshot if quality is high enough."""
        if score < self.min_score_to_store or mask is None or area < 1:
            return
        if not self._passes_shape_guard(obj_id, area, bbox):
            return
        if obj_id not in self.snapshots:
            self.snapshots[obj_id] = []

        entry = (score, frame_idx, mask.copy(), bbox.copy() if bbox is not None else None, area)
        snaps = self.snapshots[obj_id]
        snaps.append(entry)
        snaps.sort(key=lambda e: e[0], reverse=True)
        if len(snaps) > self.max_snapshots:
            snaps[:] = snaps[:self.max_snapshots]

        # Refresh the reference ("golden") area as the median across snapshots.
        areas = [s[4] for s in snaps]
        self.golden_area[obj_id] = float(np.median(areas))
        aspects = [self._bbox_aspect_ratio(s[3]) for s in snaps if s[3] is not None]
        if aspects:
            self.golden_aspect[obj_id] = float(np.median(aspects))

    def update_occlusion_state(self, obj_id, mask_area, frame_idx):
        """
        Track whether object is in long occlusion and detect recovery.
        Returns True if a re-seed should happen this frame.
        """
        golden = self.golden_area.get(obj_id)
        if golden is None or golden < 1:
            return False

        area_fraction = mask_area / golden

        # Object is mostly gone?
        if area_fraction < 0.1:
            self.zero_streak[obj_id] = self.zero_streak.get(obj_id, 0) + 1
        else:
            self.zero_streak[obj_id] = 0

        was_occluded = self.occluded.get(obj_id, False)

        # Enter occlusion mode after sustained absence
        if self.zero_streak.get(obj_id, 0) >= self.occlusion_enter_frames:
            self.occluded[obj_id] = True

        # Detect recovery: was occluded, now partial mask is back
        if was_occluded and area_fraction >= self.recovery_area_ratio:
            self.occluded[obj_id] = False
            self.zero_streak[obj_id] = 0

            # Debounce: don't re-seed too frequently
            last = self.last_reseed_frame.get(obj_id, -999)
            if frame_idx - last >= self.reseed_debounce_frames:
                self.recovery_pending[obj_id] = True
                self.last_reseed_frame[obj_id] = frame_idx
                return True
        return False

    def pop_reseed_mask(self, obj_id):
        """
        Return the best golden mask for re-seeding, or None.
        Clears the pending flag.
        """
        self.recovery_pending.pop(obj_id, None)
        snaps = self.snapshots.get(obj_id)
        if not snaps:
            return None
        return snaps[0][2]  # highest-score snapshot mask array

    def is_recovery_pending(self, obj_id):
        return self.recovery_pending.get(obj_id, False)

    def has_snapshots(self, obj_id):
        return bool(self.snapshots.get(obj_id))


class MultiObjectMotionTracker:
    """
    Manages Kalman filter trackers for multiple objects with
    motion-aware tracking, drift detection, and automatic prompt injection.
    
    Supports OC-SORT enhancements:
    - Virtual trajectory during occlusions
    - ORU (Observation-Centric Re-Update) for drift correction
    """
    def __init__(self, motion_score_threshold=0.3, use_kalman=True,
                 consecutive_low_threshold=3, area_change_threshold=0.5,
                 use_ocsort=False, ocsort_inertia=0.2, max_history_frames=1000,
                 adaptive_threshold=True, threshold_window=30, hysteresis_margin=0.05,
                 max_correction_jump_px=80.0, max_correction_area_ratio=2.5,
                 enable_appearance_memory=True, appearance_min_score=0.6,
                 appearance_max_snapshots=5, occlusion_enter_frames=5,
                 recovery_area_ratio=0.15, reseed_debounce_frames=5,
                 shape_area_ratio_range=(0.7, 1.3), max_aspect_ratio_change=0.35):
        self.trackers = {}  # obj_id -> KalmanBoxTracker
        self.scores = {}    # obj_id -> {frame_idx: score}
        self.bboxes = {}    # obj_id -> {frame_idx: bbox} for temporal consistency
        self.areas = {}     # obj_id -> {frame_idx: area}
        self.motion_score_threshold = motion_score_threshold
        self.use_kalman = use_kalman
        self.kalman_available = True
        self.consecutive_low_threshold = consecutive_low_threshold  # Frames before recovery
        self.area_change_threshold = area_change_threshold  # Max allowed area change ratio
        self.consecutive_low_count = {}  # obj_id -> count of consecutive low scores
        self.needs_correction = {}  # obj_id -> bool
        
        # OC-SORT parameters
        self.use_ocsort = use_ocsort
        self.ocsort_inertia = ocsort_inertia  # Velocity smoothing (paper: 0.2)
        self.max_history_frames = max_history_frames
        self.adaptive_threshold = adaptive_threshold
        self.threshold_window = threshold_window
        self.hysteresis_margin = hysteresis_margin
        self.max_correction_jump_px = max_correction_jump_px
        self.max_correction_area_ratio = max_correction_area_ratio
        self.low_state = {}  # obj_id -> bool (hysteresis state)

        # Long-term appearance memory for occlusion recovery
        self.enable_appearance_memory = enable_appearance_memory
        self.appearance_memory = AppearanceMemoryBank(
            max_snapshots=appearance_max_snapshots,
            min_score_to_store=appearance_min_score,
            occlusion_enter_frames=occlusion_enter_frames,
            recovery_area_ratio=recovery_area_ratio,
            reseed_debounce_frames=reseed_debounce_frames,
            shape_area_ratio_range=shape_area_ratio_range,
            max_aspect_ratio_change=max_aspect_ratio_change,
        ) if enable_appearance_memory else None

        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            self.kalman_available = False
            self.use_kalman = False
    
    def initialize_tracker(self, obj_id, bbox, frame_idx=0):
        """Initialize a new tracker for an object."""
        if not self.use_kalman or not self.kalman_available:
            return
        if bbox is None:
            return
        try:
            self.trackers[obj_id] = KalmanBoxTracker(
                bbox, 
                delta_t=1, 
                inertia=self.ocsort_inertia
            )
            self.trackers[obj_id].last_observation_frame = frame_idx
            self.scores[obj_id] = {}
            self.bboxes[obj_id] = {}
            self.areas[obj_id] = {}
            self.consecutive_low_count[obj_id] = 0
            self.needs_correction[obj_id] = False
            self.low_state[obj_id] = False
        except Exception as e:
            logger.debug("Failed to initialize tracker for obj %s: %s", obj_id, e)
    
    def predict(self, obj_id):
        """Return the current Kalman state estimate without advancing it."""
        if obj_id not in self.trackers:
            return None
        try:
            return self.trackers[obj_id].get_state()
        except Exception as e:
            logger.debug("Failed to get state for obj %s: %s", obj_id, e)
            return None
    
    def predict_and_advance(self, obj_id, frame_idx=None):
        """Advance the Kalman state by one step and return the predicted bbox.

        When OC-SORT is enabled and a frame index is provided, the tracker
        uses a virtual trajectory so occlusions do not zero out velocity.
        """
        if obj_id not in self.trackers:
            return None
        try:
            if self.use_ocsort and frame_idx is not None:
                return self.trackers[obj_id].predict_with_virtual_trajectory(frame_idx)
            else:
                return self.trackers[obj_id].predict()
        except Exception as e:
            logger.debug("Failed to predict tracker for obj %s: %s", obj_id, e)
            return None
    
    def get_predicted_bbox_for_correction(self, obj_id):
        """Return the Kalman bbox as an [x1, y1, x2, y2] box prompt."""
        if obj_id not in self.trackers:
            return None
        try:
            bbox = self.trackers[obj_id].get_state()
            if bbox is None:
                return None
            return bbox.tolist()
        except Exception as e:
            logger.debug("Failed to get correction bbox for obj %s: %s", obj_id, e)
            return None

    def get_frame_score(self, obj_id, frame_idx):
        if obj_id not in self.scores:
            return None
        return self.scores[obj_id].get(frame_idx)

    def get_effective_threshold(self, obj_id):
        base = self.motion_score_threshold
        if not self.adaptive_threshold or obj_id not in self.scores:
            return base
        frames = sorted(self.scores[obj_id].keys())[-self.threshold_window:]
        if len(frames) < 5:
            return base
        vals = np.array([self.scores[obj_id][f] for f in frames], dtype=np.float32)
        mean = float(vals.mean())
        std = float(vals.std())
        adaptive = mean - 0.5 * std
        lower = max(0.05, base - 0.15)
        upper = min(0.95, base + 0.15)
        return float(np.clip(adaptive, lower, upper))

    def is_correction_bbox_sane(self, obj_id, pred_bbox):
        if pred_bbox is None or obj_id not in self.bboxes or not self.bboxes[obj_id]:
            return False
        last_frame = max(self.bboxes[obj_id].keys())
        last_bbox = self.bboxes[obj_id][last_frame]
        if last_bbox is None:
            return False

        pred_bbox = np.asarray(pred_bbox, dtype=np.float32)
        last_bbox = np.asarray(last_bbox, dtype=np.float32)

        pred_center = np.array(
            [(pred_bbox[0] + pred_bbox[2]) * 0.5, (pred_bbox[1] + pred_bbox[3]) * 0.5],
            dtype=np.float32,
        )
        last_center = np.array(
            [(last_bbox[0] + last_bbox[2]) * 0.5, (last_bbox[1] + last_bbox[3]) * 0.5],
            dtype=np.float32,
        )
        center_shift = float(np.linalg.norm(pred_center - last_center))
        if center_shift > self.max_correction_jump_px:
            return False

        pred_area = max((pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1]), 1.0)
        last_area = max((last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]), 1.0)
        area_ratio = max(pred_area / last_area, last_area / pred_area)
        return area_ratio <= self.max_correction_area_ratio
    
    def update(self, obj_id, mask, mask_logit, frame_idx):
        """
        Update tracker with observed mask and compute quality score.
        
        Returns:
            score: Quality score for this frame
            should_use_for_memory: Whether this frame should be used for memory
        """
        actual_bbox = mask_to_bbox(mask)
        
        if actual_bbox is None:
            if obj_id in self.consecutive_low_count:
                self.consecutive_low_count[obj_id] += 1
            # OC-SORT: Still predict with virtual trajectory to maintain state
            if self.use_ocsort and obj_id in self.trackers:
                try:
                    self.trackers[obj_id].predict_with_virtual_trajectory(frame_idx)
                except Exception as e:
                    logger.debug(
                        "Failed virtual trajectory for obj %s at frame %s: %s",
                        obj_id,
                        frame_idx,
                        e,
                    )
            return 0.0, False

        mask_area = float(np.sum(mask > 0))

        if obj_id not in self.trackers:
            self.initialize_tracker(obj_id, actual_bbox, frame_idx)
            if obj_id not in self.scores:
                self.scores[obj_id] = {}
            self.scores[obj_id][frame_idx] = 1.0
            self.bboxes[obj_id] = {frame_idx: actual_bbox}
            self.areas[obj_id] = {frame_idx: mask_area}
            self.low_state[obj_id] = False
            return 1.0, True

        # OC-SORT virtual trajectory: advance Kalman state through occlusions
        # using frame_idx as the time index.
        predicted_bbox = self.predict_and_advance(obj_id, frame_idx) if self.use_kalman else None

        score, obj_score, motion_iou = compute_mask_score(
            mask_logit, predicted_bbox, actual_bbox
        )

        # Temporal consistency: large area jumps usually indicate mask drift,
        # so penalise the score when the current area deviates from the
        # running mean over the last five frames.
        if obj_id in self.areas and self.areas[obj_id]:
            recent_frames = sorted(self.areas[obj_id].keys())[-5:]
            if recent_frames:
                avg_area = np.mean([self.areas[obj_id][f] for f in recent_frames])
                if avg_area > 0:
                    area_ratio = mask_area / avg_area
                    if area_ratio < (1 - self.area_change_threshold) or \
                       area_ratio > (1 + self.area_change_threshold):
                        score *= 0.5

        # OC-SORT Observation-Centric Re-Update: feed the real observation back
        # into the tracker keyed on frame_idx.
        if self.use_kalman and obj_id in self.trackers:
            try:
                self.trackers[obj_id].update(actual_bbox, frame_idx)
            except Exception as e:
                logger.debug("Failed to update tracker for obj %s at frame %s: %s", obj_id, frame_idx, e)
        
        # Store data
        if obj_id not in self.scores:
            self.scores[obj_id] = {}
        self.scores[obj_id][frame_idx] = score
        self.bboxes[obj_id][frame_idx] = actual_bbox
        self.areas[obj_id][frame_idx] = mask_area
        self._prune_history(obj_id)
        
        # Adaptive threshold + hysteresis to reduce flicker in keep/drop decisions.
        threshold = self.get_effective_threshold(obj_id)
        low_cut = max(0.0, threshold - self.hysteresis_margin)
        high_cut = min(1.0, threshold + self.hysteresis_margin)
        in_low_state = self.low_state.get(obj_id, False)
        if in_low_state:
            should_use = score >= high_cut
        else:
            should_use = score >= low_cut
        self.low_state[obj_id] = not should_use

        # Track consecutive low scores for drift detection
        if not should_use:
            self.consecutive_low_count[obj_id] = self.consecutive_low_count.get(obj_id, 0) + 1
        else:
            self.consecutive_low_count[obj_id] = 0
        
        # Flag for correction if too many consecutive low scores
        if self.consecutive_low_count.get(obj_id, 0) >= self.consecutive_low_threshold:
            self.needs_correction[obj_id] = True
        else:
            self.needs_correction[obj_id] = False

        # Long-term appearance memory: store golden masks + detect occlusion recovery
        if self.appearance_memory is not None:
            self.appearance_memory.store_if_golden(
                obj_id, mask, actual_bbox, mask_area, score, frame_idx
            )
            self.appearance_memory.update_occlusion_state(obj_id, mask_area, frame_idx)

        return score, should_use

    def _prune_history(self, obj_id):
        """Keep only recent history to cap memory growth."""
        if self.max_history_frames is None or self.max_history_frames <= 0:
            return
        for store in (self.scores.get(obj_id), self.bboxes.get(obj_id), self.areas.get(obj_id)):
            if not store or len(store) <= self.max_history_frames:
                continue
            trim_count = len(store) - self.max_history_frames
            for frame_idx in sorted(store.keys())[:trim_count]:
                del store[frame_idx]
    
    def check_needs_correction(self, obj_id):
        """Check if object needs prompt correction due to drift."""
        return self.needs_correction.get(obj_id, False)
    
    def reset_correction_flag(self, obj_id):
        """Reset correction flag after applying correction."""
        self.needs_correction[obj_id] = False
        self.consecutive_low_count[obj_id] = 0
        self.low_state[obj_id] = False
    
    def get_low_score_frames(self, obj_id, threshold=None):
        """Get list of frames with scores below threshold."""
        if threshold is None:
            threshold = self.motion_score_threshold
        if obj_id not in self.scores:
            return []
        return [f for f, s in self.scores[obj_id].items() if s < threshold]
    
    def get_recent_scores(self, obj_id, n_frames=10):
        """Get scores for last n frames."""
        if obj_id not in self.scores:
            return []
        frames = sorted(self.scores[obj_id].keys())[-n_frames:]
        return [(f, self.scores[obj_id][f]) for f in frames]
    
    def get_best_memory_frames(self, obj_id, n_frames=6):
        """
        Get the N highest-scoring frames for memory bank prioritization.
        Returns list of (frame_idx, score) tuples.
        """
        if obj_id not in self.scores:
            return []
        sorted_frames = sorted(self.scores[obj_id].items(), key=lambda x: x[1], reverse=True)
        return sorted_frames[:n_frames]

