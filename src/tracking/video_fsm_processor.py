import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

try:
    from .volleyball_fsm import VolleyballFSM, GameState, RallyPhase, ActionType, CourtBounds
    from .volleyball_visualizer import VolleyballVisualizer, PlayerKeypoints
    from ..detection.simple_pose_estimator import SimplePoseEstimatorFactory, Pose, KeyPoint
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from volleystat.src.tracking.volleyball_fsm import VolleyballFSM, GameState, RallyPhase, ActionType, CourtBounds
    from volleystat.src.tracking.volleyball_visualizer import VolleyballVisualizer, PlayerKeypoints
    from volleystat.src.detection.simple_pose_estimator import SimplePoseEstimatorFactory, Pose, KeyPoint


@dataclass
class BallDetection:
    """Ball detection result"""
    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    timestamp: float


@dataclass
class ProcessingStats:
    """Processing statistics"""
    total_frames: int = 0
    processed_frames: int = 0
    ball_detections: int = 0
    player_detections: int = 0
    avg_processing_time: float = 0.0
    detection_rate: float = 0.0


class VideoFSMProcessor:
    """Video processor with FSM integration"""
    
    def __init__(self,
                 ball_detector_type: str = 'yolo',
                 pose_estimator_type: str = 'yolo11',
                 court_width: float = 18.0,
                 court_height: float = 9.0,
                 net_height: float = 2.43,
                 confidence_threshold: float = 0.3):
        """
        Initialize video FSM processor
        
        Args:
            ball_detector_type: Type of ball detector ('yolo', 'siamese', 'color')
            pose_estimator_type: Type of pose estimator ('yolo11', 'mediapipe')
            court_width: Width of volleyball court in meters
            court_height: Height of volleyball court in meters
            net_height: Height of net in meters
            confidence_threshold: Minimum confidence for detections
        """
        self.ball_detector_type = ball_detector_type
        self.pose_estimator_type = pose_estimator_type
        self.confidence_threshold = confidence_threshold
        
        # Initialize court bounds
        self.court_bounds = CourtBounds(
            x_min=0, x_max=court_width,
            y_min=0, y_max=court_height,
            net_x=court_width / 2,
            net_height=net_height
        )
        
        # Initialize FSM
        self.fsm = VolleyballFSM(self.court_bounds)
        
        # Initialize visualizer
        self.visualizer = VolleyballVisualizer(self.court_bounds)
        
        # Initialize detectors
        self._init_detectors()
        
        # Processing state
        self.current_frame = 0
        self.last_ball_position: Optional[Tuple[float, float]] = None
        self.ball_trajectory: List[BallDetection] = []
        self.stats = ProcessingStats()
        
        # Video properties
        self.video_fps: float = 30.0
        self.video_width: int = 1920
        self.video_height: int = 1080
        
    def _init_detectors(self) -> None:
        """Initialize detection models"""
        try:
            # Initialize pose estimator
            self.pose_estimator = SimplePoseEstimatorFactory.create_estimator(
                estimator_type=self.pose_estimator_type
            )
            print(f"✅ Initialized {self.pose_estimator_type} pose estimator")
            
            # Initialize ball detector based on type
            if self.ball_detector_type == 'yolo':
                self._init_yolo_ball_detector()
            elif self.ball_detector_type == 'siamese':
                self._init_siamese_ball_detector()
            elif self.ball_detector_type == 'color':
                self._init_color_ball_detector()
            else:
                raise ValueError(f"Unknown ball detector type: {self.ball_detector_type}")
                
        except Exception as e:
            print(f"❌ Error initializing detectors: {e}")
            # Fallback to simple color detection
            self._init_color_ball_detector()
            
    def _init_yolo_ball_detector(self) -> None:
        """Initialize YOLO ball detector"""
        try:
            from ultralytics import YOLO
            model_path = Path("volleystat/models/best/best.pt")
            if not model_path.exists():
                model_path = Path("yolov8n.pt")  # Fallback to general YOLO
                
            self.ball_detector = YOLO(str(model_path))
            print(f"✅ Initialized YOLO ball detector: {model_path}")
        except Exception as e:
            print(f"❌ Failed to initialize YOLO detector: {e}")
            self._init_color_ball_detector()
            
    def _init_siamese_ball_detector(self) -> None:
        """Initialize Siamese UNet ball detector"""
        try:
            # This would require importing Siamese UNet model
            # For now, fallback to color detection
            print("⚠️ Siamese UNet detector not implemented, using color detection")
            self._init_color_ball_detector()
        except Exception as e:
            print(f"❌ Failed to initialize Siamese detector: {e}")
            self._init_color_ball_detector()
            
    def _init_color_ball_detector(self) -> None:
        """Initialize simple color-based ball detector"""
        self.ball_detector = None  # Will use built-in color detection
        print("✅ Initialized color-based ball detector")
        
    def detect_ball(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Detect ball in frame"""
        timestamp = time.time()
        
        if self.ball_detector_type == 'yolo' and hasattr(self, 'ball_detector'):
            return self._detect_ball_yolo(frame, timestamp)
        else:
            return self._detect_ball_color(frame, timestamp)
            
    def _detect_ball_yolo(self, frame: np.ndarray, timestamp: float) -> Optional[BallDetection]:
        """Detect ball using YOLO"""
        try:
            results = self.ball_detector(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Check if it's a sports ball (class 32 in COCO) or volleyball
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if (class_id == 32 or class_id == 0) and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            return BallDetection(
                                center=(center_x, center_y),
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                timestamp=timestamp
                            )
            return None
            
        except Exception as e:
            print(f"❌ YOLO ball detection error: {e}")
            return self._detect_ball_color(frame, timestamp)
            
    def _detect_ball_color(self, frame: np.ndarray, timestamp: float) -> Optional[BallDetection]:
        """Detect ball using color-based approach"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define volleyball color ranges (white/yellow)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # Create masks
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask = cv2.bitwise_or(mask_white, mask_yellow)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_detection = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100 and area < 5000:  # Size constraints
                    # Get bounding circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    if 5 <= radius <= 50:  # Radius constraints
                        # Calculate circularity score
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            score = circularity * min(1.0, area / 1000.0)  # Prefer larger, more circular objects
                            
                            if score > best_score:
                                best_score = score
                                x1, y1, x2, y2 = cv2.boundingRect(contour)
                                best_detection = BallDetection(
                                    center=(x, y),
                                    bbox=(x1, y1, x1 + x2, y1 + y2),
                                    confidence=min(0.9, score),
                                    timestamp=timestamp
                                )
            
            return best_detection
            
        except Exception as e:
            print(f"❌ Color ball detection error: {e}")
            return None
            
    def detect_players(self, frame: np.ndarray) -> List[PlayerKeypoints]:
        """Detect players and extract keypoints"""
        try:
            poses = self.pose_estimator.detect_poses(frame)
            players = []
            
            for i, pose in enumerate(poses):
                if pose.confidence > 0.5:
                    # Extract key points
                    center = self._get_pose_center(pose)
                    if center is None:
                        continue
                        
                    # Determine team based on court position
                    team = 'team_a' if center[1] < self.video_height / 2 else 'team_b'
                    
                    # Extract specific keypoints
                    head = self._get_keypoint_position(pose, 0)  # Nose
                    shoulders = self._get_shoulder_keypoints(pose)
                    arms = self._get_arm_keypoints(pose)
                    hands = self._get_hand_keypoints(pose)
                    
                    player = PlayerKeypoints(
                        player_id=i + 1,
                        team=team,
                        center=center,
                        head=head,
                        shoulders=shoulders,
                        arms=arms,
                        hands=hands,
                        confidence=pose.confidence
                    )
                    players.append(player)
                    
            return players
            
        except Exception as e:
            print(f"❌ Player detection error: {e}")
            return []
            
    def _get_pose_center(self, pose: Pose) -> Optional[Tuple[float, float]]:
        """Get center point of pose"""
        valid_points = [(kp.x, kp.y) for kp in pose.keypoints if kp.confidence > 0.5]
        if valid_points:
            x = sum(p[0] for p in valid_points) / len(valid_points)
            y = sum(p[1] for p in valid_points) / len(valid_points)
            return (x, y)
        return None
        
    def _get_keypoint_position(self, pose: Pose, index: int) -> Optional[Tuple[float, float]]:
        """Get specific keypoint position"""
        if index < len(pose.keypoints) and pose.keypoints[index].confidence > 0.5:
            kp = pose.keypoints[index]
            return (kp.x, kp.y)
        return None
        
    def _get_shoulder_keypoints(self, pose: Pose) -> List[Tuple[float, float]]:
        """Get shoulder keypoints"""
        shoulders = []
        # Left shoulder (index 5 for COCO, 11 for MediaPipe)
        left_idx = 5 if len(pose.keypoints) == 17 else 11
        right_idx = 6 if len(pose.keypoints) == 17 else 12
        
        for idx in [left_idx, right_idx]:
            pos = self._get_keypoint_position(pose, idx)
            if pos:
                shoulders.append(pos)
        return shoulders
        
    def _get_arm_keypoints(self, pose: Pose) -> List[Tuple[float, float]]:
        """Get arm keypoints (elbows)"""
        arms = []
        # Elbow indices
        left_elbow = 7 if len(pose.keypoints) == 17 else 13
        right_elbow = 8 if len(pose.keypoints) == 17 else 14
        
        for idx in [left_elbow, right_elbow]:
            pos = self._get_keypoint_position(pose, idx)
            if pos:
                arms.append(pos)
        return arms
        
    def _get_hand_keypoints(self, pose: Pose) -> List[Tuple[float, float]]:
        """Get hand keypoints (wrists)"""
        hands = []
        # Wrist indices
        left_wrist = 9 if len(pose.keypoints) == 17 else 15
        right_wrist = 10 if len(pose.keypoints) == 17 else 16
        
        for idx in [left_wrist, right_wrist]:
            pos = self._get_keypoint_position(pose, idx)
            if pos:
                hands.append(pos)
        return hands
        
    def screen_to_world(self, screen_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        x_screen, y_screen = screen_pos
        
        # Simple linear mapping (can be improved with camera calibration)
        x_world = (x_screen / self.video_width) * self.court_bounds.x_max
        y_world = (y_screen / self.video_height) * self.court_bounds.y_max
        
        return (x_world, y_world)
        
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x_world, y_world = world_pos
        
        x_screen = int((x_world / self.court_bounds.x_max) * self.video_width)
        y_screen = int((y_world / self.court_bounds.y_max) * self.video_height)
        
        return (x_screen, y_screen)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with FSM"""
        start_time = time.time()
        
        # Detect ball
        ball_detection = self.detect_ball(frame)
        if ball_detection:
            self.stats.ball_detections += 1
            
            # Convert to world coordinates
            ball_world = self.screen_to_world(ball_detection.center)
            
            # Update FSM
            self.fsm.update(ball_world, self.current_frame / self.video_fps)
            
            # Store ball trajectory
            self.ball_trajectory.append(ball_detection)
            if len(self.ball_trajectory) > 100:  # Keep last 100 detections
                self.ball_trajectory.pop(0)
                
        # Detect players
        players = self.detect_players(frame)
        self.stats.player_detections += len(players)
        
        # Convert players to world coordinates
        world_players = []
        for player in players:
            world_center = self.screen_to_world(player.center)
            world_player = PlayerKeypoints(
                player_id=player.player_id,
                team=player.team,
                center=world_center,
                head=self.screen_to_world(player.head) if player.head else None,
                shoulders=[self.screen_to_world(s) for s in player.shoulders] if player.shoulders else None,
                arms=[self.screen_to_world(a) for a in player.arms] if player.arms else None,
                hands=[self.screen_to_world(h) for h in player.hands] if player.hands else None,
                confidence=player.confidence
            )
            world_players.append(world_player)
            
        # Create visualization
        ball_world_pos = self.screen_to_world(ball_detection.center) if ball_detection else None
        vis_frame = self.visualizer.create_frame(
            ball_position=ball_world_pos,
            players=world_players,
            fsm=self.fsm,
            frame_width=self.video_width,
            frame_height=self.video_height
        )
        
        # Overlay on original frame
        result_frame = cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0)
        
        # Draw detections on original frame
        if ball_detection:
            # Draw ball
            center = (int(ball_detection.center[0]), int(ball_detection.center[1]))
            cv2.circle(result_frame, center, 8, (0, 255, 0), -1)
            cv2.circle(result_frame, center, 12, (0, 255, 0), 2)
            
            # Draw ball trajectory
            if len(self.ball_trajectory) > 1:
                points = [self.ball_trajectory[i].center for i in range(len(self.ball_trajectory))]
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    alpha = i / len(points)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(result_frame, pt1, pt2, (255, 0, 0), thickness)
                    
        # Draw player keypoints
        for player in players:
            color = (255, 255, 0) if player.team == 'team_a' else (0, 255, 255)
            
            # Draw center
            center = (int(player.center[0]), int(player.center[1]))
            cv2.circle(result_frame, center, 6, color, -1)
            
            # Draw keypoints
            if player.head:
                head_pos = (int(player.head[0]), int(player.head[1]))
                cv2.circle(result_frame, head_pos, 4, color, -1)
                
            if player.hands:
                for hand in player.hands:
                    hand_pos = (int(hand[0]), int(hand[1]))
                    cv2.circle(result_frame, hand_pos, 3, (0, 255, 255), -1)
                    
        # Draw stats
        self._draw_stats(result_frame, time.time() - start_time)
        
        # Update frame counter
        self.current_frame += 1
        self.stats.processed_frames += 1
        
        return result_frame
        
    def _draw_stats(self, frame: np.ndarray, processing_time: float) -> None:
        """Draw processing statistics on frame"""
        # Update processing time
        self.stats.avg_processing_time = (
            (self.stats.avg_processing_time * (self.stats.processed_frames - 1) + processing_time) 
            / self.stats.processed_frames
        )
        
        # Calculate detection rate
        if self.stats.processed_frames > 0:
            self.stats.detection_rate = self.stats.ball_detections / self.stats.processed_frames * 100
            
        # Draw stats panel
        panel_height = 120
        panel_width = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text info
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        
        stats_text = [
            f"Frame: {self.current_frame}",
            f"FSM State: {self.fsm.state.name}",
            f"Rally Phase: {self.fsm.rally_phase.name}",
            f"Ball Detection: {self.stats.detection_rate:.1f}%",
            f"Processing: {self.stats.avg_processing_time*1000:.1f}ms"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (15, y_offset + i * 18), font, font_scale, color, 1)
            
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     start_sec: float = 0,
                     duration_sec: Optional[float] = None,
                     show_preview: bool = True) -> ProcessingStats:
        """
        Process entire video with FSM
        
        Args:
            input_path: Input video path
            output_path: Output video path
            start_sec: Start time in seconds
            duration_sec: Duration to process (None for entire video)
            show_preview: Whether to show live preview
            
        Returns:
            Processing statistics
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
            
        # Get video properties
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {self.video_width}x{self.video_height} @ {self.video_fps:.1f}fps, {total_frames} frames")
        
        # Calculate frame range
        start_frame = int(start_sec * self.video_fps)
        if duration_sec:
            end_frame = start_frame + int(duration_sec * self.video_fps)
        else:
            end_frame = total_frames
            
        end_frame = min(end_frame, total_frames)
        frames_to_process = end_frame - start_frame
        
        print(f"🎬 Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")
        
        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Setup output video
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.video_fps, 
                             (self.video_width, self.video_height))
        
        # Reset stats
        self.stats = ProcessingStats()
        self.stats.total_frames = frames_to_process
        
        start_time = time.time()
        
        try:
            for frame_idx in range(frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Volleyball FSM Processing', 
                              cv2.resize(processed_frame, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                # Progress update
                if frame_idx % 30 == 0:  # Every second
                    progress = (frame_idx + 1) / frames_to_process * 100
                    elapsed = time.time() - start_time
                    fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                    
                    print(f"⚡ Progress: {progress:.1f}% | "
                          f"FPS: {fps:.1f} | "
                          f"Ball Detection: {self.stats.detection_rate:.1f}% | "
                          f"Players: {self.stats.player_detections}/{frame_idx+1}")
                          
        finally:
            # Cleanup
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
                
        # Final stats
        total_time = time.time() - start_time
        print(f"\n✅ Processing completed!")
        print(f"📊 Total time: {total_time:.1f}s")
        print(f"📊 Average FPS: {self.stats.processed_frames/total_time:.1f}")
        print(f"📊 Ball detections: {self.stats.ball_detections}/{self.stats.processed_frames} ({self.stats.detection_rate:.1f}%)")
        print(f"📊 Player detections: {self.stats.player_detections}")
        print(f"📊 Output saved: {output_path}")
        
        return self.stats


def main():
    """Example usage"""
    # Create processor
    processor = VideoFSMProcessor(
        ball_detector_type='color',  # Use color detection for compatibility
        pose_estimator_type='yolo11',
        confidence_threshold=0.3
    )
    
    # Process video
    try:
        input_video = "data/test_video.mp4"  # Replace with actual video path
        output_video = "data/fsm_output.mp4"
        
        stats = processor.process_video(
            input_path=input_video,
            output_path=output_video,
            start_sec=30,
            duration_sec=60,
            show_preview=True
        )
        
        print(f"✅ Processing completed with {stats.ball_detections} ball detections")
        
    except FileNotFoundError:
        print("❌ Test video not found. Please provide a valid video path.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 