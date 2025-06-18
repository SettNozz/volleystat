import os
import cv2
import numpy as np
import random
import shutil
from src.utils.person_detection import PersonDetector, create_person_ball_dataset


def denormalize_points(points, width, height):
    """Convert normalized coordinates to absolute pixel coordinates."""
    return [(int(x * width), int(y * height)) for x, y in points]


def create_masks_from_yolo_labels(images_dir, labels_dir, masks_dir):
    """Create binary masks from YOLO OBB label files."""
    os.makedirs(masks_dir, exist_ok=True)
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        # Corresponding image
        image_file = label_file.replace(".txt", ".jpg")
        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[!] Image not found or unreadable: {image_path}")
            continue
        h, w = img.shape[:2]

        # Empty mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Read YOLO OBB boxes
        with open(os.path.join(labels_dir, label_file), "r") as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]  # x1,y1,...,x4,y4
                points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
                polygon = np.array([denormalize_points(points, w, h)], dtype=np.int32)
                cv2.fillPoly(mask, polygon, color=255)  # white mask

        # Save mask
        mask_path = os.path.join(masks_dir, label_file.replace(".txt", "_mask.png"))
        cv2.imwrite(mask_path, mask)


def create_one_shot_dataset(base_path, output_base, crop_size=64):
    """Create one-shot segmentation dataset structure from YOLO dataset."""
    images_path = os.path.join(base_path, "images")
    labels_path = os.path.join(base_path, "labels")

    # Create output directories
    support_dir = os.path.join(output_base, "support")
    query_dir = os.path.join(output_base, "query")
    masks_dir = os.path.join(output_base, "masks")

    os.makedirs(support_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Process all annotations
    label_files = [f for f in os.listdir(labels_path) if f.endswith(".txt")]
    support_counter = 1

    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        image_file = os.path.join(images_path, base_name + ".jpg")

        if not os.path.exists(image_file):
            print(f"[!] Skipped: {image_file} not found.")
            continue

        img = cv2.imread(image_file)
        if img is None:
            print(f"[!] Cannot read: {image_file}")
            continue
        h, w = img.shape[:2]

        # Save original query image
        query_path = os.path.join(query_dir, base_name + ".jpg")
        cv2.imwrite(query_path, img)

        # Empty mask for query
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(os.path.join(labels_path, label_file), "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 9:
                continue

            cls_id = int(parts[0])
            if cls_id != 1:  # only ball
                continue

            # YOLO OBB → absolute coordinates
            coords = parts[1:]
            polygon = np.array([(coords[i] * w, coords[i+1] * h) for i in range(0, 8, 2)], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], color=255)

            # Support crop
            x_min = max(int(np.min(polygon[:, 0])), 0)
            y_min = max(int(np.min(polygon[:, 1])), 0)
            x_max = min(int(np.max(polygon[:, 0])), w - 1)
            y_max = min(int(np.max(polygon[:, 1])), h - 1)

            ball_crop = img[y_min:y_max, x_min:x_max]
            if ball_crop.size == 0:
                continue

            ball_crop_resized = cv2.resize(ball_crop, (crop_size, crop_size))
            support_path = os.path.join(support_dir, f"{base_name}_ball.jpg")
            cv2.imwrite(support_path, ball_crop_resized)
            support_counter += 1

        # Save mask
        mask_path = os.path.join(masks_dir, base_name + "_mask.png")
        cv2.imwrite(mask_path, mask)

    print("✅ Successfully created one-shot structure:")
    print(f"📁 support/: {support_counter - 1} examples")
    print(f"📁 query/: {len(os.listdir(query_dir))} images")
    print(f"📁 masks/: {len(os.listdir(masks_dir))} masks")
    print(f"query_dir: {query_dir}")


def create_person_ball_one_shot_dataset(video_path, output_base, start_sec=30, duration_sec=30, 
                                       frame_interval=1, crop_size=64, model_path='yolo11n.pt'):
    """
    Create one-shot dataset with both person and ball detections from video.
    
    Args:
        video_path (str): Path to input video
        output_base (str): Output directory
        start_sec (int): Start time in seconds
        duration_sec (int): Duration to process
        frame_interval (int): Extract every Nth frame
        crop_size (int): Size of support crops
        model_path (str): Path to YOLO model
    """
    # Create output directories
    support_dir = os.path.join(output_base, "support")
    query_dir = os.path.join(output_base, "query")
    masks_dir = os.path.join(output_base, "masks")
    person_support_dir = os.path.join(output_base, "person_support")

    os.makedirs(support_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(person_support_dir, exist_ok=True)

    # Initialize detector
    detector = PersonDetector(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_sec)
    end_frame = min(int(fps * (start_sec + duration_sec)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    saved_count = 0
    
    print(f"🔁 Creating person-ball dataset from video...")
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Detect persons and balls
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.model(img_rgb, verbose=False)[0]
            
            person_boxes = []
            ball_boxes = []
            
            for box in results.boxes:
                cls = int(box.cls.cpu().item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                conf = float(box.conf.cpu())
                
                if cls == 0:  # Person
                    person_boxes.append((x1, y1, x2, y2, conf))
                elif cls == 32:  # Ball (COCO dataset)
                    ball_boxes.append((x1, y1, x2, y2, conf))
            
            # Save frame and annotations if detections found
            if person_boxes or ball_boxes:
                frame_filename = f"frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(query_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Create mask
                h, w = frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Add ball detections to mask
                for x1, y1, x2, y2, conf in ball_boxes:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
                # Save mask
                mask_path = os.path.join(masks_dir, f"frame_{saved_count:05d}_mask.png")
                cv2.imwrite(mask_path, mask)
                
                # Save ball support crops
                for i, (x1, y1, x2, y2, conf) in enumerate(ball_boxes):
                    ball_crop = frame[y1:y2, x1:x2]
                    if ball_crop.size > 0:
                        ball_crop_resized = cv2.resize(ball_crop, (crop_size, crop_size))
                        support_path = os.path.join(support_dir, f"frame_{saved_count:05d}_ball_{i}.jpg")
                        cv2.imwrite(support_path, ball_crop_resized)
                
                # Save person support crops
                for i, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
                    # Expand to square crop
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    half_size = crop_size // 2
                    
                    crop_x1 = max(0, center_x - half_size)
                    crop_y1 = max(0, center_y - half_size)
                    crop_x2 = min(w, center_x + half_size)
                    crop_y2 = min(h, center_y + half_size)
                    
                    person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if person_crop.size > 0:
                        person_crop_resized = cv2.resize(person_crop, (crop_size, crop_size))
                        person_support_path = os.path.join(person_support_dir, f"frame_{saved_count:05d}_person_{i}.jpg")
                        cv2.imwrite(person_support_path, person_crop_resized)
                
                saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Person-ball dataset created with {saved_count} frames")
    print(f"📁 Saved to: {output_base}")


def split_dataset_consistently(base_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test sets consistently."""
    query_dir = os.path.join(base_dir, "query")
    support_dir = os.path.join(base_dir, "support")
    mask_dir = os.path.join(base_dir, "masks")

    query_files = [f for f in os.listdir(query_dir) if f.endswith(".jpg")]
    valid_files = []

    for qf in query_files:
        mask_name = qf.replace(".jpg", "_mask.png")
        support_name = qf.replace(".jpg", "_ball.jpg")

        mask_path = os.path.join(mask_dir, mask_name)
        support_path = os.path.join(support_dir, support_name)

        if os.path.exists(mask_path) and os.path.exists(support_path):
            valid_files.append(qf)
        else:
            print(f"[!] Skipped {qf} — missing mask or support")

    random.shuffle(valid_files)
    total = len(valid_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        "train": valid_files[:train_end],
        "val": valid_files[train_end:val_end],
        "test": valid_files[val_end:]
    }

    for split, files in splits.items():
        for sub in ["query", "support", "masks"]:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

        for qf in files:
            mask_name = qf.replace(".jpg", "_mask.png")
            support_name = qf.replace(".jpg", "_ball.jpg")

            shutil.copy(os.path.join(query_dir, qf), os.path.join(output_dir, split, "query", qf))
            shutil.copy(os.path.join(support_dir, support_name), os.path.join(output_dir, split, "support", support_name))
            shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(output_dir, split, "masks", mask_name))

    print("\n✅ Dataset split:")
    for split, files in splits.items():
        print(f"{split}: {len(files)} images")


def check_mask_values(mask_dir, num_samples=10):
    """Check mask values to ensure they are binary."""
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    print(f"Checking first {num_samples} masks in {mask_dir} ...")

    for f in mask_files[:num_samples]:
        mask_path = os.path.join(mask_dir, f)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[!] Failed to load {f}")
            continue

        unique_vals = np.unique(mask)
        print(f"{f}: unique values {unique_vals}")

        # Check for binary (0 and 1, or 0 and 255)
        if not (set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({0, 255})):
            print(f"  [!] Mask {f} is NOT binary!")


def process_video_with_person_detection(video_path, output_path, start_sec=30, duration_sec=30, model_path='yolo11n.pt'):
    """
    Process video segment with person detection and save annotated video.
    
    Args:
        video_path (str): Input video path
        output_path (str): Output video path
        start_sec (int): Start time in seconds
        duration_sec (int): Duration to process
        model_path (str): Path to YOLO model
    """
    detector = PersonDetector(model_path)
    detector.process_video_segment(video_path, output_path, start_sec, duration_sec) 