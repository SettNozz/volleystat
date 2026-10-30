import os
import argparse
import cv2
import glob
import math
import torch
import random
import tabulate
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision.io import read_video
from torchvision.transforms import v2
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader

# custom modules
from util.io import store_json, load_json
from train_e2e_spatial import E2EModel
from util.score import (
    filter_events_by_score,
    non_max_suppression_events,
)


def get_args(**overrides):
    args = dict(
        dilate_len=0,
        mixup=False,
        modality="rgb",
        feature_arch="rny008_gsm",
        temporal_arch="gru",
        clip_len=32,
        batch_size=2,
        predict_location=True,
        num_workers=4,
        eval_only=False,
        checkpoint_path="exp/best/checkpoint_138.pt",
        criterion="map",
        fg_upsample=None,
        gpu_parallel=False,
        debug_only=False,
        # classes="serve receive set spike dig block".split(),
        # classes="serve receive set spike block score".split(),
        classes="block dig net pass receive score serve set spike".split(),
        num_videos=None,
    )
    args.update(overrides)
    args = edict(args)
    # print tabulated args
    print(
        tabulate.tabulate(
            [
                (key, args[key])
                for key in sorted(args.keys())
                if not key.startswith("_")
            ],
            headers=["Argument", "Value"],
            tablefmt="fancy_outline",
        )
    )
    return args


def get_model(args):
    e2em = E2EModel(
        len(args.classes) + 1,
        args.feature_arch,
        args.temporal_arch,
        clip_len=args.clip_len,
        modality=args.modality,
        multi_gpu=args.gpu_parallel,
        predict_location=args.predict_location,
    )

    e2em.load(torch.load(os.path.abspath(args.checkpoint_path), weights_only=True))
    return e2em


class VideoFrameSlidingDataset(torch.utils.data.Dataset):
    """MP4 mode only works with short videos (few seconds long) due to memory constraints."""

    def __init__(
        self,
        input_data,  # Could be either a list of numpy images or a video path
        mode="list",  # "list" for numpy arrays, "mp4" for video file
        transform=None,
        window_size=16,
        stride=1,
        channel_first=False,  # (C, T, H, W) if True, (T, C, H, W) if False
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.channel_first = channel_first
        self.transform = transform

        if self.mode == "list":
            self.image_list = input_data
            self.num_frames = len(self.image_list)
            print(f"Total frames in list: {self.num_frames}")
        elif self.mode == "mp4":
            self.video_path = input_data
            # Use cv2 for lazy loading to save memory
            cap = cv2.VideoCapture(self.video_path)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.cap = None  # Initialize as None, created in worker
            print(f"Video path: {self.video_path}, Total frames: {self.num_frames}")

        else:
            raise ValueError("Mode must be either 'list' or 'mp4'")

    @property
    def videos(self):
        return [(self.video_path, self.num_frames)]

    def _load_sequence(self, start_idx):
        """Load a sequence of frames starting from `start_idx`."""
        frames = []
        org_sizes = []

        if self.mode == "mp4":
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        for i in range(self.window_size):
            actual_idx = start_idx + i
            if actual_idx < self.num_frames:
                if self.mode == "list":
                    # Load frame from the list of numpy arrays
                    frame = torch.tensor(
                        self.image_list[actual_idx]
                    )  # Convert to tensor
                    org_size = (frame.shape[2], frame.shape[1])  # (Width, Height)
                elif self.mode == "mp4":
                    # Load frame from video using cv2
                    ret, frame = self.cap.read()
                    if not ret:
                        # Should not happen if num_frames is correct
                        raise RuntimeError(
                            f"Failed to read frame {actual_idx} from {self.video_path}"
                        )

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame)  # (H, W, C)
                    org_size = (frame.shape[1], frame.shape[0])  # (Width, Height)
                    frame = frame.permute(2, 0, 1)  # (C, H, W)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
            else:
                # Zero-padding if not enough frames
                if not org_sizes:
                    raise RuntimeError(
                        "Padding requested but no previous frames to determine size."
                    )
                last_org_size = org_sizes[-1]
                frame = torch.zeros(
                    (3, last_org_size[1], last_org_size[0]),
                    dtype=torch.float,
                )
                org_size = last_org_size

            frames.append(frame)
            org_sizes.append(org_size)

        return frames, org_sizes

    def __len__(self):
        # Calculate the number of sequences based on stride and window_size
        return max(1, math.ceil((self.num_frames - self.window_size) / self.stride + 1))

    def __getitem__(self, idx):
        # Calculate start index for this sequence
        start_idx = idx * self.stride

        # Load sequence of frames and their original sizes
        frames, org_sizes = self._load_sequence(start_idx)

        # Stack frames into a tensor of shape (T, C, H, W)
        frames = torch.stack(frames, dim=0)

        # Apply transform to the entire sequence (all frames in the sequence)
        if self.transform:
            frames = self.transform(frames)

        # Convert to (C, T, H, W) if channel_first is True
        if self.channel_first:
            frames = frames.permute(
                1, 0, 2, 3
            )  # Convert from (T, C, H, W) to (C, T, H, W)

        return {
            "game_id": self.video_path,
            "seg_id": idx,
            "frames": frames,
            "start": start_idx,
        }


def run_inference(model, dataloader, classes, pred_file, postprocess=False):
    pred_dict = {}

    for video, video_len in dataloader.dataset.videos:
        pred_dict[video] = (
            np.zeros(
                (video_len, len(classes) + 1), np.float32
            ),  # Stores scores for each class
            np.zeros(video_len, np.int32),  # Stores support (number of frames)
            np.zeros((video_len, 2), np.float32),  # Stores location predictions (x, y)
        )

    for clip in tqdm(dataloader):
        _, batch_pred_scores, batch_pred_loc = model.predict(clip["frames"])

        batch_pred_loc = batch_pred_loc.reshape(batch_pred_scores.shape[0], -1, 2)
        for i in range(clip["frames"].shape[0]):
            video = clip["game_id"][i]
            scores, support, locations = pred_dict[video]

            pred_scores = batch_pred_scores[i]  # Predicted scores for the current batch
            pred_loc = batch_pred_loc[i]  # Predicted locations for the current batch

            start = clip["start"][i].item()
            end = min(start + pred_scores.shape[0], len(support))

            pred_scores = pred_scores[: end - start, :]
            pred_loc = pred_loc[: end - start, :]

            scores[start:end, :] += pred_scores  # Accumulate scores
            support[
                start:end
            ] += 1  # Increment the support count because the frame is present
            locations[start:end, :] = pred_loc
            print(f"Processed {video} from {start} to {end}")

    pred_events = []
    pred_scores = {}
    for video, (scores, support, locations_pred) in sorted(pred_dict.items()):
        # breakpoint()
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        # locations_pred /= support[:, None]
        pred = np.argmax(scores, axis=1)

        pred_scores[video] = scores.tolist()

        events = []
        for i in range(pred.shape[0]):

            if pred[i] != 0:
                events.append(
                    {
                        "label": classes[pred[i] - 1],
                        "frame": i,
                        "score": scores[i, pred[i]].item(),
                        "xy": locations_pred[i].tolist(),
                    }
                )

        pred_events.append({"video": video, "events": events})

    if postprocess:

        pred_events = filter_events_by_score(pred_events, fg_threshold=0.15)

        pred_events = non_max_suppression_events(pred_events, 3)

    if pred_file:
        os.makedirs(os.path.dirname(pred_file), exist_ok=True)
        store_json(pred_file, pred_events)

    print(f"Saved predictions to {pred_file}")
    return pred_events

def render_video(
    video_file,
    events,
    freeze_frames=25,
    main_color=(242, 243, 244),
    out_width=1280,
    out_height=720,
):

    event_by_frame = {x["frame"]: x for x in events}
    if video_file.endswith(".mp4"):
        output_video_file = video_file.replace(".mp4", "_output.mp4")
    elif video_file.endswith(".MP4"):
        output_video_file = video_file.replace(".MP4", "_output.mp4")
    else:
        raise ValueError(f"Unsupported video format: {video_file}")

    # Use cv2 for video reading and writing to save memory
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (out_width, out_height))

    for idx in tqdm(range(total_frames), desc="Rendering video"):
        ret, frame = cap.read()
        if not ret:
            break

        if idx not in event_by_frame:
            # Resize if needed
            if out_width != width or out_height != height:
                frame = cv2.resize(frame, (out_width, out_height))
            out.write(frame)
        else:
            event = event_by_frame[idx]
            x, y = int(event["xy"][0] * width), int(event["xy"][1] * height)
            score = event["score"]
            label = event["label"]
            text = f"{label.upper()} {score:.0%}"

            for it in range(freeze_frames):
                dt = it / freeze_frames
                canvas = frame.copy()

                draw_filled_parallelogram(text, (x, y), canvas, dt)

                alpha = max(0.9, 0.65 + 0.25 * dt)
                canvas = cv2.addWeighted(frame, 1 - alpha, canvas, alpha, 0)

                radius = 10 + 40 * ((1 - dt) ** 8)
                thickness = 1 + dt * 2
                cv2.circle(
                    canvas,
                    (x, y),
                    int(radius),
                    main_color,
                    int(thickness),
                    lineType=cv2.LINE_AA,
                )
                
                # Resize if needed
                if out_width != width or out_height != height:
                    canvas = cv2.resize(canvas, (out_width, out_height))
                out.write(canvas)

    cap.release()
    out.release()

    return output_video_file


def draw_filled_parallelogram(text, center_coordinates, image, dt):
    # Initial parameters
    radius = 10
    color = (255, 255, 255)  # White color in BGR
    circle_thickness = 1  # Solid circle
    delta_shadow = np.array([3, 3])
    color_shadow = (57, 61, 71)  # Black color for the shadow
    color_text = (0, 0, 0)  # Black color for the text

    # Text and padding
    font_scale = 0.7
    font_thickness = 1
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    left_padding = 7
    right_padding = 20
    top_padding = 12
    bottom_padding = 7

    # Adjust the length of the parallelogram based on the text size and padding
    parallelogram_length = text_size[0] + left_padding + right_padding
    parallelogram_height = text_size[1] + top_padding + bottom_padding

    # Define angle for the 60-degree line
    angle = np.pi / 3  # 60 degrees in radians
    positive_angle = 2 * np.pi / 3  # 120 degrees in radians

    # Calculate the 60-degree line points
    start_point = center_coordinates
    end_point = (
        int(center_coordinates[0] + 100 * np.cos(angle)),
        int(center_coordinates[1] - 100 * np.sin(angle)),
    )

    # Calculate the points of the parallelogram, with the first point being below the 60-degree line
    gap = 5
    point1 = (end_point[0] + gap, end_point[1] + gap)  # Below the 60-degree line
    point2 = (
        int(point1[0] + parallelogram_length),
        point1[1],
    )  # Move horizontally (parallel to horizontal line)
    point3 = (
        int(point2[0] + parallelogram_height * np.cos(positive_angle)),
        int(point2[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees
    point4 = (
        int(point1[0] + parallelogram_height * np.cos(positive_angle)),
        int(point1[1] + parallelogram_height * np.sin(positive_angle)),
    )  # Adjusted to 120 degrees

    # Draw the white dot with radius 10
    cv2.circle(
        image,
        center_coordinates,
        radius,
        color,
        circle_thickness,
        lineType=cv2.LINE_AA,
    )

    # Draw the 60-degree line
    end_point_ = np.add(
        start_point, np.subtract(end_point, center_coordinates) * min(1.0, dt * 4)
    ).astype(np.int32)

    start_point_shadow = np.add(start_point, delta_shadow * 0.5).astype(np.int32)
    end_point_shadow = np.add(end_point_, delta_shadow * 0.5).astype(np.int32)

    cv2.line(
        image,
        start_point_shadow,
        end_point_shadow,
        color_shadow,  # Black color for the shadow
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.line(image, start_point, end_point_, color, thickness=2, lineType=cv2.LINE_AA)

    # Draw the horizontal line from the end of the first line

    if dt >= 1 / 4.0:
        horizontal_end_point = (
            int(end_point[0] + 60 * min(1.0, 4 * (dt - 0.25))),
            end_point[1],
        )
        horizontal_end_point_shadow = np.add(
            horizontal_end_point, delta_shadow * 0.5
        ).astype(np.int32)
        cv2.line(
            image,
            end_point_shadow,
            horizontal_end_point_shadow,
            color_shadow,  # Black color for the shadow
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            image,
            end_point,
            horizontal_end_point,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    # Fill the parallelogram by using the fillPoly function
    delta = np.array([max(0, 1 - dt * 4) * 15, 0])
    pts = np.array([point1, point2, point3, point4], np.int32)
    pts = (pts + delta).astype(np.int32)
    pts_shadow = (pts + delta_shadow).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts_shadow = pts_shadow.reshape((-1, 1, 2))

    # draw main parallelogram
    cv2.fillPoly(image, [pts_shadow], color_shadow, lineType=cv2.LINE_AA)
    cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_AA)

    # Calculate text position (centered inside the parallelogram with adjusted padding)
    text_x = pts[0][0][0] + left_padding
    text_y = pts[0][0][1] + top_padding + text_size[1] // 2

    # Add the text inside the parallelogram
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color_text,
        font_thickness,
        lineType=cv2.LINE_AA,
    )

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--checkpoint_path", type=str, default="exp/e2espatial_vnl2/checkpoint_best.pt", help="Path to the model checkpoint")
    parser.add_argument("--save_dir", type=str, default="exp/demo1", help="Directory to save predictions and output video")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--clip_len", type=int, default=64, help="Clip length (window size)")
    parser.add_argument("--render", action="store_true", help="Render the output video with annotations")
    
    args = parser.parse_args()

    # Set the arguments for inference
    inference_args = get_args(
        video_mp4=args.video_path,
        resize=(224, 224),
        clip_len=args.clip_len,
        checkpoint_path=args.checkpoint_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    inference_args.pred_file = os.path.join(inference_args.save_dir, "predictions.json")

    transform = v2.Compose(
        [
            v2.Resize(inference_args.resize, antialias=False),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = VideoFrameSlidingDataset(
        inference_args.video_mp4,
        mode="mp4",
        window_size=inference_args.clip_len,
        stride=inference_args.clip_len,
        channel_first=False,
        transform=transform,
    )

    # Create the DataLoader for batch processing
    dataloader = DataLoader(
        dataset, batch_size=inference_args.batch_size, num_workers=inference_args.num_workers
    )

    # Load the pre-trained model
    model = get_model(inference_args)

    # Run the inference and save the predictions
    pred_events = run_inference(
        model,
        dataloader,
        inference_args.classes,
        inference_args.pred_file,
        postprocess=True,
    )

    if args.render:
        # pred_events is a list of dicts, we take the first one for the single video
        events = pred_events[0]["events"]
        render_video(
            args.video_path,
            events,
        )

