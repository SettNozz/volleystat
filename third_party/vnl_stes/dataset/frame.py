#!/usr/bin/env python3

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors

from util.io import load_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameReader:

    IMG_NAME = "{:06d}.jpg"

    def __init__(self, frame_dir, modality):
        self._frame_dir = frame_dir
        self._is_flow = modality == "flow"

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path).float() / 255
        if self._is_flow:
            img = img[1:, :, :]  # GB channels contain data
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1, randomize=False):
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        for frame_num in range(start, end, stride):
            if randomize and stride > 1:
                frame_num += random.randint(0, stride - 1)

            if frame_num < 0:
                n_pad_start += 1
                continue

            frame_path = os.path.join(
                self._frame_dir, video_name, FrameReader.IMG_NAME.format(
                    frame_num)
            )
            try:
                img = self.read_frame(frame_path)
                ret.append(img)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0)
            )
        return ret


# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5


def check_for_nan(img, transform_name):
    if torch.isnan(img[0] if isinstance(img, tuple) else img).any():
        print(f"NaN found after {transform_name}")
    return img


def _get_img_transforms(crop_dim=224, is_eval=False):
    p = 0.0 if is_eval else 0.25

    geometric_transforms = []
    if not is_eval:
        geometric_transforms.append(
            transforms.RandomChoice(
                [
                    transforms.RandomCrop((crop_dim, crop_dim)),
                    transforms.RandomResizedCrop((crop_dim, crop_dim)),
                    transforms.Resize((crop_dim, crop_dim)),  # No-op
                ]
            )
        )
        geometric_transforms.append(
            transforms.RandomChoice(
                [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomZoomOut(p=1.0),
                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                    transforms.RandomRotation(degrees=(0, 45)),
                    transforms.RandomAffine(
                        degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                    ),
                    transforms.Resize((crop_dim, crop_dim)),  # No-op
                ]
            ),
        )

    geometric_transforms.append(transforms.Resize((crop_dim, crop_dim)))

    img_transforms = [
        # ColorJittering
        transforms.RandomApply(nn.ModuleList(
            [transforms.ColorJitter(hue=0.2)]), p=p),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(saturation=(0.7, 1.2))]), p=p
        ),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(brightness=(0.7, 1.2))]), p=p
        ),
        transforms.RandomApply(
            nn.ModuleList([transforms.ColorJitter(contrast=(0.7, 1.2))]), p=p
        ),
        transforms.RandomApply(nn.ModuleList(
            [transforms.GaussianBlur(5)]), p=p),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    # return transforms.Compose(geometric_transforms + img_transforms)

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: check_for_nan(img, "Initial")),
            *geometric_transforms,
            transforms.Lambda(lambda img: check_for_nan(
                img, "Geometric Transforms")),
            *img_transforms,
            transforms.Lambda(
                lambda img: check_for_nan(
                    img, "Color Jittering and Normalization")
            ),
        ]
    )


def _print_info_helper(src_file, labels):
    num_frames = sum([x["num_frames"] for x in labels])
    num_events = sum([len(x["events"]) for x in labels])
    print(
        "{} : {} videos, {} frames, {:0.5f}% non-bg".format(
            src_file, len(labels), num_frames, num_events / num_frames * 100
        )
    )


IGNORED_NOT_SHOWN_FLAG = False


class ActionSpotDataset(Dataset):

    def __init__(
        self,
        classes,  # dict of class names to idx
        label_file,  # path to label json
        frame_dir,  # path to frames
        modality,  # [rgb, bw, flow]
        clip_len,
        dataset_len,  # Number of clips
        is_eval=True,  # Disable random augmentation
        crop_dim=None,
        stride=1,  # Downsample frame rate
        same_transform=True,  # Apply the same random augmentation to
        # each frame in a clip
        dilate_len=0,  # Dilate ground truth labels
        mixup=False,
        pad_len=DEFAULT_PAD_LEN,  # Number of frames to pad the start
        # and end of videos
        fg_upsample=-1,  # Sample foreground explicitly
        dataset="finediving",
    ):
        self._dataset = dataset
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x["video"]: i for i, x in enumerate(self._labels)}

        # Sample videos weighted by their length
        num_frames = [v["num_frames"] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval

        # Label modifications
        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        # Sample based on foreground labels
        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                for event in x["events"]:
                    if event["frame"] < x["num_frames"]:
                        self._flat_labels.append((i, event["frame"]))

        self._mixup = mixup

        self._crop_dim = crop_dim
        self._transform = _get_img_transforms(self._crop_dim, is_eval)
        self._frame_reader = FrameReader(frame_dir, modality)

    def _sample_uniform(self):
        video_meta = random.choices(
            self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta["num_frames"]
        base_idx = -self._pad_len * self._stride + random.randint(
            0,
            max(0, video_len - 1 + (2 * self._pad_len -
                self._clip_len) * self._stride),
        )
        return video_meta, base_idx

    def _sample_foreground(self):
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta["num_frames"]

        lower_bound = max(
            -self._pad_len * self._stride, frame_idx - self._clip_len * self._stride + 1
        )
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) *
            self._stride, frame_idx
        )

        base_idx = (
            random.randint(lower_bound, upper_bound)
            if upper_bound > lower_bound
            else lower_bound
        )

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self):
        if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
            video_meta, base_idx = self._sample_foreground()
        else:
            video_meta, base_idx = self._sample_uniform()

        labels = np.zeros(self._clip_len, np.int64)
        event_xys = (
            np.zeros((self._clip_len, 2), np.float32)
            if (
                "kovo" in self._dataset
                or "imrcvolley" in self._dataset
                or "vnl" in self._dataset
            )
            else None
        )

        for event in video_meta["events"]:
            event_frame = event["frame"]

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride

            if (
                label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event["label"]]
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1),
                ):
                    labels[i] = label
                    if event_xys is not None:
                        event_xys[i] = event.get("xy", [0, 0])

        frames = self._frame_reader.load_frames(
            video_meta["video"],
            base_idx,
            base_idx + self._clip_len * self._stride,
            pad=True,
            stride=self._stride,
            randomize=not self._is_eval,
        )  # T, C, H, W

        if self._transform is not None:
            h, w = frames.shape[-2:]
            # create a nan vector
            out_frames = torch.zeros(1) / 0.0
            if event_xys is not None:
                scale = torch.tensor([w, h]).reshape(1, 2).to(torch.float32)
                # ensure numpy -> float32 tensor to avoid dtype issues
                in_boxes = torch.from_numpy(event_xys.astype("float32")) * scale  # T, 2
                in_boxes = in_boxes.repeat((1, 2))  # T, 4

                canvas_size = (int(h), int(w))
                in_boxes = tv_tensors.BoundingBoxes(
                    in_boxes, format="XYXY", canvas_size=canvas_size
                )

                while out_frames.isnan().any():
                    # if some transformation messes up the image we try again
                    out_frames, out_boxes = self._transform(frames, in_boxes)

                frames = out_frames
                event_xys = out_boxes.data[:, :2] / torch.tensor(
                    [self._crop_dim, self._crop_dim]
                ).reshape(1, 2)
            else:
                # No boxes available: run transforms on frames only
                while out_frames.isnan().any():
                    out_frames = self._transform(frames)
                frames = out_frames

        return {
            "video": video_meta["video"],
            "start": base_idx,
            "end": base_idx + self._clip_len * self._stride,
            "frame": frames,
            "contains_event": int(np.sum(labels) > 0),
            "label": labels,
            "xy": event_xys,
        }

    def __getitem__(self, unused):
        ret = self._get_one()

        if self._mixup:
            mix = self._get_one()  # Sample another clip
            l = random.betavariate(0.2, 0.2)
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))
            label_dist[range(self._clip_len), ret["label"]] = l
            label_dist[range(self._clip_len), mix["label"]] += 1.0 - l

            if self._gpu_transform is None:
                ret["frame"] = l * ret["frame"] + (1.0 - l) * mix["frame"]
            else:
                ret["mix_frame"] = mix["frame"]
                ret["mix_weight"] = l

            ret["contains_event"] = max(
                ret["contains_event"], mix["contains_event"])
            ret["label"] = label_dist

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class ActionSpotVideoDataset(Dataset):

    def __init__(
        self,
        classes,
        label_file,
        frame_dir,
        modality,
        clip_len,
        overlap_len=0,
        crop_dim=None,
        stride=1,
        pad_len=DEFAULT_PAD_LEN,
        flip=False,
        multi_crop=False,
        skip_partial_end=True,
        num_videos=None,
        is_eval=True,
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)

        if isinstance(num_videos, int) and num_videos > 0:
            self._labels = self._labels[:num_videos]

        self._class_dict = classes
        self._video_idxs = {x["video"]: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride

        self._transform = _get_img_transforms(crop_dim, is_eval)

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(frame_dir, modality)

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(
                    0, l["num_frames"] -
                    (overlap_len * stride) * int(skip_partial_end)
                ),  # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride,
            ):
                has_clip = True
                self._clips.append((l["video"], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, start = self._clips[idx]
        frames = self._frame_reader.load_frames(
            video_name,
            start,
            start + self._clip_len * self._stride,
            pad=True,
            stride=self._stride,
        )

        if self._transform is not None:
            frames = self._transform(frames)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {"video": video_name, "start": start // self._stride, "frame": frames}

    def get_labels(self, video, with_locations=False):
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta["num_frames"]
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int32)
        if with_locations:
            locations = np.zeros((num_labels, 2), np.float32)
        for event in meta["events"]:
            frame = event["frame"]
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event["label"]]
                if with_locations:
                    locations[frame // self._stride] = event.get("xy", [0, 0])
            else:
                print(
                    "Warning: {} >= {} is past the end {}".format(
                        frame, num_frames, meta["video"]
                    )
                )
        if with_locations:
            return labels, locations
        return labels

    @property
    def augment(self):
        return self._flip or self._multi_crop

    @property
    def videos(self):
        return sorted(
            [
                (v["video"], v["num_frames"] //
                 self._stride, v["fps"] / self._stride)
                for v in self._labels
            ]
        )

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy["fps"] /= self._stride
                x_copy["num_frames"] //= self._stride
                for e in x_copy["events"]:
                    e["frame"] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x["num_frames"] for x in self._labels])
        num_events = sum([len(x["events"]) for x in self._labels])
        print(
            "{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg".format(
                self._src_file,
                len(self._labels),
                num_frames,
                self._stride,
                num_events / num_frames * 100,
            )
        )


if __name__ == "__main__":
    from util.dataset import DATASETS, load_classes
    from easydict import EasyDict as edict

    args = edict()
    args.dataset = "kovo_288p"
    classes = load_classes(os.path.join("data", args.dataset, "class.txt"))
    args.crop_dim = 224
    args.dilate_len = 0
    args.mixup = False
    args.clip_len = 16
    args.modality = "rgb"
    dataset_len = 300_000 // args.clip_len
    dataset_kwargs = {
        "crop_dim": args.crop_dim,
        "dilate_len": args.dilate_len,
        "mixup": args.mixup,
        "dataset": args.dataset,
    }
    train_data = ActionSpotDataset(
        classes,
        os.path.join("data", args.dataset, "train.json"),
        args.frame_dir,
        args.modality,
        args.clip_len,
        dataset_len,
        is_eval=False,
        **dataset_kwargs,
    )
    train_data.print_info()
