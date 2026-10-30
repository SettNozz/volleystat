#!/usr/bin/env python3
"""Training for E2E-Spot"""

from torchvision.ops.focal_loss import sigmoid_focal_loss
from util.score import compute_mAPs, compute_mAPs_with_locations
from util.dataset import DATASETS, load_classes
from util.io import load_json, store_json, store_gz_json, clear_files
from util.eval import process_frame_predictions_with_location
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset
from model.modules import *
from model.shift import make_temporal_shift
from model.common import step, BaseRGBModel, MLP
from tqdm import tqdm
import timm
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import torch.nn.functional as F
import torch.nn as nn
import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
from tabulate import tabulate
import torch
import wandb  # Add wandb import

torch.backends.cudnn.benchmark = True


EPOCH_NUM_FRAMES = 500000

BASE_NUM_WORKERS = 4

BASE_NUM_VAL_EPOCHS = 20

INFERENCE_BATCH_SIZE = 4


# Prevent the GRU params from going too big (cap it at a RegNet-Y 800MF)
MAX_GRU_HIDDEN_DIM = 768


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("frame_dir", type=str, help="Path to extracted frames")

    parser.add_argument(
        "--modality", type=str, choices=["rgb", "bw", "flow"], default="rgb"
    )
    parser.add_argument(
        "-m",
        "--feature_arch",
        type=str,
        required=True,
        choices=[
            # From torchvision
            "rn18",
            "rn18_tsm",
            "rn18_gsm",
            "rn50",
            "rn50_tsm",
            "rn50_gsm",
            # From timm (following its naming conventions)
            "rny002",
            "rny002_tsm",
            "rny002_gsm",
            "rny008",
            "rny008_tsm",
            "rny008_gsm",
            # From timm
            "convnextt",
            "convnextt_tsm",
            "convnextt_gsm",
            # hg cv v2
            "convnextt2",
            "convnextt2_tsm",
            "convnextt2_gsm",
        ],
        help="CNN architecture for feature extraction",
    )
    parser.add_argument(
        "-t",
        "--temporal_arch",
        type=str,
        default="gru",
        # choices=['', 'gru', 'deeper_gru', 'mstcn', 'asformer'],
        help="Spotting architecture, after spatial pooling",
    )

    parser.add_argument(
        "-p",
        "--pred_loc_arch",
        type=str,
        default="mlp",
    )

    parser.add_argument(
        "--use_channel_attention",
        action="store_true",
        help="Use channel attention after the Backbone",
    )

    parser.add_argument("--clip_len", type=int, default=100)
    parser.add_argument("--crop_dim", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "-ag", "--acc_grad_iter", type=int, default=1, help="Use gradient accumulation"
    )

    parser.add_argument("--warm_up_epochs", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        required=True,
        help="Dir to save checkpoints and predictions",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint in <save_dir>",
    )

    parser.add_argument(
        "--predict_location", action="store_true", help="As the name suggests"
    )

    parser.add_argument("--start_val_epoch", type=int, default=30)
    parser.add_argument("--criterion", choices=["map", "loss"], default="map")

    parser.add_argument(
        "--dilate_len", type=int, default=0, help="Label dilation when training"
    )
    parser.add_argument("--mixup", type=bool, default=False)

    parser.add_argument(
        "-j", "--num_workers", type=int, help="Base number of dataloader workers"
    )

    # Sample based on foreground
    parser.add_argument("--fg_upsample", type=float)

    parser.add_argument("-mgpu", "--gpu_parallel", action="store_true")

    parser.add_argument(
        "--use_mse_loss",
        action="store_true",
        help="add MSE loss term to the training loss",
    )
    parser.add_argument(
        "--debug_only", action="store_true", help="As the name suggests"
    )

    parser.add_argument("--time_backward", action="store_true")

    # Eval mode
    parser.add_argument("--eval_only", action="store_true", help="As the name suggests")
    parser.add_argument("--eval_split", type=str, default="test")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path of checkpoint for eval",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="spot", help="Name of the W&B project."
    )

    return parser.parse_args()


def focal_loss_multiclass_with_logits(
    input,
    target,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
):
    ce_loss = F.cross_entropy(input, target, reduction="none")
    pt = torch.exp(-ce_loss)
    weight = (1 - pt) ** gamma
    focal_loss = weight * ce_loss
    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss


def calculate_loss_contrast(im_feat, labels):
    """
    Calculate the loss contrast based on the provided labels and image features.

    Args:
    - labels (torch.Tensor): A tensor of labels corresponding to the features.
    - im_feat (torch.Tensor): A tensor representing the image features.
    - num_classes (int): The number of classes (default is 3).
    - target_class (int): The class label of interest for contrast calculation (default is 2).

    Returns:
    - loss_contrast (float): The calculated loss contrast value.
    """
    if (labels == 0).all():
        return torch.tensor(0.0)
    # Create foreground mask (excluding background class 0)
    fg_mask = labels != 0

    # Extract foreground and background features based on masks
    fg_feat = im_feat[fg_mask]
    bg_feat = im_feat[~fg_mask]

    # Calculate the delta (difference) between foreground and background features
    delta = torch.norm(fg_feat.unsqueeze(1) - bg_feat.unsqueeze(0), dim=-1).mean()

    # Compute the loss contrast
    loss_contrast = 1 / (delta.mean() + 1e-6)

    return loss_contrast


class E2EModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(
            self,
            num_classes,
            feature_arch,
            temporal_arch,
            clip_len,
            modality,
            predict_location=False,
            pred_loc_arch="mlp",
            time_backward=False,
            use_channel_attention=False,
        ):
            """
            Initializes the end-to-end spatial model.

            This constructor sets up the feature extraction backbone, temporal modeling head,
            and optionally, a location prediction head. It supports various architectures
            for both feature extraction and temporal modeling.

            Args:
                num_classes (int): Number of classes for the final classification.
                feature_arch (str): Architecture for the feature extractor.
                    Supported: "rn18", "rn50" (ResNet variants),
                               "rny002", "rny008" (RegNetY variants),
                               "convnextt", "convnextt2" (ConvNeXt variants).
                    Suffixes "_tsm" or "_gsm" can be added to enable Temporal Shift
                    Modules or Gated Shift Modules, respectively.
                temporal_arch (str): Architecture for the temporal modeling head.
                    Supported: "gru", "deeper_gru", "mingru" (GRU variants),
                               "" (simple Fully Connected layer).
                clip_len (int): Length of the input video clips (number of frames).
                modality (str): Input data modality.
                    Supported: "rgb", "flow", "bw" (black & white).
                predict_location (bool, optional): If True, a location prediction
                    head is added. Defaults to False.
                pred_loc_arch (str, optional): Architecture for the location
                    prediction head. Supported: "mlp". Defaults to "mlp".
                time_backward (bool, optional): If True, processes temporal sequences
                    in reverse for models like GRU. Defaults to False.
                use_channel_attention (bool, optional): If True, adds a channel
                    attention module after the feature extractor. Defaults to False.
            """

            super().__init__()
            self._pred_loc_arch = pred_loc_arch
            self.time_backward = time_backward
            is_rgb = modality == "rgb"
            in_channels = {"flow": 2, "bw": 1, "rgb": 3}[modality]
            self.feature_arch = feature_arch
            self._use_channel_attention = use_channel_attention

            if feature_arch.startswith(("rn18", "rn50")):
                resnet_name = feature_arch.split("_")[0].replace("rn", "resnet")
                features = getattr(torchvision.models, resnet_name)(pretrained=is_rgb)
                feat_dim = features.fc.in_features
                features.fc = nn.Identity()
                # import torchsummary
                # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))

                # Flow has only two input channels
                if not is_rgb:
                    # FIXME: args maybe wrong for larger resnet
                    features.conv1 = nn.Conv2d(
                        in_channels,
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False,
                    )

            elif feature_arch.startswith(("rny002", "rny008")):
                features = timm.create_model(
                    {
                        "rny002": "regnety_002",
                        "rny008": "regnety_008",
                    }[feature_arch.rsplit("_", 1)[0]],
                    pretrained=is_rgb,
                )
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

                if not is_rgb:
                    features.stem.conv = nn.Conv2d(
                        in_channels,
                        32,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1),
                        bias=False,
                    )

            elif "convnextt2" in feature_arch:
                from transformers import ConvNextV2ForImageClassification

                features = ConvNextV2ForImageClassification.from_pretrained(
                    "facebook/convnextv2-pico-1k-224"
                )
                feat_dim = features.classifier.in_features
                features.classifier = nn.Identity()

            elif "convnextt" in feature_arch:
                features = timm.create_model("convnext_tiny", pretrained=is_rgb)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

                if not is_rgb:
                    features.stem[0] = nn.Conv2d(
                        in_channels, 96, kernel_size=4, stride=4
                    )

            else:
                raise NotImplementedError(feature_arch)

            if self._use_channel_attention:
                from model.modules import ChannelAttention

                self._channel_attention = ChannelAttention(feat_dim, reduction=8)

            else:
                self._channel_attention = nn.Identity()
            # Add Temporal Shift Modules
            self._require_clip_len = -1
            if feature_arch.endswith("_tsm"):
                make_temporal_shift(features, clip_len, is_gsm=False)
                self._require_clip_len = clip_len
            elif feature_arch.endswith("_gsm"):
                make_temporal_shift(features, clip_len, is_gsm=True)
                self._require_clip_len = clip_len

            self._features = features
            self._feat_dim = feat_dim

            if "gru" in temporal_arch:
                hidden_dim = feat_dim
                if hidden_dim > MAX_GRU_HIDDEN_DIM:
                    hidden_dim = MAX_GRU_HIDDEN_DIM
                    print(
                        "Clamped GRU hidden dim: {} -> {}".format(feat_dim, hidden_dim)
                    )
                if temporal_arch in ("gru", "deeper_gru"):
                    self._pred_fine = GRUPrediction(
                        feat_dim,
                        num_classes,
                        hidden_dim,
                        num_layers=3 if temporal_arch[0] == "d" else 1,
                    )
                elif temporal_arch == "mingru":
                    from model.min_gru import MinRNNPredictor

                    self._pred_fine = MinRNNPredictor(
                        input_size=feat_dim,
                        hidden_size=hidden_dim,
                        output_size=num_classes,
                        n_layers=3,
                        rnn_type="mingru",
                        batch_first=True,
                    )
                else:
                    raise NotImplementedError(temporal_arch)
            elif temporal_arch == "":
                self._pred_fine = FCPrediction(feat_dim, num_classes)
            else:
                raise NotImplementedError(temporal_arch)

            self._predict_location = predict_location
            if self._predict_location:
                if pred_loc_arch == "mlp":
                    self._pred_loc = nn.Sequential(
                        MLP(
                            hidden_dim,
                            hidden_dim * 3,
                            output_dim=hidden_dim,
                            num_layers=2,
                        ),
                        nn.Linear(hidden_dim, 2),
                    )
                else:
                    raise NotImplementedError(
                        f"Unimplemented location predictor: {pred_loc_arch}"
                    )

        def forward(self, x):
            batch_size, true_clip_len, channels, height, width = x.shape

            if self.time_backward:
                x = x.flip(1)

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                # TSM module requires clip len to be known
                assert (
                    true_clip_len <= self._require_clip_len
                ), "Expected {}, got {}".format(self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len
            im_feat = self._features(x.view(-1, channels, height, width))

            if "convnextt2" in self.feature_arch:
                im_feat = im_feat.logits

            im_feat = im_feat.reshape(batch_size, clip_len, self._feat_dim)
            if self._use_channel_attention:
                im_feat = self._channel_attention(im_feat)

            if true_clip_len != clip_len:
                # Undo padding
                im_feat = im_feat[:, :true_clip_len, :]

            loc_feat = None
            if self._predict_location:
                loc_feat = self._pred_loc(im_feat)
                if self.time_backward:
                    loc_feat = loc_feat.flip(1)

            ev_feat = self._pred_fine(im_feat)
            return {
                "im_feat": ev_feat if not self.time_backward else ev_feat.flip(1),
                "loc_feat": loc_feat,
                "cnn_feat": im_feat,
            }

        def print_stats(self):
            print(f"Model params:{sum(p.numel() for p in self.parameters()):,}")
            print(
                f"CNN features:{sum(p.numel() for p in self._features.parameters()):,}"
            )
            print(
                f"Temporal Head:{sum(p.numel() for p in self._pred_fine.parameters()):,}"
            )
            if hasattr(self, "_pred_loc"):
                print(
                    f"Spatial Head:{sum(p.numel() for p in self._pred_loc.parameters()):,}"
                )

    def __init__(
        self,
        num_classes,
        feature_arch,
        temporal_arch,
        clip_len,
        modality,
        device="cuda",
        predict_location=False,
        multi_gpu=False,
        pred_loc_arch="mlp",
        time_backward=False,
        use_channel_attention=False,
    ):
        self.device = device
        self._multi_gpu = multi_gpu
        self._model = E2EModel.Impl(
            num_classes,
            feature_arch,
            temporal_arch,
            clip_len,
            modality,
            predict_location,
            pred_loc_arch=pred_loc_arch,
            time_backward=time_backward,
            use_channel_attention=use_channel_attention,
        )
        # self._model = torch.compile(self._model)
        self._model.print_stats()

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes
        self._temporal_arch = temporal_arch

    def epoch(
        self,
        loader,
        optimizer=None,
        scaler=None,
        lr_scheduler=None,
        acc_grad_iter=1,
        fg_weight=5,
    ):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs["weight"] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)
            ).to(self.device)

        epoch_loss = 0.0
        epoch_loss_cls = 0.0
        epoch_loss_loc = 0.0
        epoch_loss_contrast = 0.0

        ctx = torch.cuda.amp.autocast()

        with torch.no_grad() if optimizer is None else nullcontext():
            pbar = tqdm(loader)
            for batch_idx, batch in enumerate(pbar):
                # frame = loader.dataset.load_frame_gpu(batch, self.device)
                frame = batch["frame"].to(self.device)
                label = batch["label"].to(self.device)

                if self._model._predict_location:
                    target_xy = batch["xy"].to(self.device).reshape(-1, 2)  # B*T, 2

                # Depends on whether mixup is used
                label = (
                    label.flatten()
                    if len(label.shape) == 2
                    else label.view(-1, label.shape[-1])
                )

                with ctx if optimizer is not None else nullcontext():
                    preds = self._model(frame)

                    cnn_feat = preds["cnn_feat"]
                    pred = preds["im_feat"]
                    loc = preds["loc_feat"]

                    loss = 0.0
                    loss_cls = 0.0
                    loss_loc = 0.0

                    # loss_contrast = calculate_loss_contrast(cnn_feat.flatten(0,1), label)
                    loss_contrast = torch.tensor(0.0).to(self.device)

                    if len(pred.shape) == 3:
                        pred = pred.unsqueeze(0)

                    for i in range(pred.shape[0]):
                        loss_cls += F.cross_entropy(
                            pred[i].reshape(-1, self._num_classes), label, **ce_kwargs
                        )

                        # loss_cls += focal_loss_multiclass_with_logits(
                        #     pred[i].reshape(-1, self._num_classes), label, gamma=2.0, reduction='mean'
                        # )

                    if self._model._predict_location:
                        # Assume the objectness score is the first element in loc[i], and the rest are x and y coordinates.
                        pred_loc = loc.reshape(-1, 2)  # B*T, 2

                        event_mask = (label != 0).float().reshape(-1)  # B*T

                        # Apply the standard L1 loss to the masked x and y coordinates
                        xy_loss = F.l1_loss(
                            pred_loc.sigmoid(), target_xy, reduction="none"
                        ).sum(dim=-1)

                        loss_loc += (xy_loss * event_mask).sum() / (
                            event_mask.sum() + 1e-6
                        )

                    loss = loss_cls + loss_loc + loss_contrast * 0.1

                if optimizer is not None:
                    step(
                        optimizer,
                        scaler,
                        loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0,
                    )

                epoch_loss += loss.detach().item()
                epoch_loss_cls += loss_cls.detach().item()
                epoch_loss_contrast += loss_contrast.detach().item()

                if self._model._predict_location:
                    epoch_loss_loc += loss_loc.detach().item()

                pbar.set_postfix(
                    {
                        "sum": loss.detach().item(),
                        "cls": loss_cls.detach().item(),
                        "loc": loss_loc.detach().item(),
                        "contrast": loss_contrast.detach().item(),
                    }
                )

        return {
            "sum": epoch_loss / len(loader),
            "cls": epoch_loss_cls / len(loader),
            "loc": epoch_loss_loc / len(loader),
            "contrast": epoch_loss_contrast / len(loader),
        }

    def predict(self, seq, use_amp=False, presence_threshold=0.0):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)

        ctx = torch.cuda.amp.autocast()

        self._model.eval()
        with torch.no_grad():
            with ctx if use_amp else nullcontext():
                pred_dict = self._model(seq)
            pred_cls_score = torch.softmax(pred_dict["im_feat"], axis=2).cpu().numpy()
            pred_cls = torch.argmax(pred_dict["im_feat"], axis=2).cpu().numpy()
            if self._model._predict_location:
                pred_loc = (
                    pred_dict["loc_feat"]
                    .reshape(seq.shape[0], -1, 2)
                    .sigmoid()
                    .cpu()
                    .numpy()
                )  # B, T, 2
                return pred_cls, pred_cls_score, pred_loc
            else:
                return pred_cls, pred_cls_score


def evaluate(
    model,
    dataset,
    split,
    classes,
    save_pred,
    calc_stats=True,
    save_scores=True,
    predict_location=False,
    px_scale=224,
):
    # Initialize the prediction dictionary and location errors list
    pred_dict = {}

    # Initialize the prediction dictionary with zero arrays for each video
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros(
                (video_len, len(classes) + 1), np.float32
            ),  # Stores scores for each class
            np.zeros(video_len, np.int32),  # Stores support (number of frames)
            # Stores location predictions (x, y)
            np.zeros((video_len, 2), np.float32),
        )

    # Determine batch size based on whether the dataset is augmented
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    idx = 0
    # Iterate over the dataset using DataLoader
    for clip in tqdm(
        DataLoader(
            dataset,
            num_workers=BASE_NUM_WORKERS * 2,
            # pin_memory=True,
            batch_size=batch_size,
        )
    ):
        if batch_size > 1:
            # When batch size is greater than 1 (batched by dataloader)
            if predict_location:
                # Predict scores and locations if location prediction is enabled
                _, batch_pred_scores, batch_pred_loc = model.predict(clip["frame"])

                batch_pred_loc = batch_pred_loc.reshape(
                    batch_pred_scores.shape[0], -1, 2
                )
            else:
                # Predict only scores if location prediction is not enabled
                _, batch_pred_scores = model.predict(clip["frame"])

            for i in range(clip["frame"].shape[0]):
                video = clip["video"][i]
                scores, support, locations = pred_dict[video]

                pred_scores = batch_pred_scores[
                    i
                ]  # Predicted scores for the current batch
                if predict_location:
                    pred_loc = batch_pred_loc[
                        i
                    ]  # Predicted locations for the current batch

                start = clip["start"][i].item()
                if start < 0:
                    # Adjust predictions if the start index is negative
                    pred_scores = pred_scores[-start:, :]
                    if predict_location:
                        pred_loc = pred_loc[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    # Adjust predictions if the end index exceeds the video length
                    end = scores.shape[0]
                    pred_scores = pred_scores[: end - start, :]
                    if predict_location:
                        pred_loc = pred_loc[: end - start, :]
                scores[start:end, :] += pred_scores  # Accumulate scores
                support[
                    start:end
                ] += 1  # Increment the support count because the frame is present
                if predict_location:
                    # fg_mask = np.argmax(pred_scores, axis=-1) > 0
                    locations[start:end, :] = pred_loc

        else:
            # When batch size is 1 (batched by dataset)
            scores, support, locations = pred_dict[clip["video"][0]]

            start = clip["start"][0].item()
            if predict_location:
                # Predict scores and locations if location prediction is enabled
                _, pred_scores, pred_loc = model.predict(clip["frame"][0])
            else:
                # Predict only scores if location predictiobn is not enabled
                _, pred_scores = model.predict(clip["frame"][0])

            if start < 0:
                # Adjust predictions if the start index is negative
                pred_scores = pred_scores[:, -start:, :]
                if predict_location:
                    pred_loc = pred_loc[-start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                # Adjust predictions if the end index exceeds the video length
                end = scores.shape[0]
                pred_scores = pred_scores[:, : end - start, :]
                if predict_location:
                    pred_loc = pred_loc[:, : end - start, :]

            # Accumulate scores
            scores[start:end, :] += np.sum(pred_scores, axis=0)
            # Increment the support count
            support[start:end] += pred_scores.shape[0]
            if predict_location:
                # fg_mask = np.argmax(pred_scores, axis=-1) > 0
                locations[start:end, :] = pred_loc

    # Process the frame-level predictions (class)
    err, f1, pred_events, pred_events_high_recall, pred_scores = (
        process_frame_predictions_with_location(dataset, classes, pred_dict)
    )

    avg_mAP_t = None
    if calc_stats:
        for fg_threshold in [0.25]:
            # Print the evaluation results
            # print("=== Results on {} (w/o NMS) ===".format(split))
            print("=== Results on {} (FG_THRES={:.2f}) ===".format(split, fg_threshold))
            print("Error (frame-level): {:0.2f}\n".format(err.get() * 100))

            def get_f1_tab_row(str_k):
                k = classes[str_k] if str_k != "any" else None
                return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]

            rows = [get_f1_tab_row("any")]
            for c in sorted(classes):
                rows.append(get_f1_tab_row(c))
            print(
                tabulate(
                    rows,
                    headers=["Exact frame", "F1", "TP", "FP", "FN"],
                    floatfmt="0.2f",
                )
            )
            print()

            # Calculate mean average precision (mAP)
            # mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall)
            mAPs_t, mAPs_p = compute_mAPs_with_locations(
                dataset.labels,
                pred_events_high_recall,
                px_scale=px_scale,
                fg_threshold=fg_threshold,
            )
            avg_mAP_t = np.mean(mAPs_t[1:])
            avg_mAP_s = np.mean(mAPs_p)

            # hamornic mean
            avg_mAP = 2 * avg_mAP_t * avg_mAP_s / (avg_mAP_t + avg_mAP_s + 1e-6)
            print("Harmonic mean (temporal and spatial mAPs): {:0.2%}".format(avg_mAP))

    if save_pred is not None:
        os.makedirs(os.path.dirname(save_pred), exist_ok=True)
        # Save predictions and scores if requested
        store_json(save_pred + ".json", pred_events)
        store_gz_json(save_pred + ".recall.json.gz", pred_events_high_recall)
        if save_scores:
            store_gz_json(save_pred + ".score.json.gz", pred_scores)

    print("=" * 50)
    return avg_mAP  # Return the average mean average precision (mAP)


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith("optim_"):
            continue
        epoch = int(os.path.splitext(file_name)[0].split("optim_")[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, "loss.json"))
    if criterion == "map":
        key = "val_mAP"
        best = max(data, key=lambda x: x[key])
    else:
        key = "val"
        best = min(data, key=lambda x: x[key])
    return data, best["epoch"], best[key]


def get_datasets(args):
    # Prefer explicit class file; if missing, try to derive classes from
    # a train.json (supports `data/<dataset>/volli_dataset_json/train.json`).
    base_dir = os.path.join("data", args.dataset)
    class_file = os.path.join(base_dir, "class.txt")
    if os.path.exists(class_file):
        classes = load_classes(class_file)
    else:
        # Look for train.json in standard location or in `volli_dataset_json`
        candidate = os.path.join(base_dir, "train.json")
        alt_candidate = os.path.join(base_dir, "volli_dataset_json", "train.json")
        chosen = (
            candidate
            if os.path.exists(candidate)
            else (alt_candidate if os.path.exists(alt_candidate) else None)
        )
        if chosen is None:
            raise FileNotFoundError(
                f"Missing class.txt and no train.json found for dataset {args.dataset}"
            )
        data = load_json(chosen)
        labels = set()
        for vid in data:
            for ev in vid.get("events", []):
                labels.add(ev["label"])
        labels = sorted(labels)
        classes = {x: i + 1 for i, x in enumerate(labels)}

    dataset_len = EPOCH_NUM_FRAMES // args.clip_len
    dataset_kwargs = {
        "crop_dim": args.crop_dim,
        "dilate_len": args.dilate_len,
        "mixup": args.mixup,
        "dataset": args.dataset,
    }

    val_data_frames = None
    if args.criterion == "map":
        # Only perform mAP evaluation during training if criterion is mAP
        val_data_frames = ActionSpotVideoDataset(
            classes,
            os.path.join("data", args.dataset, "val.json"),
            args.frame_dir,
            args.modality,
            args.clip_len,
            is_eval=True,
            crop_dim=args.crop_dim,
            overlap_len=0,
            num_videos=2 if args.debug_only else None,
        )

    if args.fg_upsample is not None:
        assert args.fg_upsample > 0
        dataset_kwargs["fg_upsample"] = args.fg_upsample

    if not args.eval_only:
        print("Dataset size:", dataset_len)
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
        val_data = ActionSpotDataset(
            classes,
            os.path.join("data", args.dataset, "val.json"),
            args.frame_dir,
            args.modality,
            args.clip_len,
            dataset_len // 4,
            is_eval=True,
            **dataset_kwargs,
        )
        val_data.print_info()
        return classes, train_data, val_data, val_data_frames

    return classes, val_data_frames


def load_from_save(args, model, optimizer, scaler, lr_scheduler):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print("Loading from epoch {}".format(epoch))
    checkpoint = torch.load(
        os.path.join(args.save_dir, "checkpoint_{:03d}.pt".format(epoch))
    )

    # Filter out fc layer which causes size mismatch
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if "_fc_out" not in k}
    filter_applied = len(filtered_checkpoint) < len(checkpoint)

    if filter_applied:
        print(
            f"Loading checkpoint with strict=False (filtered {len(checkpoint) - len(filtered_checkpoint)} keys containing '_fc_out')"
        )
    else:
        print("Loading checkpoint with strict=False")

    model.load(filtered_checkpoint, strict=False)

    if args.resume:
        if filter_applied:
            print(
                "Skipping optimizer/scheduler load due to model architecture change (keys filtered)."
            )
        else:
            # print('(Resume) Train loss:', model.epoch(train_loader))
            # print('(Resume) Val loss:', model.epoch(val_loader))
            try:
                opt_data = torch.load(
                    os.path.join(args.save_dir, "optim_{:03d}.pt".format(epoch))
                )
                optimizer.load_state_dict(opt_data["optimizer_state_dict"])
                scaler.load_state_dict(opt_data["scaler_state_dict"])
                lr_scheduler.load_state_dict(opt_data["lr_state_dict"])
            except Exception as e:
                print(
                    f"Warning: Failed to load optimizer/scheduler state (likely due to model architecture change). Resetting optimizer. Error: {e}"
                )

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion
    )
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        "dataset": args.dataset,
        "num_classes": len(classes),
        "modality": args.modality,
        "feature_arch": args.feature_arch,
        "temporal_arch": args.temporal_arch,
        "clip_len": args.clip_len,
        "batch_size": args.batch_size,
        "crop_dim": args.crop_dim,
        "num_epochs": num_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "learning_rate": args.learning_rate,
        "start_val_epoch": args.start_val_epoch,
        "gpu_parallel": args.gpu_parallel,
        "epoch_num_frames": EPOCH_NUM_FRAMES,
        "dilate_len": args.dilate_len,
        "mixup": args.mixup,
        "fg_upsample": args.fg_upsample,
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    # if args.gpu_parallel:
    #     n *= torch.cuda.device_count()
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print(
        "Using Linear Warmup ({}) + Cosine Annealing LR ({})".format(
            args.warm_up_epochs, cosine_epochs
        )
    )
    return args.num_epochs, ChainedScheduler(
        [
            LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=args.warm_up_epochs * num_steps_per_epoch,
            ),
            CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs),
        ]
    )


def main(args):
    if args.debug_only:
        global EPOCH_NUM_FRAMES
        EPOCH_NUM_FRAMES = 500

        global BASE_NUM_VAL_EPOCHS
        BASE_NUM_VAL_EPOCHS = 2

    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

        assert args.batch_size % args.acc_grad_iter == 0
        if args.start_val_epoch is None:
            args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
        if args.crop_dim <= 0:
            args.crop_dim = None

    wandb.init(project=args.wandb_project, config=args)  # Initialize wandb

    _data = get_datasets(args)
    classes = _data[0]

    model = E2EModel(
        len(classes) + 1,
        args.feature_arch,
        args.temporal_arch,
        clip_len=args.clip_len,
        modality=args.modality,
        multi_gpu=args.gpu_parallel,
        predict_location=args.predict_location,
        pred_loc_arch=args.pred_loc_arch,
        time_backward=args.time_backward,
        use_channel_attention=args.use_channel_attention,
    )

    if not args.eval_only:
        classes, train_data, val_data, val_data_frames = _data

        def worker_init_fn(id):
            random.seed(id + epoch * 100)

        loader_batch_size = args.batch_size // args.acc_grad_iter
        train_loader = DataLoader(
            train_data,
            shuffle=False,
            batch_size=loader_batch_size,
            pin_memory=True,
            num_workers=get_num_train_workers(args),
            prefetch_factor=1 if args.num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=loader_batch_size,
            pin_memory=True,
            num_workers=BASE_NUM_WORKERS,
            worker_init_fn=worker_init_fn,
        )

        optimizer, scaler = model.get_optimizer({"lr": args.learning_rate})

        # Warmup schedule
        num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch
        )

        losses = []
        best_epoch = None
        best_criterion = -0.1 if args.criterion == "map" else float("inf")

        epoch = 0
        if args.resume:
            epoch, losses, best_epoch, best_criterion = load_from_save(
                args, model, optimizer, scaler, lr_scheduler
            )
            epoch += 1

        # model._model = torch.compile(model._model)
        # Write it to console
        store_config("/dev/stdout", args, num_epochs, classes)

        for epoch in range(epoch, num_epochs):
            train_loss_dict = model.epoch(
                train_loader,
                optimizer,
                scaler,
                lr_scheduler=lr_scheduler,
                acc_grad_iter=args.acc_grad_iter,
            )
            val_loss_dict = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)

            print(
                "\n",
                tabulate(
                    [
                        [
                            "Train loss",
                            train_loss_dict["cls"],
                            train_loss_dict["loc"],
                            train_loss_dict["contrast"],
                            train_loss_dict["sum"],
                        ],
                        [
                            "Val loss",
                            val_loss_dict["cls"],
                            val_loss_dict["loc"],
                            val_loss_dict["contrast"],
                            val_loss_dict["sum"],
                        ],
                    ],
                    headers=[f"Epoch: {epoch}", "cls", "loc", "contrast", "sum"],
                    floatfmt=".5f",
                ),
            )

            val_mAP = 0
            if args.criterion == "loss":
                if val_loss_dict["sum"] < best_criterion:
                    best_criterion = val_loss_dict["sum"]
                    best_epoch = epoch
                    print("New best epoch!")
            elif args.criterion == "map":
                if epoch >= args.start_val_epoch:
                    pred_file = None
                    if args.save_dir is not None:
                        pred_file = os.path.join(
                            args.save_dir, "pred-val.{}".format(epoch)
                        )
                        os.makedirs(args.save_dir, exist_ok=True)
                    val_mAP = evaluate(
                        model,
                        val_data_frames,
                        "VAL",
                        classes,
                        pred_file,
                        save_scores=False,
                        predict_location=args.predict_location,
                    )
                    if args.criterion == "map" and val_mAP > best_criterion:
                        best_criterion = val_mAP
                        best_epoch = epoch
                        print("New best epoch!")
            else:
                print("Unknown criterion:", args.criterion)

            losses.append(
                {
                    "epoch": epoch,
                    "train": train_loss_dict["sum"],
                    "val": val_loss_dict["sum"],
                    "val_mAP": val_mAP,
                }
            )
            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss_dict["sum"],
                        "val_loss": val_loss_dict["sum"],
                        "val_mAP": val_mAP,
                    }
                )  # Log metrics to wandb
                store_json(
                    os.path.join(args.save_dir, "loss.json"), losses, pretty=True
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, "checkpoint_{:03d}.pt".format(epoch)),
                )
                clear_files(args.save_dir, r"optim_\d+\.pt")
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "lr_state_dict": lr_scheduler.state_dict(),
                    },
                    os.path.join(args.save_dir, "optim_{:03d}.pt".format(epoch)),
                )
                store_config(
                    os.path.join(args.save_dir, "config.json"),
                    args,
                    num_epochs,
                    classes,
                )

        print("Best epoch: {}\n".format(best_epoch))

        if args.save_dir is not None:
            model.load(
                torch.load(
                    os.path.join(
                        args.save_dir, "checkpoint_{:03d}.pt".format(best_epoch)
                    )
                )
            )

    else:
        best_epoch = "eval_only"
        model.load(torch.load(os.path.abspath(args.checkpoint_path)))

    # Evaluate on VAL if not already done
    eval_splits = ["val"] if args.criterion != "map" else []

    # Evaluate on hold out splits
    eval_splits += [args.eval_split, "challenge"]
    for split in eval_splits:
        split_path = os.path.join("data", args.dataset, "{}.json".format(split))
        if os.path.exists(split_path):
            split_data = ActionSpotVideoDataset(
                classes,
                split_path,
                args.frame_dir,
                args.modality,
                args.clip_len,
                is_eval=True,
                # overlap_len=args.clip_len // 2,
                overlap_len=0,
                crop_dim=args.crop_dim,
                num_videos=2 if args.debug_only else None,
            )
            split_data.print_info()

            pred_file = None
            if args.save_dir is not None:
                pred_file = os.path.join(
                    args.save_dir, "pred-{}.{}".format(split, best_epoch)
                )

            evaluate(
                model,
                split_data,
                split.upper(),
                classes,
                pred_file,
                calc_stats=split != "challenge",
                predict_location=args.predict_location,
            )


if __name__ == "__main__":
    main(get_args())
