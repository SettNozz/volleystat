from gradio import Blocks, Video, Dataframe, Button, State
import gradio as gr
import easydict
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from util.io import load_json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from util.score import (
    filter_events_by_score,
    non_max_suppression_events,
)

from inference_on_mp4 import (
    get_args,
    get_model,
    VideoFrameSlidingDataset,
    run_inference,
    render_video,
)
import json
from openai import OpenAI

raw_results = None  # add global variable to store predictions


def reply_message(message, history):
    global raw_results  # use global raw_results
    system_message = "Volleyball Events: "
    if raw_results:
        system_message += json.dumps(raw_results)
    else:
        system_message += "No prediction results available."
    history.extend(
        [
            {"role": "user", "content": message},
            {"role": "system", "content": system_message},
        ]
    )
    stream = client.chat.completions.create(
        messages=history,
        model="google/gemini-2.0-flash-lite-preview-02-05:free",
        stream=True,
    )
    chunks = []
    for chunk in stream:
        chunks.append(chunk.choices[0].delta.content or "")
        yield "".join(chunks)

    return message


def process_video(video_file):
    dataset = VideoFrameSlidingDataset(
        video_file,
        mode="mp4",
        window_size=args.clip_len,
        stride=args.clip_len,
        channel_first=False,
        transform=transform,
    )
    fps = dataset.fps

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=os.cpu_count() // 2
    )

    pred_events = run_inference(
        model,
        dataloader,
        args.classes,
        args.pred_file,
        postprocess=True,
    )
    results = pred_events[0]["events"]
    global raw_results
    raw_results = results  # save the results globally

    if len(results) == 0:
        results_df = pd.DataFrame([{"message": "No events detected"}])
        btn_update = gr.update(visible=False)
        vid_update = gr.update(visible=False)
    else:
        results = sorted(results, key=lambda x: x["frame"])
        # Add timestamp (in seconds) to each event
        for event in results:
            event["timestamp"] = round(event["frame"] / fps, 2)
        results_df = pd.DataFrame(results)
        # Rename columns and include Timestamp
        results_df.columns = ["Event", "Frame", "Score", "Location", "Timestamp (sec)"]
        results_df["Score"] = results_df["Score"].apply(lambda x: round(x, 2))
        results_df["Location"] = results_df["Location"].apply(
            lambda coords: [round(v, 2) for v in coords]
        )
        btn_update = gr.update(visible=True)
        vid_update = gr.update(visible=True)
    # Return predictions, original video file, raw results and UI updates for render_button and preview.
    return (
        results_df,
        video_file,
        results,
        btn_update,
        vid_update,
        gr.update(visible=True),
    )


def render_results(video_file, results):
    print("Rendering video...")
    rendered_video = render_video(video_file, results)
    print("Video rendered:", rendered_video)
    return rendered_video





if __name__ == "__main__":
    global model, transform, args, client, results_state

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )

    print("Loading model")
    config_dir = "exp/e2espatial_vnl2"
    config_path = os.path.join(config_dir, "config.json")
    checkpoint_path = os.path.join(config_dir, "checkpoint_140.pt")
    save_dir = os.path.join(config_dir, "tmp")
    os.makedirs(save_dir, exist_ok=True)
    args = get_args(
        **load_json(config_path), checkpoint_path=checkpoint_path, save_dir=save_dir
    )

    transform = v2.Compose(
        [
            v2.Resize(args.crop_dim, antialias=False),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = get_model(args)
    args = easydict.EasyDict(args)
    args.pred_file = os.path.join(args.save_dir, "predictions.json")

    # Set up a Blocks interface with states and buttons
    with Blocks() as demo:
        gr.Markdown("# Spatio-Temporal Event Spotting in Volleyball Videos")
        gr.Markdown(
            "Upload a video clip and click 'Detect Events' to analyze events in your video."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_input = Video(label="Upload Video Clip", autoplay=True)
                process_button = Button("Detect Events")
                predictions_output = Dataframe(
                    label="Event Detection Results",
                    value=pd.DataFrame([{"Message": "..."}]),
                )
                video_state = State()  # stores the original video file
                results_state = State()  # stores raw results from inference

            with gr.Column(scale=1):
                render_button = Button("Render results", visible=False)
                rendered_video = Video(
                    label="Rendered Video Preview",
                    visible=False,
                    autoplay=True,
                    # height=240,
                )
                chat_container = gr.Column(visible=False)
                with chat_container:
                    gr.ChatInterface(
                        fn=reply_message,
                        type="messages",
                        title="Ask me about the volleyball events!",
                    )

        process_button.click(
            process_video,
            inputs=video_input,
            outputs=[
                predictions_output,
                video_state,
                results_state,
                render_button,
                rendered_video,
                chat_container,
            ],
        )
        render_button.click(
            render_results, inputs=[video_state, results_state], outputs=rendered_video
        )

    demo.launch(server_name="0.0.0.0", share=False, pwa=True)
