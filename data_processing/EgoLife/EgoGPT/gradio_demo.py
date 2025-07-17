###  -----------------  ###
# Standard library imports
import copy
import os
import re
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)

import json
import shutil
import threading
from datetime import datetime

import gradio as gr
import librosa

# Third-party imports
import numpy as np
import requests
import spaces
import torch
import torch.distributed as dist
import uvicorn
import whisper
from decord import VideoReader, cpu
from egogpt.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    SPEECH_TOKEN_INDEX,
)
from egogpt.conversation import SeparatorStyle, conv_templates
from egogpt.mm_utils import get_model_name_from_path, process_images

# Local imports
from egogpt.model.builder import load_pretrained_model
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

# pretrained = "/mnt/sfs-common/jkyang/EgoGPT/checkpoints/EgoGPT-llavaov-7b-EgoIT-109k-release"
# pretrained = "/mnt/sfs-common/jkyang/EgoGPT/checkpoints/EgoGPT-llavaov-0.5b-EgoLife-Depersonalized-EgoIT-159k"
# pretrained = "/mnt/sfs-common/jkyang/EgoGPT/checkpoints/EgoGPT-llavaov-7b-EgoIT-120k"
pretrained = "/mnt/sfs-common/jkyang/EgoGPT_release/checkpoints/EgoGPT-7b-Demo"
device = "cuda"
device_map = "cuda"


# Add this initialization code before loading the model
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12377"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


setup(0, 1)
tokenizer, model, max_length = load_pretrained_model(pretrained, device_map=device_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

title_markdown = """
<div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(90deg, rgba(72,219,251,0.1), rgba(29,209,161,0.1)); border-radius: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px;">
    <div style="display: flex; align-items: center;">
        <a href="https://egolife-ai.github.io/" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
            <img src="https://egolife-ai.github.io/egolife.png" alt="EgoLife" style="max-width: 100px; height: auto; border-radius: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        </a>
        <div>
            <h1 style="margin: 0; background: linear-gradient(90deg, #48dbfb, #1dd1a1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em; font-weight: 700;">EgoLife</h1>
            <h2 style="margin: 10px 0; color: #2d3436; font-weight: 500;">Towards Egocentric Life Assistant</h2>
            <div style="display: flex; gap: 15px; margin-top: 10px;">
                <a href="https://egolife-ntu.github.io/" style="text-decoration: none; color: #48dbfb; font-weight: 500; transition: color 0.3s;">Project Page</a> |
                <a href="https://github.com/EvolvingLMMs-Lab/EgoLife" style="text-decoration: none; color: #48dbfb; font-weight: 500; transition: color 0.3s;">Github</a> |
                <a href="https://huggingface.co/lmms-lab" style="text-decoration: none; color: #48dbfb; font-weight: 500; transition: color 0.3s;">Huggingface</a> |
                <a href="https://arxiv.org/" style="text-decoration: none; color: #48dbfb; font-weight: 500; transition: color 0.3s;">Paper</a> |
                <a href="https://x.com/" style="text-decoration: none; color: #48dbfb; font-weight: 500; transition: color 0.3s;">Twitter (X)</a>
            </div>
        </div>
    </div>
    <div style="text-align: right; margin-left: 20px;">
        <h1 style="margin: 0; background: linear-gradient(90deg, #48dbfb, #1dd1a1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em; font-weight: 700;">EgoGPT</h1>
        <h2 style="margin: 10px 0; background: linear-gradient(90deg, #48dbfb, #1dd1a1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8em; font-weight: 600;">An Egocentric Video-Audio-Text Model<br>from EgoLife Project</h2>
    </div>
</div>
"""

notice_html = """
<div style="background-color: #f9f9f9; border-left: 5px solid #48dbfb; padding: 20px; margin-top: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
    <p style="font-size: 1.1em; color: #ff9933; margin-bottom: 10px; font-weight: bold;">💡 Pro Tip: Try accessing this demo from your phone's browser. You can use your phone's camera to capture and analyze egocentric videos, making the experience more interactive and personal.</p>
    <p style="font-size: 1.1em; color: #555; margin-bottom: 10px;">EgoGPT-7B is built upon LLaVA-OV and has been finetuned on the EgoIT dataset and a partially de-identified EgoLife dataset. Its primary goal is to serve as an egocentric captioner, supporting EgoRAG for EgoLifeQA tasks. Please note that due to inherent biases in the EgoLife dataset, the model may occasionally hallucinate details about people in custom videos based on patterns from the training data (for example, describing someone as "wearing a blue t-shirt" or "with pink hair"). We are actively working on improving the model to make it more universally applicable and will continue to release updates regularly. If you're interested in contributing to the development of future iterations of EgoGPT or the EgoLife project, we welcome you to reach out and contact us. (Contact us at <a href="mailto:jingkang001@e.ntu.edu.sg">jingkang001@e.ntu.edu.sg</a>)</p>
</div>
"""

bibtext = """
### Citation
```
@inproceedings{yang2025egolife,
  title={EgoLife\: Towards Egocentric Life Assistant},
  author={Yang, Jingkang and Liu, Shuai and Guo, Hongming and Dong, Yuhao and Zhang, Xiamengwei and Zhang, Sicheng and Wang, Pengyun and Zhou, Zitang and Xie, Binzhu and Wang, Ziyue and Ouyang, Bei and Lin, Zhengyu and Cominelli, Marco and Cai, Zhongang and Zhang, Yuanhan and Zhang, Peiyuan and Hong, Fangzhou and Widmer, Joerg and Gringoli, Francesco and Yang, Lei and Li, Bo and Liu, Ziwei},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```
"""

cur_dir = os.path.dirname(os.path.abspath(__file__))

# Add this after cur_dir definition
UPLOADS_DIR = os.path.join(cur_dir, "user_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


def time_to_frame_idx(time_int: int, fps: int) -> int:
    """
    Convert time in HHMMSSFF format (integer or string) to frame index.
    :param time_int: Time in HHMMSSFF format, e.g., 10483000 (10:48:30.00) or "10483000".
    :param fps: Frames per second of the video.
    :return: Frame index corresponding to the given time.
    """
    # Ensure time_int is a string for slicing
    time_str = str(time_int).zfill(
        8
    )  # Pad with zeros if necessary to ensure it's 8 digits

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    frames = int(time_str[6:8])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = total_seconds * fps + frames  # Convert to total frames

    return total_frames


def split_text(text, keywords):
    # 创建一个正则表达式模式，将所有关键词用 | 连接，并使用捕获组
    pattern = "(" + "|".join(map(re.escape, keywords)) + ")"
    # 使用 re.split 保留分隔符
    parts = re.split(pattern, text)
    # 去除空字符串
    parts = [part for part in parts if part]
    return parts


warnings.filterwarnings("ignore")

# Create FastAPI instance
app = FastAPI()


def load_video(
    video_path: Optional[str] = None,
    max_frames_num: int = 16,
    fps: int = 1,
    video_start_time: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_based_processing: bool = False,
) -> tuple:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    target_sr = 16000

    # Process video frames first
    if time_based_processing:
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()

        # Convert time to frame index based on the actual video FPS
        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        print("start frame", start_frame)
        print("end frame", end_frame)

        # Ensure the end time does not exceed the total frame number
        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        # Adjust start_frame and end_frame based on video start time
        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = max(0, int(round(start_frame)))  # 确保不会小于0
        end_frame = min(total_frame_num, int(round(end_frame)))  # 确保不会超过总帧数
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        # Sample frames based on the provided fps (e.g., 1 frame per second)
        frame_idx = [
            i
            for i in range(start_frame, end_frame)
            if (i - start_frame) % int(video_fps / fps) == 0
        ]

        # Get the video frames for the sampled indices
        video = vr.get_batch(frame_idx).asnumpy()
    else:
        # Original video processing logic
        total_frame_num = len(vr)
        avg_fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

        if max_frames_num > 0:
            if len(frame_idx) > max_frames_num:
                uniform_sampled_frames = np.linspace(
                    0, total_frame_num - 1, max_frames_num, dtype=int
                )
                frame_idx = uniform_sampled_frames.tolist()

        video = vr.get_batch(frame_idx).asnumpy()

    # Try to load audio, return None for speech if failed
    try:
        if time_based_processing:
            y, _ = librosa.load(video_path, sr=target_sr)
            start_sample = int(start_time * target_sr)
            end_sample = int(end_time * target_sr)
            speech = y[start_sample:end_sample]
        else:
            speech, _ = librosa.load(video_path, sr=target_sr)

        # Process audio if it exists
        speech = whisper.pad_or_trim(speech.astype(np.float32))
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])

        return video, speech, speech_lengths, True  # True indicates real audio

    except Exception as e:
        print(f"Warning: Could not load audio from video: {e}")
        # Create dummy silent audio
        duration = 10  # 10 seconds
        speech = np.zeros(duration * target_sr, dtype=np.float32)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]])
        return video, speech, speech_lengths, False  # False indicates no real audio


class PromptRequest(BaseModel):
    prompt: str
    video_path: str = None
    max_frames_num: int = 16
    fps: int = 1
    video_start_time: float = None
    start_time: float = None
    end_time: float = None
    time_based_processing: bool = False


# @spaces.GPU(duration=120)
def save_interaction(video_path, prompt, output, audio_path=None):
    """Save user interaction data and files"""
    if not video_path:
        return

    # Create timestamped directory for this interaction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interaction_dir = os.path.join(UPLOADS_DIR, timestamp)
    os.makedirs(interaction_dir, exist_ok=True)

    # Copy video file
    video_ext = os.path.splitext(video_path)[1]
    new_video_path = os.path.join(interaction_dir, f"video{video_ext}")
    shutil.copy2(video_path, new_video_path)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "prompt": prompt,
        "output": output,
        "video_path": new_video_path,
    }

    # Only try to save audio if it's a file path (str), not audio data (tuple)
    if audio_path and isinstance(audio_path, (str, bytes, os.PathLike)):
        audio_ext = os.path.splitext(audio_path)[1]
        new_audio_path = os.path.join(interaction_dir, f"audio{audio_ext}")
        shutil.copy2(audio_path, new_audio_path)
        metadata["audio_path"] = new_audio_path

    with open(os.path.join(interaction_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def extract_audio_from_video(video_path, audio_path=None):
    print("Processing audio from video...", video_path, audio_path)
    if video_path is None:
        return None

    if isinstance(video_path, dict) and "name" in video_path:
        video_path = video_path["name"]

    try:
        y, sr = librosa.load(video_path, sr=8000, mono=True, res_type="kaiser_fast")
        # Check if the audio is silent
        if np.abs(y).mean() < 0.001:
            print("Video appears to be silent")
            return None
        return (sr, y)
    except Exception as e:
        print(f"Warning: Could not extract audio from video: {e}")
        return None


import time


def generate_text(video_path, audio_track, prompt):
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    max_frames_num = 30
    fps = 1
    conv_template = "qwen_1_5"
    if video_path is None and audio_track is None:
        question = prompt
        speech = None
        speech_lengths = None
        has_real_audio = False
        image = None
        image_sizes = None
        modalities = ["image"]
        image_tensor = None
    # Load video and potentially audio
    else:
        video, speech, speech_lengths, has_real_audio = load_video(
            video_path=video_path,
            max_frames_num=max_frames_num,
            fps=fps,
        )

        # Prepare the prompt based on whether we have real audio
        if not has_real_audio:
            question = f"<image>\n{prompt}"  # Video-only prompt
        else:
            question = f"<speech>\n<image>\n{prompt}"  # Video + speech prompt

        speech = torch.stack([speech]).to("cuda").half()
        processor = model.get_vision_tower().image_processor
        processed_video = processor.preprocess(video, return_tensors="pt")[
            "pixel_values"
        ]
        image = [(processed_video, video[0].size, "video")]
        image_tensor = [image[0][0].half()]
        image_sizes = [image[0][1]]
        modalities = ["video"]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    parts = split_text(prompt_question, ["<image>", "<speech>"])
    input_ids = []
    for part in parts:
        if "<image>" == part:
            input_ids += [IMAGE_TOKEN_INDEX]
        elif (
            "<speech>" == part and speech is not None
        ):  # Only add speech token if we have audio
            input_ids += [SPEECH_TOKEN_INDEX]
        else:
            input_ids += tokenizer(part).input_ids

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generate_kwargs = {"eos_token_id": tokenizer.eos_token_id}

    def generate_response():
        model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            speech=speech,
            speech_lengths=speech_lengths,
            do_sample=False,
            temperature=0.7,
            max_new_tokens=512,
            repetition_penalty=1.2,
            modalities=modalities,
            streamer=streamer,
            **generate_kwargs,
        )

    # Start generation in a separate thread
    thread = threading.Thread(target=generate_response)
    thread.start()

    # Stream the output word by word
    generated_text = ""
    partial_word = ""
    cursor = "|"
    cursor_visible = True
    last_cursor_toggle = time.time()

    for new_text in streamer:
        partial_word += new_text
        # Toggle the cursor visibility every 0.5 seconds
        if time.time() - last_cursor_toggle > 0.5:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()
        current_cursor = cursor if cursor_visible else " "
        if partial_word.endswith(" ") or partial_word.endswith("\n"):
            generated_text += partial_word
            # Yield the current text with the cursor appended
            yield generated_text + current_cursor
            partial_word = ""
        else:
            # Yield the current text plus the partial word and the cursor
            yield generated_text + partial_word + current_cursor

    # Handle any remaining partial word at the end
    if partial_word:
        generated_text += partial_word
        yield generated_text

    # Save the interaction after generation is complete
    save_interaction(video_path, prompt, generated_text, audio_track)


head = """
<head>
    <title>EgoGPT Demo - EgoLife</title>
    <link rel="icon" type="image/x-icon" href="./egolife_circle.ico">
</head>
<style>
/* Submit按钮默认和悬停效果 */
button.lg.secondary.svelte-5st68j {
    background-color: #ff9933 !important;
    transition: background-color 0.3s ease !important;
}

button.lg.secondary.svelte-5st68j:hover {
    background-color: #ff7777 !important;  /* 悬停时颜色加深 */
}

/* 确保按钮文字始终清晰可见 */
button.lg.secondary.svelte-5st68j span {
    color: white !important;
}

/* 隐藏表头中的第二列 */
.table-wrap .svelte-p5q82i th:nth-child(2) {
    display: none;
}

/* 隐藏表格内容中的第二列 */
.table-wrap .svelte-p5q82i td:nth-child(2) {
    display: none;
}

.table-wrap {
    max-height: 300px;
    overflow-y: auto;
}

</style>

<script>
function initializeControls() {
    const video = document.querySelector('[data-testid="Video-player"]');
    const waveform = document.getElementById('waveform');
    
    // 如果元素还没准备好，直接返回
    if (!video || !waveform) {
        return;
    }
    
    // 尝试获取音频元素
    const audio = waveform.querySelector('div')?.shadowRoot?.querySelector('audio');
    if (!audio) {
        return;
    }

    console.log('Elements found:', { video, audio });
    
   // 监听视频播放进度
  video.addEventListener("play", () => {
    if (audio.paused) {
      audio.play();  // 如果音频暂停，开始播放
    }
  });

  // 监听音频播放进度
  audio.addEventListener("play", () => {
    if (video.paused) {
      video.play();  // 如果视频暂停，开始播放
    }
  });

  // 同步视频和音频的播放进度
  video.addEventListener("timeupdate", () => {
    if (Math.abs(video.currentTime - audio.currentTime) > 0.1) {
      audio.currentTime = video.currentTime; // 如果时间差超过0.1秒，同步
    }
  });

  audio.addEventListener("timeupdate", () => {
    if (Math.abs(audio.currentTime - video.currentTime) > 0.1) {
      video.currentTime = audio.currentTime; // 如果时间差超过0.1秒，同步
    }
  });

  // 监听暂停事件，确保视频和音频都暂停
  video.addEventListener("pause", () => {
    if (!audio.paused) {
      audio.pause();  // 如果音频未暂停，暂停音频
    }
  });

  audio.addEventListener("pause", () => {
    if (!video.paused) {
      video.pause();  // 如果视频未暂停，暂停视频
    }
  });
}

// 创建观察器监听DOM变化
const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.addedNodes.length) {
            // 当有新节点添加时，尝试初始化
            const waveform = document.getElementById('waveform');
            if (waveform?.querySelector('div')?.shadowRoot?.querySelector('audio')) {
                console.log('Audio element detected');
                initializeControls();
                // 可选：如果不需要继续监听，可以断开观察器
                // observer.disconnect();
            }
        }
    }
});

// 开始观察
observer.observe(document.body, {
    childList: true,
    subtree: true
});

// 页面加载完成时也尝试初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    initializeControls();

    // Ensure title and favicon are set correctly
    document.title = "EgoGPT Demo - EgoLife";
    
    // Create/update favicon link
    let link = document.querySelector("link[rel~='icon']");
    if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
    }
    link.href = './egolife_circle.ico';

});

</script>
"""

with gr.Blocks(title="EgoGPT Demo - EgoLife", head=head) as demo:
    gr.Markdown(title_markdown)
    gr.Markdown(notice_html)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Video",
                autoplay=True,
                loop=True,
                format="mp4",
                width=600,
                height=400,
                show_label=False,
                elem_id="video",
            )
            # Make audio display conditionally visible
            audio_display = gr.Audio(
                label="Video Audio Track",
                autoplay=False,
                show_label=True,
                visible=True,
                interactive=False,
                elem_id="audio",
            )
            text_input = gr.Textbox(
                label="Question",
                placeholder="Enter your message here...",
                value="Describe everything I saw, did, and heard, using the first perspective. Transcribe all the speech.",
            )

        with gr.Column():
            output_text = gr.Textbox(label="Response", lines=14, max_lines=14)
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/videos/cheers.mp4",
                        f"{cur_dir}/videos/cheers.mp3",
                        "Describe everything I saw, did, and heard from the first perspective.",
                    ],
                    [
                        f"{cur_dir}/videos/DAY3_A6_SHURE_14550000.mp4",
                        f"{cur_dir}/videos/DAY3_A6_SHURE_14550000.mp3",
                        "请按照时间顺序描述我所见所为，并转录所有声音。",
                    ],
                    [
                        f"{cur_dir}/videos/shopping.mp4",
                        f"{cur_dir}/videos/shopping.mp3",
                        "Please only transcribe all the speech.",
                    ],
                    [
                        f"{cur_dir}/videos/japan.mp4",
                        f"{cur_dir}/videos/japan.mp3",
                        "Describe everything I see, do, and hear from the first-person view.",
                    ],
                ],
                inputs=[video_input, audio_display, text_input],
                outputs=[output_text],
            )

    def handle_video_change(video):
        if video is None:
            return gr.update(visible=False), None

        audio = extract_audio_from_video(video)
        # Update audio display visibility based on whether we have audio
        return gr.update(visible=audio is not None), audio

    # Update the video input change event
    video_input.change(
        fn=handle_video_change,
        inputs=[video_input],
        outputs=[
            audio_display,
            audio_display,
        ],  # First for visibility, second for audio data
    )

    # Add clear handler
    def clear_outputs(video):
        if video is None:
            return gr.update(visible=False), "", None
        return gr.skip()

    video_input.clear(
        fn=clear_outputs,
        inputs=[video_input],
        outputs=[audio_display, output_text, audio_display],
    )

    text_input.submit(
        fn=generate_text,
        inputs=[video_input, audio_display, text_input],
        outputs=[output_text],
        api_name="generate_streaming",
    )

    # Add submit button and its event handler
    submit_btn = gr.Button("Submit")
    submit_btn.click(
        fn=generate_text,
        inputs=[video_input, audio_display, text_input],
        outputs=[output_text],
        api_name="generate_streaming",
    )

# Launch the Gradio app
# demo.launch(share=True)
demo.launch(server_name="127.0.0.1", server_port=8081)
