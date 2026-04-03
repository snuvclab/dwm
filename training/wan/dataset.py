import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]
MULTIVIEW_EXT_PATTERN = re.compile(r"^(?P<base>.+)__ext(?P<view>[0-9]+)$")


def _normalize_exclude_videos_files(exclude_videos_file: Optional[Any]) -> List[str]:
    if not exclude_videos_file:
        return []
    if isinstance(exclude_videos_file, (str, Path)):
        return [str(exclude_videos_file)]
    try:
        return [str(path) for path in exclude_videos_file if path]
    except TypeError:
        return [str(exclude_videos_file)]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
        prompt_subdir: str = "prompts",
        prompt_embeds_subdir: str = "prompt_embeds_rewrite_wan",
        static_video_subdir: str = "videos_static",
        hand_video_subdir: str = "videos_hands",
        hand_video_latents_subdir: str = "hand_video_latents_wan",
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        image_latents_subdir: str = "image_latents_wan",
        fun_inp_i2v_latents_subdir: str = "fun_inp_i2v_latents_wan",
        clip_image_embeds_subdir: str = "clip_image_embeds",
        exclude_videos_file: Optional[Any] = None,
        align_width_to_32: bool = False,
        max_prompt_length: int = 512,
    ) -> None:
        super().__init__()
        self.align_width_to_32 = align_width_to_32
        self.max_prompt_length = max_prompt_length

        self.data_root = Path(data_root)
        # Handle dataset_file as list or single path
        if dataset_file is None:
            self.dataset_file_list: Optional[List[str]] = None
        elif isinstance(dataset_file, (list, tuple)):
            self.dataset_file_list = [str(path) for path in dataset_file]
        else:
            self.dataset_file_list = [str(dataset_file)]
        self.dataset_file = dataset_file
        self.caption_column = caption_column  # not used
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video
        self.prompt_subdir = prompt_subdir
        self.prompt_embeds_subdir = prompt_embeds_subdir
        self.static_video_subdir = static_video_subdir
        self.hand_video_subdir = hand_video_subdir
        self.hand_video_latents_subdir = hand_video_latents_subdir
        self.video_latents_subdir = video_latents_subdir
        self.static_video_latents_subdir = static_video_latents_subdir
        self.image_latents_subdir = image_latents_subdir
        self.fun_inp_i2v_latents_subdir = fun_inp_i2v_latents_subdir
        self.clip_image_embeds_subdir = clip_image_embeds_subdir
        self.exclude_videos_file = exclude_videos_file
        self.excluded_videos = self._load_excluded_video_entries(exclude_videos_file)
        
        # Ensure buckets are lists
        if not isinstance(self.height_buckets, list):
            self.height_buckets = [self.height_buckets]
        if not isinstance(self.width_buckets, list):
            self.width_buckets = [self.width_buckets]
        if not isinstance(self.frame_buckets, list):
            self.frame_buckets = [self.frame_buckets]
        
        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        # Two methods of loading data are supported.
        #   - Using a CSV: caption_column and video_column must be some column in the CSV. One could
        #     make use of other columns too, such as a motion score or aesthetic score, by modifying the
        #     logic in CSV processing.
        #   - Using two files containing line-separate captions and relative paths to videos.
        # For a more detailed explanation about preparing dataset format, checkout the README.
        if self.dataset_file_list is None:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_local_path()
        else:
            (
                self.prompts,
                self.video_paths,
            ) = self._load_dataset_from_datafile()
            # ) = self._load_dataset_from_csv()

        if self.excluded_videos:
            self.prompts, self.video_paths = self._apply_exclude_video_filter(self.prompts, self.video_paths)

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def _find_nearest_resolution(self, height: int, width: int) -> Tuple[int, int]:
        """Pick the nearest (H, W) resolution bucket, optionally aligning W to a multiple of 32.

        Note: alignment is only meaningful when a subclass actually resizes frames to the chosen bucket.
        """
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        nearest_h, nearest_w = nearest_res[1], nearest_res[2]

        # For WAN 2.2: align width to 32 for compatibility
        # 32 = 8 (VAE spatial_compression) * 4, ensures clean division through the pipeline
        if getattr(self, "align_width_to_32", False):
            # Always round to nearest multiple of 32, regardless of whether it's in buckets
            # This ensures WAN 2.2 compatibility even if the aligned value is not in the bucket list
            nearest_w = round(nearest_w / 32) * 32

        return nearest_h, nearest_w

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_excluded_video_entries(self, exclude_videos_file: Optional[Any]) -> set[str]:
        exclude_paths_raw = _normalize_exclude_videos_files(exclude_videos_file)
        if not exclude_paths_raw:
            return set()

        entries: set[str] = set()
        for exclude_path_raw in exclude_paths_raw:
            exclude_path = Path(exclude_path_raw)
            if not exclude_path.is_absolute():
                exclude_path = self.data_root / exclude_path
            if not exclude_path.exists():
                raise FileNotFoundError(f"Exclude videos file not found: {exclude_path}")

            path_entries = {
                line.strip()
                for line in exclude_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            entries.update(path_entries)
            logger.info("Loaded %d excluded videos from %s", len(path_entries), exclude_path)

        logger.info("Loaded %d unique excluded videos from %d file(s)", len(entries), len(exclude_paths_raw))
        return entries

    def _video_path_to_rel_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.data_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _get_candidate_action_dirs(self, path: Path) -> List[Path]:
        action_dir = path.parent.parent
        candidate_dirs = [action_dir]
        if action_dir.name == "processed2":
            candidate_dirs.append(action_dir.parent)
        return candidate_dirs

    def _resolve_existing_subdir(self, path: Path, subdir: str) -> Path:
        candidate_dirs = [action_dir / subdir for action_dir in self._get_candidate_action_dirs(path)]
        for candidate_dir in candidate_dirs:
            if candidate_dir.exists():
                return candidate_dir
        return candidate_dirs[0]

    def _resolve_existing_file(self, path: Path, subdir: str, filename: str) -> Path:
        candidate_files = [
            action_dir / subdir / filename for action_dir in self._get_candidate_action_dirs(path)
        ]
        for candidate_file in candidate_files:
            if candidate_file.exists():
                return candidate_file
        return candidate_files[0]

    def _apply_exclude_video_filter(
        self, prompts: List[str], video_paths: List[Path]
    ) -> Tuple[List[str], List[Path]]:
        filtered_prompts: List[str] = []
        filtered_video_paths: List[Path] = []
        excluded_count = 0

        for prompt, video_path in zip(prompts, video_paths):
            rel_key = self._video_path_to_rel_key(video_path)
            if rel_key in self.excluded_videos:
                excluded_count += 1
                continue
            filtered_prompts.append(prompt)
            filtered_video_paths.append(video_path)

        if excluded_count:
            logger.warning(
                "Excluded %d videos listed in %s from the dataset.",
                excluded_count,
                self.exclude_videos_file,
            )
        if not filtered_video_paths:
            raise ValueError("All dataset videos were excluded. Check exclude_videos_file.")

        return filtered_prompts, filtered_video_paths

    def _build_raw_clip_plan(self, path: Path) -> Dict[str, int]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        return self._build_raw_clip_plan_from_reader(video_reader)

    def _build_raw_clip_plan_from_reader(self, video_reader) -> Dict[str, int]:
        video_num_frames = len(video_reader)
        if video_num_frames <= 0:
            raise ValueError("Video has no frames")

        first_frame = video_reader[0]
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        target_height, target_width = self._find_nearest_resolution(height, width)
        target_num_frames = int(self.max_num_frames)
        max_start = max(video_num_frames - target_num_frames, 0)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0

        return {
            "start_idx": start_idx,
            "target_num_frames": target_num_frames,
            "target_height": target_height,
            "target_width": target_width,
        }

    def _resolve_raw_clip_plan(self, video_reader, clip_plan: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        if clip_plan is None:
            return self._build_raw_clip_plan_from_reader(video_reader)

        resolved_plan = dict(clip_plan)
        target_num_frames = int(resolved_plan.get("target_num_frames", self.max_num_frames))
        video_num_frames = len(video_reader)
        max_start = max(video_num_frames - target_num_frames, 0)
        resolved_plan["target_num_frames"] = target_num_frames
        resolved_plan["start_idx"] = min(int(resolved_plan.get("start_idx", 0)), max_start)
        return resolved_plan

    def _load_raw_frames(self, video_reader, clip_plan: Dict[str, int]) -> torch.Tensor:
        video_num_frames = len(video_reader)
        target_num_frames = int(clip_plan["target_num_frames"])
        start_idx = int(clip_plan["start_idx"])
        end_idx = min(video_num_frames, start_idx + target_num_frames)
        frame_indices = list(range(start_idx, end_idx))

        if not frame_indices:
            raise ValueError("No frame indices selected for video clip")

        frames = video_reader.get_batch(frame_indices).float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        if frames.shape[0] < target_num_frames:
            pad_frames = frames[-1:].repeat(target_num_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)

        return frames

    def _get_raw_video_sample(self, index: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
        clip_plan = self._build_raw_clip_plan(self.video_paths[index])
        image, video, _ = self._preprocess_video(self.video_paths[index], clip_plan=clip_plan)

        sample = {
            "prompt": self.id_token + self.prompts[index],
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }
        return sample, clip_plan

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     # Here, index is actually a list of data objects that we need to return.
        #     # The BucketSampler should ideally return indices. But, in the sampler, we'd like
        #     # to have information about num_frames, height and width. Since this is not stored
        #     # as metadata, we need to read the video to get this information. You could read this
        #     # information without loading the full video in memory, but we do it anyway. In order
        #     # to not load the video twice (once to get the metadata, and once to return the loaded video
        #     # based on sampled indices), we cache it in the BucketSampler. When the sampler is
        #     # to yield, we yield the cache data instead of indices. So, this special check ensures
        #     # that data is not loaded a second time. PRs are welcome for improvements.
        #     return index

        # Handle index out of range with mod operation to prevent training interruption
        original_index = index
        if index >= len(self.video_paths):
            logger.warning(f"Index {index} is out of range (dataset size: {len(self.video_paths)}). Using mod operation: {index} % {len(self.video_paths)} = {index % len(self.video_paths)}")
            index = index % len(self.video_paths)

        # try:
        if self.load_tensors:
            first_frame_latents, fun_inp_i2v_latents, video_latents, prompt_embeds, clip_image_embeds = self._load_preprocessed_latents_and_embeds(
                self.video_paths[index]
            )

            # This is hardcoded for now.
            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": first_frame_latents,
                "fun_inp_i2v_latents": fun_inp_i2v_latents,
                "clip_image_embeds": clip_image_embeds,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            sample, _ = self._get_raw_video_sample(index)
            return sample
        # except Exception as e:
        #     logger.error(f"Failed to load main video data for index {index}: {e}")
        #     # Return a minimal valid data structure to prevent training from crashing
        #     return {
        #         "prompt": "ERROR_LOADING_DATA",
        #         "image": None,
        #         "video": torch.zeros(1, 3, 16, 256, 256),  # Dummy video tensor
        #         "video_metadata": {
        #             "num_frames": 16,
        #             "height": 256,
        #             "width": 256,
        #         },
        #     }

    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        video_path = self.data_root.joinpath(self.video_column)

        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )

        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        # Derive prompt paths from video paths (same as _load_dataset_from_datafile)
        prompt_paths = [path.parent.parent.joinpath("prompts", path.name.replace(".mp4", ".txt")) for path in video_paths]
        # prompts = [path.read_text().strip() for path in prompt_paths]
        prompts = ["" for path in prompt_paths]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths

    def _load_prompt_texts(self, prompt_paths: List[Path], dataset_path: Path) -> List[str]:
        prompts: List[str] = []
        missing_count = 0
        for prompt_path in prompt_paths:
            if prompt_path.is_file():
                prompts.append(prompt_path.read_text(encoding="utf-8"))
            else:
                prompts.append("")
                missing_count += 1
        if missing_count:
            logger.warning(
                "Prompt files missing for %d samples while loading %s. Using empty prompts.",
                missing_count,
                dataset_path,
            )
        return prompts

    def _load_dataset_from_datafile(self) -> Tuple[List[str], List[Path]]:
        """Load dataset from one or more datafiles (supports list of files)."""
        if not self.dataset_file_list:
            raise ValueError("dataset_file should not be empty when provided.")
        
        all_prompts: List[str] = []
        all_video_paths: List[Path] = []
        
        for dataset_file in self.dataset_file_list:
            dataset_path = self.data_root / dataset_file
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            with open(dataset_path, "r", encoding="utf-8") as f:
                file_video_paths = [
                    self.data_root.joinpath(line.strip()) for line in f.readlines() if len(line.strip()) > 0
                ]
            
            prompt_paths = [
                self._resolve_existing_file(path, self.prompt_subdir, path.name.replace(".mp4", ".txt"))
                for path in file_video_paths
            ]
            file_prompts = self._load_prompt_texts(prompt_paths, dataset_path)

            all_video_paths.extend(file_video_paths)
            all_prompts.extend(file_prompts)
        
        return all_prompts, all_video_paths

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single raw video and returns (image, video, prompt_embed=None).

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.
        """
        try:
            video_reader = decord.VideoReader(uri=path.as_posix())
            resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
            frames = self._load_raw_frames(video_reader, resolved_clip_plan)
            frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None
        except Exception as e:
            logger.error(f"Failed to preprocess video {path}: {e}")
            raise e

    def _load_preprocessed_latents_and_embeds(
        self, path: Path
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        try:
            filename_without_ext = path.name.split(".")[0]
            pt_filename = f"{filename_without_ext}.pt"

            # The current path is something like: /a/b/c/d/videos/00001.mp4
            # We need to reach: /a/b/c/d/video_latents_wan/00001.pt (using configurable subdir)
            video_latents_path = self._resolve_existing_subdir(path, self.video_latents_subdir)
            embeds_path = self._resolve_existing_subdir(path, self.prompt_embeds_subdir)
            image_latents_path = self._resolve_existing_subdir(path, self.image_latents_subdir)
            fun_inp_i2v_latents_path = self._resolve_existing_subdir(path, self.fun_inp_i2v_latents_subdir)
            clip_embeds_path = self._resolve_existing_subdir(path, self.clip_image_embeds_subdir)

            if not video_latents_path.exists() or not embeds_path.exists():
                logger.error(f"Required folders not found for {path}")
                logger.error(f"video_latents_path exists: {video_latents_path.exists()}")
                logger.error(f"embeds_path exists: {embeds_path.exists()}")
                raise ValueError(
                    f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`."
                )

            video_latent_filepath = video_latents_path.joinpath(pt_filename)
            embeds_filepath = embeds_path.joinpath(pt_filename)

            if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
                video_latent_filepath = video_latent_filepath.as_posix()
                embeds_filepath = embeds_filepath.as_posix()
                logger.error(f"Required files not found:")
                logger.error(f"video_latent_filepath exists: {Path(video_latent_filepath).exists()}")
                logger.error(f"embeds_filepath exists: {Path(embeds_filepath).exists()}")
                raise ValueError(
                    f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
                )

            try:
                video_latent_size = Path(video_latent_filepath).stat().st_size
                embeds_size = Path(embeds_filepath).stat().st_size
                if video_latent_size == 0 or embeds_size == 0:
                    raise EOFError(
                        f"Zero-byte tensor file detected: "
                        f"video_latent_size={video_latent_size}, embeds_size={embeds_size}"
                    )

                latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
                embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)
                images = latents[:, :1, :, :].clone() if self.image_to_video else None
                fun_inp_i2v_latents = None
                clip_image_embeds = None

                if self.image_to_video:
                    image_latent_filepath = image_latents_path.joinpath(pt_filename)
                    fun_inp_i2v_latent_filepath = fun_inp_i2v_latents_path.joinpath(pt_filename)
                    clip_embed_filepath = clip_embeds_path.joinpath(pt_filename)

                    if image_latent_filepath.is_file():
                        images = torch.load(image_latent_filepath, map_location="cpu", weights_only=True)
                    if fun_inp_i2v_latent_filepath.is_file():
                        fun_inp_i2v_latents = torch.load(fun_inp_i2v_latent_filepath, map_location="cpu", weights_only=True)
                    if clip_embed_filepath.is_file():
                        clip_image_embeds = torch.load(clip_embed_filepath, map_location="cpu", weights_only=True)

                # Pad prompt embeddings to max_prompt_length for uniform batch sizes
                # embeds shape: [seq_len, dim] -> pad to [max_prompt_length, dim]
                if embeds.shape[0] < self.max_prompt_length:
                    padding = torch.zeros(
                        self.max_prompt_length - embeds.shape[0],
                        embeds.shape[1],
                        dtype=embeds.dtype,
                    )
                    embeds = torch.cat([embeds, padding], dim=0)
                elif embeds.shape[0] > self.max_prompt_length:
                    # Truncate if longer than max_prompt_length
                    embeds = embeds[:self.max_prompt_length]

            except Exception as e:
                logger.error(
                    f"Failed to load torch files for {path}: "
                    f"{type(e).__name__}: {repr(e)}"
                )
                logger.error(f"video_latent_filepath: {video_latent_filepath}")
                logger.error(f"embeds_filepath: {embeds_filepath}")
                raise e

            return images, fun_inp_i2v_latents, latents, embeds, clip_image_embeds
            
        except Exception as e:
            logger.error(
                f"Error in _load_preprocessed_latents_and_embeds for {path}: "
                f"{type(e).__name__}: {repr(e)}"
            )
            raise e


class VideoDatasetWithConditions(VideoDataset):
    """
    Extended VideoDataset that supports additional condition videos:
    - hand_videos: Egocentric hand mesh videos (automatically derived from main video paths)
    - static_videos: Static scene videos (automatically derived from main video paths)
    """

    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
        prompt_subdir: str = "prompts",
        prompt_embeds_subdir: str = "prompt_embeds_rewrite_wan",
        static_video_subdir: str = "videos_static",
        static_video_source_mode: str = "directory",
        hand_video_subdir: str = "videos_hands",
        hand_video_latents_subdir: str = "hand_video_latents_wan",
        static_disparity_subdir: Optional[str] = None,
        static_disparity_latents_subdir: Optional[str] = None,
        hand_disparity_subdir: Optional[str] = None,
        hand_disparity_latents_subdir: Optional[str] = None,
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        image_latents_subdir: str = "image_latents_wan",
        fun_inp_i2v_latents_subdir: str = "fun_inp_i2v_latents_wan",
        clip_image_embeds_subdir: str = "clip_image_embeds",
        exclude_videos_file: Optional[Any] = None,
        require_static_videos: bool = True,
        align_width_to_32: bool = False,
        max_prompt_length: int = 512,
    ) -> None:
        # Initialize parent class with main video column
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
            max_prompt_length=max_prompt_length,
            prompt_subdir=prompt_subdir,
            prompt_embeds_subdir=prompt_embeds_subdir,
            static_video_subdir=static_video_subdir,
            hand_video_subdir=hand_video_subdir,
            hand_video_latents_subdir=hand_video_latents_subdir,
            video_latents_subdir=video_latents_subdir,
            static_video_latents_subdir=static_video_latents_subdir,
            image_latents_subdir=image_latents_subdir,
            fun_inp_i2v_latents_subdir=fun_inp_i2v_latents_subdir,
            clip_image_embeds_subdir=clip_image_embeds_subdir,
            exclude_videos_file=exclude_videos_file,
            align_width_to_32=align_width_to_32,
        )

        # Store the use_gray_hand_videos flag
        self.use_gray_hand_videos = use_gray_hand_videos
        self.static_disparity_subdir = static_disparity_subdir
        self.static_disparity_latents_subdir = static_disparity_latents_subdir
        self.hand_disparity_subdir = hand_disparity_subdir
        self.hand_disparity_latents_subdir = hand_disparity_latents_subdir
        self.static_video_source_mode = str(static_video_source_mode).strip().lower()
        self.require_static_videos = bool(require_static_videos)
        if self.static_video_source_mode not in {"directory", "copy_first_frame"}:
            raise ValueError(
                f"Unsupported static_video_source_mode: {static_video_source_mode}. "
                "Use one of: directory, copy_first_frame."
            )
        # Store the align_width_to_32 flag (for WAN 2.2 compatibility)
        self.align_width_to_32 = align_width_to_32
        
        # Automatically derive hand video and static video paths from main video paths
        if self.use_gray_hand_videos:
            self.hand_video_paths = self._derive_condition_video_paths("videos_hands_gray")
        else:
            self.hand_video_paths = (
                self._derive_condition_video_paths(self.hand_video_subdir)
                if self.hand_video_subdir
                else None
            )
        self.static_video_paths = self._derive_condition_video_paths(self.static_video_subdir)
        self.static_disparity_paths = (
            self._derive_condition_video_paths(self.static_disparity_subdir)
            if self.static_disparity_subdir
            else None
        )
        self.hand_disparity_paths = (
            self._derive_condition_video_paths(self.hand_disparity_subdir)
            if self.hand_disparity_subdir
            else None
        )
        # Validate that all condition videos exist
        if not self.load_tensors:
            if self.hand_video_paths is not None and any(not path.is_file() for path in self.hand_video_paths):
                missing_hand_videos = [path for path in self.hand_video_paths if not path.is_file()]
                raise ValueError(
                    f"Some hand video files are missing. First few missing files: {missing_hand_videos[:5]}"
                )
            
            # if any(not path.is_file() for path in self.hand_video_gray_paths):
            #     missing_hand_gray_videos = [path for path in self.hand_video_gray_paths if not path.is_file()]
            #     raise ValueError(
            #         f"Some hand gray video files are missing. First few missing files: {missing_hand_gray_videos[:5]}"
            #     )
            
            if self.require_static_videos and self.static_video_source_mode == "directory":
                missing_static_videos = [path for path in self.static_video_paths if not path.is_file()]
                if missing_static_videos:
                    raise ValueError(
                        f"Some static video files are missing. First few missing files: {missing_static_videos[:5]}"
                    )
            if self.static_disparity_paths is not None and any(
                not path.is_file() for path in self.static_disparity_paths
            ):
                missing_static_disparities = [path for path in self.static_disparity_paths if not path.is_file()]
                raise ValueError(
                    "Some static disparity video files are missing. "
                    f"First few missing files: {missing_static_disparities[:5]}"
                )
            if self.hand_disparity_paths is not None and any(
                not path.is_file() for path in self.hand_disparity_paths
            ):
                missing_hand_disparities = [path for path in self.hand_disparity_paths if not path.is_file()]
                raise ValueError(
                    "Some hand disparity video files are missing. "
                    f"First few missing files: {missing_hand_disparities[:5]}"
                )

    def _derive_condition_video_paths(self, condition_folder: str) -> List[Path]:
        """Derive condition video paths from main video paths.
        
        Args:
            condition_folder: Name of the condition folder (e.g., "videos_hands", "videos_static")
            
        Returns:
            List of condition video paths
        """
        condition_paths = []
        
        for video_path in self.video_paths:
            # Example: video_path = /path/to/sequences/videos/00001.mp4
            # We want: /path/to/sequences/videos_hands/00001.mp4
            # or: /path/to/sequences/videos_static/00001.mp4
            
            parent_dir = video_path.parent
            filename = video_path.name
            condition_path = parent_dir.parent / condition_folder / filename
            if not condition_path.exists() and parent_dir.parent.name == "processed2":
                fallback_condition_path = parent_dir.parent.parent / condition_folder / filename
                if fallback_condition_path.exists():
                    condition_path = fallback_condition_path
            condition_paths.append(condition_path)
        
        return condition_paths

    def _get_item_once(self, index: int) -> Dict[str, Any]:
        if self.load_tensors:
            # Get main video data from parent class
            main_data = super().__getitem__(index)

            # Load condition videos
            # Load preprocessed latents for condition videos (using configurable subdirs)
            try:
                if self.hand_video_paths is None:
                    main_data["hand_videos"] = None
                elif self.use_gray_hand_videos:
                    # Gray hand videos use a different subdir pattern
                    hand_latents_subdir = self.hand_video_latents_subdir.replace("hand_video", "hand_video_gray")
                    hand_video_latents = self._load_condition_video_latents(self.hand_video_paths[index], hand_latents_subdir)
                    main_data["hand_videos"] = hand_video_latents
                else:
                    hand_video_latents = self._load_condition_video_latents(self.hand_video_paths[index], self.hand_video_latents_subdir)
                    main_data["hand_videos"] = hand_video_latents
            except Exception as e:
                logger.warning(f"Failed to load hand video latents for index {index}: {e}")
                main_data["hand_videos"] = None

            if self.require_static_videos:
                try:
                    static_video_latents = self._load_condition_video_latents(self.static_video_paths[index], self.static_video_latents_subdir)
                    main_data["static_videos"] = static_video_latents
                except Exception as e:
                    logger.warning(f"Failed to load static video latents for index {index}: {e}")
                    main_data["static_videos"] = None
            else:
                main_data["static_videos"] = None

            if self.static_disparity_paths is not None and self.static_disparity_latents_subdir is not None:
                try:
                    static_disparity_latents = self._load_condition_video_latents(
                        self.static_disparity_paths[index], self.static_disparity_latents_subdir
                    )
                    main_data["static_disparities"] = static_disparity_latents
                except Exception as e:
                    logger.warning(f"Failed to load static disparity latents for index {index}: {e}")
                    main_data["static_disparities"] = None
            else:
                main_data["static_disparities"] = None

            if self.hand_disparity_paths is not None and self.hand_disparity_latents_subdir is not None:
                try:
                    hand_disparity_latents = self._load_condition_video_latents(
                        self.hand_disparity_paths[index], self.hand_disparity_latents_subdir
                    )
                    main_data["hand_disparities"] = hand_disparity_latents
                except Exception as e:
                    logger.warning(f"Failed to load hand disparity latents for index {index}: {e}")
                    main_data["hand_disparities"] = None
            else:
                main_data["hand_disparities"] = None
        else:
            main_data, clip_plan = self._get_raw_video_sample(index)

            # Load raw videos for condition videos
            try:
                if self.hand_video_paths is None:
                    main_data["hand_videos"] = None
                else:
                    _, hand_video, _ = self._preprocess_video(self.hand_video_paths[index], clip_plan=clip_plan)
                    main_data["hand_videos"] = hand_video
            except Exception as e:
                logger.warning(f"Failed to load hand video for index {index}: {e}")
                main_data["hand_videos"] = None
            
            if not self.require_static_videos:
                main_data["static_videos"] = None
            elif self.static_video_source_mode == "copy_first_frame":
                main_video = main_data["video"]
                main_data["static_videos"] = main_video[0:1].repeat(main_video.shape[0], 1, 1, 1).clone()
            else:
                try:
                    _, static_video, _ = self._preprocess_video(self.static_video_paths[index], clip_plan=clip_plan)
                    main_data["static_videos"] = static_video
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load static video for index {index}: {self.static_video_paths[index]}"
                    ) from e

            if self.static_disparity_paths is not None:
                try:
                    _, static_disparity, _ = self._preprocess_video(
                        self.static_disparity_paths[index], clip_plan=clip_plan
                    )
                    main_data["static_disparities"] = static_disparity
                except Exception as e:
                    logger.warning(f"Failed to load static disparity video for index {index}: {e}")
                    main_data["static_disparities"] = None
            else:
                main_data["static_disparities"] = None

            if self.hand_disparity_paths is not None:
                try:
                    _, hand_disparity, _ = self._preprocess_video(
                        self.hand_disparity_paths[index], clip_plan=clip_plan
                    )
                    main_data["hand_disparities"] = hand_disparity
                except Exception as e:
                    logger.warning(f"Failed to load hand disparity video for index {index}: {e}")
                    main_data["hand_disparities"] = None
            else:
                main_data["hand_disparities"] = None

        return main_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     return index

        if index >= len(self.video_paths):
            logger.warning(
                "Index %d is out of range (dataset size: %d). Using mod operation: %d %% %d = %d",
                index,
                len(self.video_paths),
                index,
                len(self.video_paths),
                index % len(self.video_paths),
            )
            index = index % len(self.video_paths)

        return self._get_item_once(index)

    def _load_condition_video_latents(self, path: Path, latent_folder: str) -> Optional[torch.Tensor]:
        """Load preprocessed latents for condition videos.
        
        Returns:
            torch.Tensor if latents are successfully loaded, None if folder or file doesn't exist.
        """
        try:
            filename_without_ext = path.name.split(".")[0]
            pt_filename = f"{filename_without_ext}.pt"
            
            # The current path is something like: /a/b/c/sequences/videos_hands/00001.mp4
            # We need to reach: /a/b/c/hand_video_latents/00001.pt
            # or: /a/b/c/static_video_latents/00001.pt
            
            # Get the action directory (parent of the condition folder)
            # Example: .../SingleHand/Kitchen_xxx/videos_hands/00001.mp4
            # -> .../SingleHand/Kitchen_xxx
            latents_path = self._resolve_existing_subdir(path, latent_folder)
            
            if not latents_path.exists():
                logger.warning(f"Latents folder not found: {latents_path}. Returning None.")
                return None
            
            latent_filepath = latents_path.joinpath(pt_filename)
            
            if not latent_filepath.is_file():
                logger.warning(f"Latent file not found: {latent_filepath}. Returning None.")
                return None
            
            try:
                latents = torch.load(latent_filepath, map_location="cpu", weights_only=True)
                return latents
            except Exception as e:
                logger.error(f"Failed to load latent file {latent_filepath}: {e}")
                raise e
                
        except Exception as e:
            # Only log and re-raise if it's not a FileNotFoundError (which we already handled above)
            if not isinstance(e, FileNotFoundError):
                logger.error(f"Error in _load_condition_video_latents for {path} in {latent_folder}: {e}")
                raise e
            # If it's a FileNotFoundError that we didn't catch above, return None
            logger.warning(f"Latents not found for {path} in {latent_folder}. Returning None.")
            return None


class VideoDatasetWithConditionsAndResizing(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that also supports resizing.
    """
    
    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class MultiViewVideoDatasetWithConditionsAndResizing(VideoDatasetWithConditionsAndResizing):
    """
    Raw-video multiview dataset that groups files like `{base}__ext1.mp4` and
    `{base}__ext2.mp4` into one sample while leaving the existing single-view
    dataset classes unchanged.
    """

    def __init__(
        self,
        *,
        expected_view_ids: Optional[List[int]] = None,
        grouping_mode: str = "ext_suffix",
        **kwargs,
    ) -> None:
        load_tensors = bool(kwargs.get("load_tensors", False))
        if load_tensors:
            raise NotImplementedError(
                "MultiViewVideoDatasetWithConditionsAndResizing currently supports raw videos only "
                "(load_tensors=False)."
            )

        static_video_source_mode = str(kwargs.pop("static_video_source_mode", "directory")).strip().lower()
        static_disparity_subdir = kwargs.pop("static_disparity_subdir", None)
        static_disparity_latents_subdir = kwargs.pop("static_disparity_latents_subdir", None)
        hand_disparity_subdir = kwargs.pop("hand_disparity_subdir", None)
        hand_disparity_latents_subdir = kwargs.pop("hand_disparity_latents_subdir", None)
        require_static_videos = bool(kwargs.pop("require_static_videos", True))

        # Intentionally bypass VideoDatasetWithConditions.__init__ so the new MV
        # path does not inherit flat-path condition validation behavior.
        VideoDataset.__init__(self, **kwargs)

        self.static_disparity_subdir = static_disparity_subdir
        self.static_disparity_latents_subdir = static_disparity_latents_subdir
        self.hand_disparity_subdir = hand_disparity_subdir
        self.hand_disparity_latents_subdir = hand_disparity_latents_subdir
        self.static_video_source_mode = static_video_source_mode
        self.require_static_videos = require_static_videos
        if self.static_video_source_mode not in {"directory", "copy_first_frame"}:
            raise ValueError(
                f"Unsupported static_video_source_mode: {static_video_source_mode}. "
                "Use one of: directory, copy_first_frame."
            )

        self.grouping_mode = str(grouping_mode).strip().lower()
        if self.grouping_mode != "ext_suffix":
            raise ValueError(
                f"Unsupported grouping_mode: {grouping_mode}. "
                "Use one of: ext_suffix."
            )

        self.expected_view_ids = [int(view_id) for view_id in (expected_view_ids or [1, 2])]
        if not self.expected_view_ids:
            raise ValueError("expected_view_ids must not be empty.")

        self.sample_groups = self._build_multiview_groups(self.video_paths, self.prompts)
        if not self.sample_groups:
            raise ValueError(
                "No valid multiview groups were found. Check dataset_file contents and expected_view_ids."
            )

    def __len__(self) -> int:
        return len(self.sample_groups)

    def _parse_multiview_path(self, path: Path) -> Optional[Tuple[str, int]]:
        match = MULTIVIEW_EXT_PATTERN.match(path.stem)
        if match is None:
            return None
        base_name = match.group("base")
        try:
            rel_parent = path.relative_to(self.data_root).parent.as_posix()
        except ValueError:
            rel_parent = path.parent.as_posix()

        pair_key = base_name if rel_parent in {"", "."} else f"{rel_parent}/{base_name}"
        return pair_key, int(match.group("view"))

    def _build_multiview_groups(
        self,
        video_paths: List[Path],
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
        invalid_groups: set[str] = set()
        skipped_missing_suffix = 0
        skipped_unexpected_view = 0
        duplicate_groups = 0
        incomplete_groups = 0

        for prompt, video_path in zip(prompts, video_paths):
            parsed = self._parse_multiview_path(video_path)
            if parsed is None:
                skipped_missing_suffix += 1
                continue

            pair_key, view_id = parsed
            if view_id not in self.expected_view_ids:
                skipped_unexpected_view += 1
                continue

            views = grouped.setdefault(pair_key, {})
            if view_id in views:
                invalid_groups.add(pair_key)
                duplicate_groups += 1
                continue

            views[view_id] = {
                "video_path": video_path,
                "prompt": prompt,
            }

        sample_groups: List[Dict[str, Any]] = []
        for pair_key, views in grouped.items():
            if pair_key in invalid_groups:
                continue

            missing_view_ids = [view_id for view_id in self.expected_view_ids if view_id not in views]
            if missing_view_ids:
                incomplete_groups += 1
                continue

            ordered_views = {view_id: views[view_id] for view_id in self.expected_view_ids}
            sample_groups.append(
                {
                    "pair_key": pair_key,
                    "views": ordered_views,
                }
            )

        logger.info(
            "Built %d multiview groups from %d flat samples (missing_suffix=%d, unexpected_view=%d, "
            "duplicate_groups=%d, incomplete_groups=%d)",
            len(sample_groups),
            len(video_paths),
            skipped_missing_suffix,
            skipped_unexpected_view,
            duplicate_groups,
            incomplete_groups,
        )
        return sample_groups

    @staticmethod
    def _derive_condition_path(video_path: Path, condition_folder: Optional[str]) -> Optional[Path]:
        if not condition_folder:
            return None

        parent_dir = video_path.parent
        filename = video_path.name
        condition_path = parent_dir.parent / condition_folder / filename
        if not condition_path.exists() and parent_dir.parent.name == "processed2":
            fallback_condition_path = parent_dir.parent.parent / condition_folder / filename
            if fallback_condition_path.exists():
                condition_path = fallback_condition_path
        return condition_path

    def _select_group_prompt(self, group: Dict[str, Any]) -> str:
        prompts = [group["views"][view_id]["prompt"] for view_id in self.expected_view_ids]
        non_empty_prompts = [prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
        prompt_candidates = non_empty_prompts or prompts
        return random.choice(prompt_candidates) if prompt_candidates else ""

    def _sample_apply_horizontal_flip(self) -> bool:
        if not self.random_flip:
            return False
        return random.random() < float(self.random_flip)

    def _preprocess_video_with_shared_flip(
        self,
        path: Path,
        clip_plan: Dict[str, int],
        *,
        apply_horizontal_flip: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        if apply_horizontal_flip:
            frames = torch.flip(frames, dims=[3])

        frames = frames / 255.0
        frames = (frames - 0.5) / 0.5
        image = frames[:1].clone() if self.image_to_video else None
        return image, frames, None

    def _load_group_target_views(
        self,
        group: Dict[str, Any],
        clip_plan: Dict[str, int],
        *,
        apply_horizontal_flip: bool,
    ) -> Tuple[Dict[int, Optional[torch.Tensor]], Dict[int, torch.Tensor], Dict[int, Dict[str, int]]]:
        images: Dict[int, Optional[torch.Tensor]] = {}
        videos: Dict[int, torch.Tensor] = {}
        metadata: Dict[int, Dict[str, int]] = {}

        for view_id in self.expected_view_ids:
            video_path = group["views"][view_id]["video_path"]
            image, video, _ = self._preprocess_video_with_shared_flip(
                video_path,
                clip_plan,
                apply_horizontal_flip=apply_horizontal_flip,
            )
            images[view_id] = image
            videos[view_id] = video
            metadata[view_id] = {
                "num_frames": int(video.shape[0]),
                "height": int(video.shape[2]),
                "width": int(video.shape[3]),
            }

        return images, videos, metadata

    def _load_group_optional_condition_videos(
        self,
        group: Dict[str, Any],
        clip_plan: Dict[str, int],
        *,
        condition_folder: Optional[str],
        log_name: str,
        apply_horizontal_flip: bool,
    ) -> Optional[Dict[int, Optional[torch.Tensor]]]:
        if not condition_folder:
            return None

        outputs: Dict[int, Optional[torch.Tensor]] = {}
        for view_id in self.expected_view_ids:
            video_path = group["views"][view_id]["video_path"]
            condition_path = self._derive_condition_path(video_path, condition_folder)
            if condition_path is None or not condition_path.is_file():
                logger.warning(
                    "Missing %s for pair=%s view=%s: %s",
                    log_name,
                    group["pair_key"],
                    view_id,
                    condition_path,
                )
                outputs[view_id] = None
                continue

            try:
                _, condition_video, _ = self._preprocess_video_with_shared_flip(
                    condition_path,
                    clip_plan,
                    apply_horizontal_flip=apply_horizontal_flip,
                )
                outputs[view_id] = condition_video
            except Exception as exc:
                logger.warning(
                    "Failed to load %s for pair=%s view=%s: %s",
                    log_name,
                    group["pair_key"],
                    view_id,
                    exc,
                )
                outputs[view_id] = None

        return outputs

    def _load_group_static_videos(
        self,
        group: Dict[str, Any],
        clip_plan: Dict[str, int],
        target_videos: Dict[int, torch.Tensor],
        *,
        apply_horizontal_flip: bool,
    ) -> Optional[Dict[int, Optional[torch.Tensor]]]:
        if not self.require_static_videos:
            return None

        if self.static_video_source_mode == "copy_first_frame":
            return {
                view_id: target_videos[view_id][0:1].repeat(target_videos[view_id].shape[0], 1, 1, 1).clone()
                for view_id in self.expected_view_ids
            }

        outputs: Dict[int, Optional[torch.Tensor]] = {}
        for view_id in self.expected_view_ids:
            video_path = group["views"][view_id]["video_path"]
            static_path = self._derive_condition_path(video_path, self.static_video_subdir)
            if static_path is None or not static_path.is_file():
                raise RuntimeError(
                    f"Missing static video for pair={group['pair_key']} view={view_id}: {static_path}"
                )
            try:
                _, static_video, _ = self._preprocess_video_with_shared_flip(
                    static_path,
                    clip_plan,
                    apply_horizontal_flip=apply_horizontal_flip,
                )
                outputs[view_id] = static_video
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load static video for pair={group['pair_key']} view={view_id}: {static_path}"
                ) from exc
        return outputs

    def _get_group_item_once(self, index: int) -> Dict[str, Any]:
        group = self.sample_groups[index]
        clip_plan_view_id = self.expected_view_ids[0]
        clip_plan = self._build_raw_clip_plan(group["views"][clip_plan_view_id]["video_path"])
        apply_horizontal_flip = self._sample_apply_horizontal_flip()

        multiview_images, multiview_videos, multiview_video_metadata = self._load_group_target_views(
            group,
            clip_plan,
            apply_horizontal_flip=apply_horizontal_flip,
        )

        multiview_hand_videos = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.hand_video_subdir,
            log_name="hand video",
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_static_videos = self._load_group_static_videos(
            group,
            clip_plan,
            multiview_videos,
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_static_disparities = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.static_disparity_subdir,
            log_name="static disparity",
            apply_horizontal_flip=apply_horizontal_flip,
        )
        multiview_hand_disparities = self._load_group_optional_condition_videos(
            group,
            clip_plan,
            condition_folder=self.hand_disparity_subdir,
            log_name="hand disparity",
            apply_horizontal_flip=apply_horizontal_flip,
        )

        prompt_text = self.id_token + self._select_group_prompt(group)
        return {
            "prompt": prompt_text,
            "pair_key": group["pair_key"],
            "multiview_view_ids": list(self.expected_view_ids),
            "multiview_videos": multiview_videos,
            "multiview_images": multiview_images,
            "multiview_video_metadata": multiview_video_metadata,
            "multiview_hand_videos": multiview_hand_videos,
            "multiview_static_videos": multiview_static_videos,
            "multiview_static_disparities": multiview_static_disparities,
            "multiview_hand_disparities": multiview_hand_disparities,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index >= len(self.sample_groups):
            logger.warning(
                "Index %d is out of range (multiview dataset size: %d). Using mod operation: %d %% %d = %d",
                index,
                len(self.sample_groups),
                index,
                len(self.sample_groups),
                index % len(self.sample_groups),
            )
            index = index % len(self.sample_groups)

        return self._get_group_item_once(index)


class VideoDatasetWithConditionsAndResizeAndRectangleCrop(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that supports resizing and rectangle crop.
    """
    
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode
    
    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithHumanMotions(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that also supports human_motions data.
    
    This dataset loads:
    - video_latents/*.pt: Main video latents
    - prompt_embeds/*.pt: Text prompt embeddings
    - first-frame image latent derived from video_latents (for image-to-video)
    - human_motions/*.pt: SMPL pose parameters for AdaLN conditioning
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        use_gray_hand_videos: bool = False,
    ) -> None:
        # Initialize parent class with main video column
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
            use_gray_hand_videos=use_gray_hand_videos,
        )
        
        # Automatically derive human_motions paths from main video paths
        self.human_motions_paths = self._derive_human_motions_paths()
        
        # Validate that all human_motions files exist
        if not self.load_tensors:
            if any(not path.is_file() for path in self.human_motions_paths):
                missing_human_motions = [path for path in self.human_motions_paths if not path.is_file()]
                raise ValueError(
                    f"Some human_motions files are missing. First few missing files: {missing_human_motions[:5]}"
                )

    def _derive_human_motions_paths(self) -> List[Path]:
        """Derive human_motions paths from main video paths.
        
        Returns:
            List of human_motions file paths
        """
        human_motions_paths = []
        
        for video_path in self.video_paths:
            # Example: video_path = /path/to/sequences/videos/00001.mp4
            # We want: /path/to/sequences/human_motions/00001.pt
            
            # Get the parent directory (sequences)
            parent_dir = video_path.parent
            # Get the filename without extension (00001)
            filename_without_ext = video_path.stem
            # Construct the human_motions path
            human_motions_path = parent_dir.parent / "human_motions" / f"{filename_without_ext}.pt"
            human_motions_paths.append(human_motions_path)
        
        return human_motions_paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     return index

        # Get main video data from parent class
        main_data = super().__getitem__(index)
        
        # Load human_motions data
        if self.load_tensors:
            # Load preprocessed human_motions latents
            try:
                human_motions = self._load_human_motions(self.human_motions_paths[index])
                main_data["human_motions"] = human_motions
            except Exception as e:
                logger.warning(f"Failed to load human_motions for index {index}: {e}")
                main_data["human_motions"] = None
        else:
            # For raw video mode, we don't load human_motions as it's SMPL data
            main_data["human_motions"] = None

        return main_data

    def _load_human_motions(self, path: Path) -> torch.Tensor:
        """Load preprocessed human_motions data and extract body_pose."""
        if not path.exists():
            raise FileNotFoundError(f"Human motions file not found: {path}")
        
        # Load SMPL pose parameters dictionary
        human_motions_dict = torch.load(path, map_location="cpu", weights_only=True)
        
        # Extract body_pose from the dictionary
        if isinstance(human_motions_dict, dict) and "body_pose" in human_motions_dict:
            body_pose = human_motions_dict["body_pose"]
            logger.debug(f"Loaded body_pose with shape: {body_pose.shape}")
            return body_pose
        else:
            available_keys = list(human_motions_dict.keys()) if isinstance(human_motions_dict, dict) else 'Not a dict'
            raise KeyError(f"body_pose not found in human_motions data: {path}. Available keys: {available_keys}")


class VideoDatasetWithHumanMotionsAndResizing(VideoDatasetWithHumanMotions):
    """
    Extended VideoDatasetWithHumanMotions that also supports resizing.
    """
    
    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithHumanMotionsAndResizeAndRectangleCrop(VideoDatasetWithHumanMotions):
    """
    Extended VideoDatasetWithHumanMotions that supports resizing and rectangle crop.
    """
    
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(
        self, path: Path, clip_plan: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_reader = decord.VideoReader(uri=path.as_posix())
        resolved_clip_plan = self._resolve_raw_clip_plan(video_reader, clip_plan)
        frames = self._load_raw_frames(video_reader, resolved_clip_plan)

        nearest_res = (
            int(resolved_clip_plan["target_height"]),
            int(resolved_clip_plan["target_width"]),
        )
        frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, batch_size: int = 8, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}

        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning(
                "Calculating the length for bucket sampler is not possible when `drop_last` is set to True. This may cause problems when setting the number of epochs used for training."
            )
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = video_metadata["num_frames"], video_metadata["height"], video_metadata["width"]

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
                yield bucket
                del self.buckets[fhw]
                self.buckets[fhw] = []
