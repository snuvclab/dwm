import random
import logging
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
        prompt_embeds_subdir: str = "prompt_embeds_ego_fun_rewrite_wan",
        hand_video_subdir: str = "videos_hands",
        hand_video_latents_subdir: str = "hand_video_latents_wan",
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        align_width_to_32: bool = False,
    ) -> None:
        super().__init__()
        self.align_width_to_32 = align_width_to_32

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
        self.hand_video_subdir = hand_video_subdir
        self.hand_video_latents_subdir = hand_video_latents_subdir
        self.video_latents_subdir = video_latents_subdir
        self.static_video_latents_subdir = static_video_latents_subdir
        
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
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index])

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
                "image": image_latents,
                "video": video_latents,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, _ = self._preprocess_video(self.video_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }
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
                path.parent.parent.joinpath(self.prompt_subdir, path.name.replace(".mp4", ".txt"))
                for path in file_video_paths
            ]
            file_prompts = [path.read_text() for path in prompt_paths]

            all_video_paths.extend(file_video_paths)
            all_prompts.extend(file_prompts)
        
        return all_prompts, all_video_paths

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        try:
            if self.load_tensors:
                return self._load_preprocessed_latents_and_embeds(path)
            else:
                video_reader = decord.VideoReader(uri=path.as_posix())
                video_num_frames = len(video_reader)

                indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
                frames = video_reader.get_batch(indices)
                frames = frames[: self.max_num_frames].float()
                frames = frames.permute(0, 3, 1, 2).contiguous()
                frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

                image = frames[:1].clone() if self.image_to_video else None

                return image, frames, None
        except Exception as e:
            logger.error(f"Failed to preprocess video {path}: {e}")
            raise e

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            filename_without_ext = path.name.split(".")[0]
            pt_filename = f"{filename_without_ext}.pt"

            # The current path is something like: /a/b/c/d/videos/00001.mp4
            # We need to reach: /a/b/c/d/video_latents_wan/00001.pt (using configurable subdir)
            image_latents_path = path.parent.parent.joinpath("image_latents")
            video_latents_path = path.parent.parent.joinpath(self.video_latents_subdir)
            embeds_path = path.parent.parent.joinpath(self.prompt_embeds_subdir)

            if (
                not video_latents_path.exists()
                or not embeds_path.exists()
                or (self.image_to_video and not image_latents_path.exists())
            ):
                logger.error(f"Required folders not found for {path}")
                logger.error(f"video_latents_path exists: {video_latents_path.exists()}")
                logger.error(f"embeds_path exists: {embeds_path.exists()}")
                if self.image_to_video:
                    logger.error(f"image_latents_path exists: {image_latents_path.exists()}")
                raise ValueError(
                    f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
                )

            if self.image_to_video:
                image_latent_filepath = image_latents_path.joinpath(pt_filename)
            video_latent_filepath = video_latents_path.joinpath(pt_filename)
            embeds_filepath = embeds_path.joinpath(pt_filename)

            if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
                if self.image_to_video:
                    image_latent_filepath = image_latent_filepath.as_posix()
                video_latent_filepath = video_latent_filepath.as_posix()
                embeds_filepath = embeds_filepath.as_posix()
                logger.error(f"Required files not found:")
                logger.error(f"video_latent_filepath exists: {Path(video_latent_filepath).exists()}")
                logger.error(f"embeds_filepath exists: {Path(embeds_filepath).exists()}")
                raise ValueError(
                    f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
                )

            try:
                images = (
                    torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
                )
                latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
                embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)
            except Exception as e:
                logger.error(f"Failed to load torch files for {path}: {e}")
                logger.error(f"image_latent_filepath: {image_latent_filepath if self.image_to_video else 'N/A'}")
                logger.error(f"video_latent_filepath: {video_latent_filepath}")
                logger.error(f"embeds_filepath: {embeds_filepath}")
                raise e

            return images, latents, embeds
            
        except Exception as e:
            logger.error(f"Error in _load_preprocessed_latents_and_embeds for {path}: {e}")
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
        prompt_embeds_subdir: str = "prompt_embeds_ego_fun_rewrite_wan",
        hand_video_subdir: str = "videos_hands",
        hand_video_latents_subdir: str = "hand_video_latents_wan",
        video_latents_subdir: str = "video_latents_wan",
        static_video_latents_subdir: str = "static_video_latents_wan",
        align_width_to_32: bool = False,
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
            prompt_subdir=prompt_subdir,
            prompt_embeds_subdir=prompt_embeds_subdir,
            hand_video_subdir=hand_video_subdir,
            hand_video_latents_subdir=hand_video_latents_subdir,
            video_latents_subdir=video_latents_subdir,
            static_video_latents_subdir=static_video_latents_subdir,
            align_width_to_32=align_width_to_32,
        )

        # Store the use_gray_hand_videos flag
        self.use_gray_hand_videos = use_gray_hand_videos
        # Store the align_width_to_32 flag (for WAN 2.2 compatibility)
        self.align_width_to_32 = align_width_to_32
        
        # Automatically derive hand video and static video paths from main video paths
        if self.use_gray_hand_videos:
            self.hand_video_paths = self._derive_condition_video_paths("videos_hands_gray")
        else:
            self.hand_video_paths = self._derive_condition_video_paths(self.hand_video_subdir)
        self.static_video_paths = self._derive_condition_video_paths("videos_static")
        # Validate that all condition videos exist
        if not self.load_tensors:
            if any(not path.is_file() for path in self.hand_video_paths):
                missing_hand_videos = [path for path in self.hand_video_paths if not path.is_file()]
                raise ValueError(
                    f"Some hand video files are missing. First few missing files: {missing_hand_videos[:5]}"
                )
            
            # if any(not path.is_file() for path in self.hand_video_gray_paths):
            #     missing_hand_gray_videos = [path for path in self.hand_video_gray_paths if not path.is_file()]
            #     raise ValueError(
            #         f"Some hand gray video files are missing. First few missing files: {missing_hand_gray_videos[:5]}"
            #     )
            
            if any(not path.is_file() for path in self.static_video_paths):
                missing_static_videos = [path for path in self.static_video_paths if not path.is_file()]
                raise ValueError(
                    f"Some static video files are missing. First few missing files: {missing_static_videos[:5]}"
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
            
            # Get the parent directory (sequences)
            parent_dir = video_path.parent
            # Get the filename (00001.mp4)
            filename = video_path.name
            # Construct the condition video path
            condition_path = parent_dir.parent / condition_folder / filename
            condition_paths.append(condition_path)
        
        return condition_paths

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # if isinstance(index, list):
        #     return index

        # Get main video data from parent class
        main_data = super().__getitem__(index)
        
        # Load condition videos
        if self.load_tensors:
            # Load preprocessed latents for condition videos (using configurable subdirs)
            try:
                if self.use_gray_hand_videos:
                    # Gray hand videos use a different subdir pattern
                    hand_latents_subdir = self.hand_video_latents_subdir.replace("hand_video", "hand_video_gray")
                    hand_video_latents = self._load_condition_video_latents(self.hand_video_paths[index], hand_latents_subdir)
                else:
                    hand_video_latents = self._load_condition_video_latents(self.hand_video_paths[index], self.hand_video_latents_subdir)
                main_data["hand_videos"] = hand_video_latents
            except Exception as e:
                logger.warning(f"Failed to load hand video latents for index {index}: {e}")
                main_data["hand_videos"] = None

            try:
                static_video_latents = self._load_condition_video_latents(self.static_video_paths[index], self.static_video_latents_subdir)
                main_data["static_videos"] = static_video_latents
            except Exception as e:
                logger.warning(f"Failed to load static video latents for index {index}: {e}")
                main_data["static_videos"] = None
        else:
            # Load raw videos for condition videos
            try:
                _, hand_video, _ = self._preprocess_video(self.hand_video_paths[index])
                main_data["hand_videos"] = hand_video
            except Exception as e:
                logger.warning(f"Failed to load hand video for index {index}: {e}")
                main_data["hand_videos"] = None
            
            try:
                _, static_video, _ = self._preprocess_video(self.static_video_paths[index])
                main_data["static_videos"] = static_video
            except Exception as e:
                logger.warning(f"Failed to load static video for index {index}: {e}")
                main_data["static_videos"] = None

        return main_data

    def _load_condition_video_latents(self, path: Path, latent_folder: str) -> Optional[torch.Tensor]:
        """Load preprocessed latents for condition videos.
        
        Returns:
            torch.Tensor if latents are successfully loaded, None if folder or file doesn't exist.
        """
        try:
            filename_without_ext = path.name.split(".")[0]
            pt_filename = f"{filename_without_ext}.pt"
            
            # The current path is something like: /a/b/c/sequences/videos_hands/00001.mp4
            # We need to reach: /a/b/c/processed/hand_video_latents/00001.pt
            # or: /a/b/c/processed/static_video_latents/00001.pt
            
            # Get the sequences directory (parent of the condition folder)
            sequences_dir = path.parent.parent
            # Get the action directory (parent of sequences)
            action_dir = sequences_dir.parent
            # Construct the latents path
            latents_path = action_dir / "processed2" / latent_folder
            
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
    
    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


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

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = self._resize_for_rectangle_crop(frames, nearest_res)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None

    # Resolution bucketing (and optional align_width_to_32) is handled by VideoDataset._find_nearest_resolution


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
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

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
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
    - image_latents/*.pt: Image latents (for image-to-video)
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
    
    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
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

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
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