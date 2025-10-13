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
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
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
        if dataset_file is None:
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

    def _load_dataset_from_datafile(self) -> Tuple[List[str], List[str]]:
        with open(self.data_root / self.dataset_file, "r") as f:
            video_paths = [self.data_root.joinpath(line.strip()) for line in f.readlines() if len(line.strip()) > 0]
        prompt_paths = [path.parent.parent.joinpath("prompts", path.name.replace(".mp4", ".txt")) for path in video_paths]
        prompts = [path.read_text() for path in prompt_paths]

        return prompts, video_paths

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

                # Warn if video doesn't have the expected number of frames
                if video_num_frames != self.max_num_frames:
                    logger.warning(
                        f"Video has {video_num_frames} frames, expected {self.max_num_frames} frames. "
                        f"Path: {path}"
                    )

                # Calculate step, ensuring it's at least 1 to avoid "range() arg 3 must not be zero" error
                step = max(1, video_num_frames // self.max_num_frames)
                indices = list(range(0, video_num_frames, step))
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
            # We need to reach: /a/b/c/d/video_latents/00001.pt
            image_latents_path = path.parent.parent.joinpath("image_latents")
            video_latents_path = path.parent.parent.joinpath("video_latents")
            embeds_path = path.parent.parent.joinpath("prompt_embeds")

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
    - raymaps: Camera ray maps for 3D scene understanding
    - image_goal_latents: Goal image latents for navigation tasks
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
        split_hands: bool = False,
        use_smpl_pos_map: bool = False,
        compress_smpl_pos_map_temporal: bool = False,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 8,
        load_raymaps: bool = False,
        load_image_goal: bool = False,
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
        )
        
        # Store the flags
        self.use_gray_hand_videos = use_gray_hand_videos
        self.split_hands = split_hands
        self.use_smpl_pos_map = use_smpl_pos_map
        self.compress_smpl_pos_map_temporal = compress_smpl_pos_map_temporal
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.load_raymaps = load_raymaps
        self.load_image_goal = load_image_goal
        
        # Automatically derive hand video and static video paths from main video paths
        if self.split_hands:
            # Split hands mode: load left and right hand videos separately
            self.hand_video_left_paths = self._derive_condition_video_paths("videos_hands_gray_left")
            self.hand_video_right_paths = self._derive_condition_video_paths("videos_hands_gray_right")
            # For backward compatibility, we'll combine them in __getitem__
            self.hand_video_paths = None  # Not used in split mode
        else:
            # Regular mode: load combined hand videos
            if self.use_gray_hand_videos:
                self.hand_video_paths = self._derive_condition_video_paths("videos_hands_gray")
            else:
                self.hand_video_paths = self._derive_condition_video_paths("videos_hands")
            self.hand_video_left_paths = None
            self.hand_video_right_paths = None
        
        self.static_video_paths = self._derive_condition_video_paths("videos_static")
        
        # Derive SMPL pos map paths if enabled
        if self.use_smpl_pos_map:
            self.smpl_pos_map_paths = self._derive_condition_video_paths("smpl_pos_map_egoallo")
        else:
            self.smpl_pos_map_paths = None
        
        # Validate that all condition videos exist
        if not self.load_tensors:
            if self.split_hands:
                # Validate left and right hand videos separately
                if any(not path.is_file() for path in self.hand_video_left_paths):
                    missing_hand_left_videos = [path for path in self.hand_video_left_paths if not path.is_file()]
                    raise ValueError(
                        f"Some left hand video files are missing. First few missing files: {missing_hand_left_videos[:5]}"
                    )
                
                if any(not path.is_file() for path in self.hand_video_right_paths):
                    missing_hand_right_videos = [path for path in self.hand_video_right_paths if not path.is_file()]
                    raise ValueError(
                        f"Some right hand video files are missing. First few missing files: {missing_hand_right_videos[:5]}"
                    )
            else:
                # Validate combined hand videos
                if any(not path.is_file() for path in self.hand_video_paths):
                    missing_hand_videos = [path for path in self.hand_video_paths if not path.is_file()]
                    raise ValueError(
                        f"Some hand video files are missing. First few missing files: {missing_hand_videos[:5]}"
                    )
            
            if any(not path.is_file() for path in self.static_video_paths):
                missing_static_videos = [path for path in self.static_video_paths if not path.is_file()]
                raise ValueError(
                    f"Some static video files are missing. First few missing files: {missing_static_videos[:5]}"
                )
            
            # Validate SMPL pos map videos if enabled
            if self.use_smpl_pos_map:
                if any(not path.is_file() for path in self.smpl_pos_map_paths):
                    missing_smpl_pos_maps = [path for path in self.smpl_pos_map_paths if not path.is_file()]
                    raise ValueError(
                        f"Some SMPL pos map files are missing. First few missing files: {missing_smpl_pos_maps[:5]}"
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
            # Try to load preprocessed latents first, fallback to raw video encoding
            try:
                if self.split_hands:
                    # Load left and right hand videos separately
                    hand_left_latents = self._load_condition_video_latents_or_encode(self.hand_video_left_paths[index], "hand_video_gray_left_latents")
                    hand_right_latents = self._load_condition_video_latents_or_encode(self.hand_video_right_paths[index], "hand_video_gray_right_latents")
                    # Concatenate along channel dimension: [C1, F, H, W] + [C2, F, H, W] -> [C1+C2, F, H, W]
                    hand_video_latents = torch.cat([hand_left_latents, hand_right_latents], dim=0)
                    main_data["hand_videos"] = hand_video_latents
                else:
                    # Regular mode: load combined hand videos
                    if self.use_gray_hand_videos:
                        hand_video_latents = self._load_condition_video_latents_or_encode(self.hand_video_paths[index], "hand_video_gray_latents")
                    else:
                        hand_video_latents = self._load_condition_video_latents_or_encode(self.hand_video_paths[index], "hand_video_latents")
                    main_data["hand_videos"] = hand_video_latents
            except Exception as e:
                logger.warning(f"Failed to load hand video latents for index {index}: {e}")
                main_data["hand_videos"] = None
            
            try:
                static_video_latents = self._load_condition_video_latents_or_encode(self.static_video_paths[index], "static_video_latents")
                main_data["static_videos"] = static_video_latents
            except Exception as e:
                logger.warning(f"Failed to load static video latents for index {index}: {e}")
                main_data["static_videos"] = None
            
            # Load SMPL pos map if enabled
            if self.use_smpl_pos_map:
                try:
                    if self.compress_smpl_pos_map_temporal:
                        # Load raw video and compress temporal+spatial
                        smpl_pos_map = self._load_and_compress_smpl_pos_map(
                            self.smpl_pos_map_paths[index],
                            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
                            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                        )
                    else:
                        # Load latents (existing behavior)
                        smpl_pos_map = self._load_condition_video_latents_or_encode(
                            self.smpl_pos_map_paths[index], 
                            "smpl_pos_map_egoallo_latents"
                        )
                    main_data["smpl_pos_map"] = smpl_pos_map
                except Exception as e:
                    logger.warning(f"Failed to load SMPL pos map for index {index}: {e}")
                    main_data["smpl_pos_map"] = None
            else:
                main_data["smpl_pos_map"] = None
            
            # Load raymaps if enabled
            if self.load_raymaps:
                try:
                    name = self.video_paths[index].stem
                    raymap = torch.load(
                        self.video_paths[index].parent.parent / "raymaps" / f"{name}.pt",
                        map_location="cpu",
                        weights_only=True
                    )
                    main_data["raymap"] = raymap
                except Exception as e:
                    logger.warning(f"Failed to load raymap for index {index}: {e}")
                    main_data["raymap"] = None
            else:
                main_data["raymap"] = None
            
            # Load image_goal_latents if enabled
            if self.load_image_goal:
                try:
                    filename_without_ext = self.video_paths[index].name.split(".")[0]
                    pt_filename = f"{filename_without_ext}.pt"
                    image_goal_latents_path = self.video_paths[index].parent.parent.joinpath("image_goal_latents")
                    image_goal_latent_filepath = image_goal_latents_path.joinpath(pt_filename)
                    
                    if image_goal_latent_filepath.is_file():
                        image_goal_latents = torch.load(
                            image_goal_latent_filepath,
                            map_location="cpu",
                            weights_only=True
                        )
                        main_data["image_goal"] = image_goal_latents
                    else:
                        logger.warning(f"Image goal latents not found: {image_goal_latent_filepath}")
                        main_data["image_goal"] = None
                except Exception as e:
                    logger.warning(f"Failed to load image_goal_latents for index {index}: {e}")
                    main_data["image_goal"] = None
            else:
                main_data["image_goal"] = None
        else:
            # Load raw videos for condition videos
            try:
                if self.split_hands:
                    # Load left and right hand videos separately
                    _, hand_left_video, _ = self._preprocess_video(self.hand_video_left_paths[index])
                    _, hand_right_video, _ = self._preprocess_video(self.hand_video_right_paths[index])
                    # Concatenate along channel dimension: [F, C1, H, W] + [F, C2, H, W] -> [F, C1+C2, H, W]
                    hand_video = torch.cat([hand_left_video, hand_right_video], dim=1)
                    main_data["hand_videos"] = hand_video
                else:
                    # Regular mode: load combined hand videos
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
            
            # Load SMPL pos map if enabled
            if self.use_smpl_pos_map:
                try:
                    _, smpl_pos_map, _ = self._preprocess_video(self.smpl_pos_map_paths[index])
                    main_data["smpl_pos_map"] = smpl_pos_map
                except Exception as e:
                    logger.warning(f"Failed to load SMPL pos map for index {index}: {e}")
                    main_data["smpl_pos_map"] = None
            else:
                main_data["smpl_pos_map"] = None
            
            # Load raymaps if enabled (raw mode)
            if self.load_raymaps:
                try:
                    name = self.video_paths[index].stem
                    raymap = np.load(self.data_root / "raymaps" / f"{name}.npz")['raymap']
                    main_data["raymap"] = torch.tensor(raymap, dtype=torch.float32)
                    
                    # Also load raymap_abs if available
                    raymap_abs_file = self.data_root / "raymaps" / f"{name}_abs.npz"
                    if raymap_abs_file.exists():
                        raymap_abs = np.load(raymap_abs_file)['raymap']
                        main_data["raymap_abs"] = torch.tensor(raymap_abs, dtype=torch.float32)
                    else:
                        main_data["raymap_abs"] = None
                except Exception as e:
                    logger.warning(f"Failed to load raymap for index {index}: {e}")
                    main_data["raymap"] = None
                    main_data["raymap_abs"] = None
            else:
                main_data["raymap"] = None
                main_data["raymap_abs"] = None
            
            # Load image_goal if enabled (raw mode)
            if self.load_image_goal:
                try:
                    # Get the last frame as image_goal from main video
                    if main_data["video"] is not None:
                        image_goal = main_data["video"][-1:].clone()
                        main_data["image_goal"] = image_goal
                    else:
                        main_data["image_goal"] = None
                except Exception as e:
                    logger.warning(f"Failed to extract image_goal for index {index}: {e}")
                    main_data["image_goal"] = None
            else:
                main_data["image_goal"] = None

        return main_data

    def _load_condition_video_latents(self, path: Path, latent_folder: str) -> torch.Tensor:
        """Load preprocessed latents for condition videos."""
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
                logger.warning(f"Latents folder not found: {latents_path}")
                raise FileNotFoundError(f"Latents folder not found: {latents_path}")
            
            latent_filepath = latents_path.joinpath(pt_filename)
            
            if not latent_filepath.is_file():
                logger.warning(f"Latent file not found: {latent_filepath}")
                raise FileNotFoundError(f"Latent file not found: {latent_filepath}")
            
            try:
                latents = torch.load(latent_filepath, map_location="cpu", weights_only=True)
                return latents
            except Exception as e:
                logger.error(f"Failed to load latent file {latent_filepath}: {e}")
                raise e
                
        except Exception as e:
            if self.split_hands:
                logger.debug(f"Latents not found for {path}, will load raw video: {e}")
            else:
                logger.error(f"Error in _load_condition_video_latents for {path} in {latent_folder}: {e}")
            raise e

    def _load_condition_video_latents_or_encode(self, path: Path, latent_folder: str) -> torch.Tensor:
        """Load preprocessed latents if available, otherwise return raw video for encoding in training."""
        # Try to load latents first
        try:
            # First try to load preprocessed latents
            return self._load_condition_video_latents(path, latent_folder)
        except (FileNotFoundError, Exception) as e:
            logger.info(f"Preprocessed latents not found for {path}, will encode from raw video in training: {e}")
            
            # Fallback: return raw video tensor
            try:
                if self.split_hands:
                    # For split_hands mode, load raw video without downsampling
                    video_reader = decord.VideoReader(uri=path.as_posix())
                    video_num_frames = len(video_reader)
                    frame_indices = list(range(video_num_frames))
                    frames = video_reader.get_batch(frame_indices)
                    frames = frames.float()
                    video = frames.permute(3, 0, 1, 2).contiguous()  # [C, F, H, W]
                    return video
                else:
                    # Regular preprocessing for non-split_hands mode
                    _, video, _ = self._preprocess_video(path)
                    return video
            except Exception as encode_error:
                logger.error(f"Failed to load raw video {path}: {encode_error}")
                raise encode_error

    def _load_and_compress_smpl_pos_map(
        self, 
        path: Path, 
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 8,
    ) -> torch.Tensor:
        """
        Load raw SMPL pos map video and compress both temporal and spatial dimensions.
        
        Temporal compression: Rearrange frames to channel dimension (like raymap processing)
        Spatial compression: Downsample by factor of vae_scale_factor_spatial using bilinear interpolation
        
        Args:
            path: Path to video file
            vae_scale_factor_temporal: Temporal compression factor (default: 4)
            vae_scale_factor_spatial: Spatial compression factor (default: 8)
        
        Returns:
            torch.Tensor: [T', C*vae_scale_factor_temporal, H/spatial, W/spatial]
                         where T' = ceil(F / vae_scale_factor_temporal)
        
        Example:
            Input: [49, 480, 720, 3] (F, H, W, C)
            Output: [13, 12, 60, 90] (T'=13, C*4=12, H/8=60, W/8=90)
        """
        from einops import rearrange
        import torch.nn.functional as F
        
        # Load video
        video_reader = decord.VideoReader(uri=path.as_posix())
        video = video_reader[:]  # [F, H, W, C]

        video = video.permute(3, 0, 1, 2)  # [C, F, H, W]
        video = video.unsqueeze(0)  # [1, C, F, H, W]
        
        # Normalize [0, 255] -> [-1, 1]
        video = video / 255.0
        video = (video - 0.5) / 0.5
        
        # Spatial downsampling (H, W) -> (H/spatial_factor, W/spatial_factor)
        b, c, f, h, w = video.shape
        # Reshape for spatial downsampling: [B, C, F, H, W] -> [B*F, C, H, W]
        video_2d = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        
        # Downsample using bilinear interpolation
        new_h = h // vae_scale_factor_spatial
        new_w = w // vae_scale_factor_spatial
        video_2d = F.interpolate(
            video_2d, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Reshape back: [B*F, C, H/8, W/8] -> [B, C, F, H/8, W/8]
        video = video_2d.reshape(b, f, c, new_h, new_w).permute(0, 2, 1, 3, 4)
        
        # Temporal padding if needed (pad at the beginning like raymap)
        num_frames = video.shape[2]
        if num_frames % vae_scale_factor_temporal != 0:
            pad_frames = vae_scale_factor_temporal - (num_frames % vae_scale_factor_temporal)
            # Repeat first pad_frames frames
            padding = video[:, :, :pad_frames, :, :]
            video = torch.cat([padding, video], dim=2)
        
        # 6. Rearrange: compress temporal dimension into channel
        # [B, C, (n*t), H, W] -> [B, t, (n*C), H, W]
        video = rearrange(
            video,
            "b c (n t) h w -> b t (n c) h w",
            n=vae_scale_factor_temporal,
        )
        
        # 7. Remove batch dimension
        video = video.squeeze(0)  # [T', C*n, H/8, W/8]
        
        logger.debug(f"Loaded and compressed SMPL pos map: {path} -> shape {video.shape}")
        
        return video


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

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


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

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


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

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


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

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


class VideoDatasetWithHumanMotions(VideoDatasetWithConditions):
    """
    Extended VideoDatasetWithConditions that also supports human_motions data.
    
    This dataset loads:
    - video_latents/*.pt: Main video latents
    - prompt_embeds/*.pt: Text prompt embeddings  
    - image_latents/*.pt: Image latents (for image-to-video)
    - human_motions/*.pt: SMPL pose parameters for AdaLN conditioning
    - raymaps/*.pt: Camera ray maps (optional)
    - image_goal_latents/*.pt: Goal image latents (optional)
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
        split_hands: bool = False,
        use_smpl_pos_map: bool = False,
        compress_smpl_pos_map_temporal: bool = False,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 8,
        load_raymaps: bool = False,
        load_image_goal: bool = False,
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
            split_hands=split_hands,
            use_smpl_pos_map=use_smpl_pos_map,
            compress_smpl_pos_map_temporal=compress_smpl_pos_map_temporal,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            load_raymaps=load_raymaps,
            load_image_goal=load_image_goal,
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

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


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

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


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