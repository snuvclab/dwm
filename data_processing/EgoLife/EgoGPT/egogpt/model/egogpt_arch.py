# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import re
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from egogpt.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX
from egogpt.mm_utils import get_anyres_image_grid_shape
from egogpt.utils import lengths_to_padding_mask, rank0_print, rank_print

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_resampler.builder import build_vision_resampler
from .speech_encoder.builder import build_speech_encoder
from .speech_projector.builder import build_speech_projector


class EgoGPTMetaModel:
    def __init__(self, config):
        super(EgoGPTMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(
                config, vision_tower=self.vision_tower
            )
            self.mm_projector = build_vision_projector(
                config, vision_cfg=self.vision_tower.config
            )

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

        if hasattr(config, "speech_encoder"):
            self.speech_encoder = build_speech_encoder(config)
            self.speech_projector = build_speech_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_speech_encoder(self):
        speech_encoder = getattr(self, "speech_encoder", None)
        if type(speech_encoder) is list:
            speech_encoder = speech_encoder[0]
        return speech_encoder

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(
            model_args, "vision_tower_pretrained", ""
        )

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(
                model_args, vision_tower=vision_tower
            )
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = getattr(
            vision_resampler, "hidden_size", vision_tower.hidden_size
        )
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, "add_faster_video"):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(
                self.config, vision_cfg=vision_tower.config
            )

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            incompatible_keys = self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
            rank0_print(
                f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )
            incompatible_keys = self.vision_resampler.load_state_dict(
                get_w(mm_projector_weights, "vision_resampler"), strict=False
            )
            rank0_print(
                f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )

    def initialize_speech_modules(self, model_args, fsdp=None):
        self.config.speech_encoder = getattr(model_args, "speech_encoder", None)
        self.config.speech_encoder_type = getattr(
            model_args, "speech_encoder_type", None
        )
        self.config.speech_projector_type = getattr(
            model_args, "speech_projector_type", "linear"
        )
        self.config.speech_encoder_ds_rate = getattr(
            model_args, "speech_encoder_ds_rate", 5
        )
        self.config.speech_encoder_hidden_size = getattr(
            model_args, "speech_encoder_hidden_size", 1280
        )
        self.config.delay_load_audio = getattr(model_args, "delay_load_audio", True)

        if self.get_speech_encoder() is None:
            speech_encoder = build_speech_encoder(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.speech_encoder = [speech_encoder]
            else:
                self.speech_encoder = speech_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                speech_encoder = self.speech_encoder[0]
            else:
                speech_encoder = self.speech_encoder
            speech_encoder.load_model(self.config)

        if getattr(self, "speech_projector", None) is None:
            self.speech_projector = build_speech_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.speech_projector.parameters():
                p.requires_grad = True

        if model_args.pretrain_speech_projector is not None:
            pretrain_speech_projector_weights = torch.load(
                model_args.pretrain_speech_projector, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.speech_projector.load_state_dict(
                get_w(pretrain_speech_projector_weights, "speech_projector"),
                strict=False,
            )


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class EgoGPTMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_speech_encoder(self):
        return self.get_model().get_speech_encoder()

    def get_speech_projector(self):
        return self.get_model().speech_projector

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        image_feature = nn.functional.avg_pool2d(image_feature, stride)
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        # if self.config.mm_spatial_pool_mode == "average":
        #     image_feature = nn.functional.avg_pool2d(image_feature, stride)
        # elif self.config.mm_spatial_pool_mode == "max":
        #     image_feature = nn.functional.max_pool2d(image_feature, stride)
        # elif self.config.mm_spatial_pool_mode == "bilinear":
        #     height, width = image_feature.shape[2:]
        #     scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        #     image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        # else:
        #     raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_speech(self, speech, speech_lengths):
        # audio cuttting
        speech_encoder_type = self.config.speech_encoder_type
        speech_encoder = self.get_speech_encoder()
        if "whisper" in speech_encoder_type.lower():
            encoder_outs = speech_encoder(speech.permute(0, 2, 1))
            speech_lengths = (speech_lengths + 1) // 2
        else:
            raise ValueError(f"Unknown speech encoder: {speech_encoder}")
        speech_projector_type = self.config.speech_projector_type
        speech_projector = self.get_speech_projector()
        if speech_projector_type == "linear":
            encoder_outs = speech_projector(encoder_outs)
            speech_lengths = speech_lengths // speech_projector.k
        else:
            raise ValueError(f"Unknown speech projector: {speech_projector_type}")
        speech_features = [
            encoder_outs[i, : speech_lengths[i]] for i in range(len(encoder_outs))
        ]
        return speech_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat(
            (
                image_feature,
                self.model.image_newline[:, None, None]
                .expand(*image_feature.shape[:-1], 1)
                .to(image_feature.device),
            ),
            dim=-1,
        )
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def prepare_inputs_labels_for_speech_and_text(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        speech,
        speech_lengths,
        images,
        image_sizes=None,
        modalities=["image"],
    ):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        speech_encoder = self.get_speech_encoder()
        if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        speech_features = self.encode_speech(speech, speech_lengths)

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            # print(f"Images: {images}, {type(images)}, {len(images)}")
            # print(f"Video idx in batch: {modalities}")
            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            # concat_images = torch.cat([torch.tensor(image) for image in images_list], dim=0)
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            concat_images.requires_grad_(True)
            encoded_image_features = self.encode_images(concat_images)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(
                self.config, "mm_newline_position", "one_token"
            )

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(
                                    all_faster_video_features[image_idx]
                                )
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(
                                            torch.cat(
                                                (
                                                    image_feature[_],
                                                    self.model.faster_token[None].to(
                                                        image_feature.device
                                                    ),
                                                ),
                                                dim=0,
                                            )
                                        )
                                    else:
                                        concat_slow_fater_token.append(
                                            torch.cat(
                                                (
                                                    faster_video_feature[_],
                                                    self.model.faster_token[None].to(
                                                        image_feature.device
                                                    ),
                                                ),
                                                dim=0,
                                            )
                                        )
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))

                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[None].to(
                                            image_feature.device
                                        ),
                                    ),
                                    dim=0,
                                )
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(
                                f"Unexpected mm_newline_position: {mm_newline_position}"
                            )
                    elif (
                        image_feature.shape[0] > 1
                    ):  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(
                                r"anyres_max_(\d+)", image_aspect_ratio
                            )
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(
                                    matched_anyres_max_num_patches.group(1)
                                )

                        if (
                            image_aspect_ratio == "anyres"
                            or "anyres_max" in image_aspect_ratio
                        ):
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = (
                                    self.get_vision_tower().image_size
                                )
                            else:
                                raise ValueError(
                                    "vision_tower_image_size is not found in the vision tower."
                                )
                            try:
                                (
                                    num_patch_width,
                                    num_patch_height,
                                ) = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    vision_tower_image_size,
                                )
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif (
                            "unpad" in mm_patch_merge_type
                            and "anyres_max" in image_aspect_ratio
                            and matched_anyres_max_num_patches
                        ):
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(
                                    image_feature,
                                    [int(h // times), int(w // times)],
                                    mode="bilinear",
                                )[0]
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat(
                                (base_image_feature, image_feature), dim=0
                            )
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.model.image_newline[None]), dim=0
                            )

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]
        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            num_image = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # if num_speech:
            #     print("has <speech>")
            # if num_image:
            #     print("has <image>")
            num_speech_images = num_speech + num_image

            if num_speech_images == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_speech_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                cur_image_idx += 1
                continue

            multimodal_token_indices = (
                [-1]
                + torch.where(
                    (cur_input_ids == SPEECH_TOKEN_INDEX)
                    | (cur_input_ids == IMAGE_TOKEN_INDEX)
                )[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_nospeech_image = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech_image = []
            for i in range(len(multimodal_token_indices) - 1):
                cur_input_ids_nospeech_image.append(
                    cur_input_ids[
                        multimodal_token_indices[i]
                        + 1 : multimodal_token_indices[i + 1]
                    ]
                )
                cur_labels_nospeech_image.append(
                    cur_labels[
                        multimodal_token_indices[i]
                        + 1 : multimodal_token_indices[i + 1]
                    ]
                )

            split_sizes = [x.shape[0] for x in cur_labels_nospeech_image]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_nospeech_image)
            )
            cur_input_embeds_no_speech_image = torch.split(
                cur_input_embeds, split_sizes, dim=0
            )
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_speech_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech_image[i])
                cur_new_labels.append(cur_labels_nospeech_image[i])
                if i < num_speech_images:
                    if i < num_image:
                        cur_images_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_images_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_images_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    else:
                        cur_speech_features = speech_features[cur_speech_idx]
                        cur_speech_idx += 1
                        cur_new_input_embeds.append(cur_speech_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_speech_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            if num_image == 0:
                cur_new_input_embeds = torch.cat(
                    [cur_new_input_embeds, image_features[cur_image_idx][0:0]], dim=0
                )
                cur_image_idx += 1

            if num_speech == 0:
                cur_new_input_embeds = torch.cat(
                    [cur_new_input_embeds, speech_features[cur_speech_idx][0:0]], dim=0
                )
                cur_speech_idx += 1

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def prepare_inputs_labels_for_speech_and_text_debug(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        speech,
        speech_lengths,
        images,
        image_sizes=None,
        modalities=["image"],
    ):
        # vision_tower = self.get_vision_tower()
        # # rank_print(modalities)
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels
        # speech_encoder = self.get_speech_encoder()
        # if speech_encoder is None or speech is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels

        speech_features = self.encode_speech(speech, speech_lengths)

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            # print(f"Images: {images}, {type(images)}, {len(images)}")
            # print(f"Video idx in batch: {modalities}")
            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            # concat_images = torch.cat([torch.tensor(image) for image in images_list], dim=0)
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            concat_images.requires_grad_(True)
            encoded_image_features = self.encode_images(concat_images)
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(
                self.config, "mm_newline_position", "one_token"
            )

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)
                            new_image_features.append(image_feature.flatten(0, 1))
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat(
                                    (
                                        image_feature,
                                        self.model.image_newline[None].to(
                                            image_feature.device
                                        ),
                                    ),
                                    dim=0,
                                )
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(
                                f"Unexpected mm_newline_position: {mm_newline_position}"
                            )
                    elif (
                        image_feature.shape[0] > 1
                    ):  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(
                                r"anyres_max_(\d+)", image_aspect_ratio
                            )
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(
                                    matched_anyres_max_num_patches.group(1)
                                )

                        if (
                            image_aspect_ratio == "anyres"
                            or "anyres_max" in image_aspect_ratio
                        ):
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = (
                                    self.get_vision_tower().image_size
                                )
                            else:
                                raise ValueError(
                                    "vision_tower_image_size is not found in the vision tower."
                                )
                            try:
                                (
                                    num_patch_width,
                                    num_patch_height,
                                ) = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    vision_tower_image_size,
                                )
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif (
                            "unpad" in mm_patch_merge_type
                            and "anyres_max" in image_aspect_ratio
                            and matched_anyres_max_num_patches
                        ):
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(
                                    image_feature,
                                    [int(h // times), int(w // times)],
                                    mode="bilinear",
                                )[0]
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat(
                                (base_image_feature, image_feature), dim=0
                            )
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.model.image_newline[None]), dim=0
                            )

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]
        new_input_embeds = []
        new_labels = []
        cur_speech_idx = 0
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == SPEECH_TOKEN_INDEX).sum()
            num_image = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_speech + num_image == 0:
                cur_speech_features = speech_features[cur_speech_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_speech_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_speech_idx += 1
                cur_image_idx += 1
                continue

            multimodal_token_indices = sorted(
                [-1]
                + torch.where(cur_input_ids == SPEECH_TOKEN_INDEX)[0].tolist()
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            for i in range(len(multimodal_token_indices) - 1):
                cur_input_ids_nospeech.append(
                    cur_input_ids[
                        multimodal_token_indices[i]
                        + 1 : multimodal_token_indices[i + 1]
                    ]
                )
                cur_labels_nospeech.append(
                    cur_labels[
                        multimodal_token_indices[i]
                        + 1 : multimodal_token_indices[i + 1]
                    ]
                )

            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_nospeech)
            )
            cur_input_embeds_no_speech = torch.split(
                cur_input_embeds, split_sizes, dim=0
            )
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_speech + num_image + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                if cur_speech_idx < num_speech:
                    try:
                        cur_speech_features = speech_features[cur_speech_idx]
                    except:
                        cur_speech_features = speech_features[cur_speech_idx - 1]
                    cur_speech_idx += 1
                    cur_new_input_embeds.append(cur_speech_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_speech_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                if cur_image_idx < num_image:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as speech features can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        print(f"new_input_embeds: {new_input_embeds[0].shape}")
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )
