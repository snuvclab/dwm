from pathlib import Path
from typing import Optional, Tuple, Union


PathLike = Union[str, Path]


def resolve_video_path(video_path: PathLike, data_root: Optional[PathLike] = None) -> Path:
    path = Path(video_path)
    if path.is_absolute() or data_root is None:
        return path
    return Path(data_root) / path


def get_action_root(video_path: PathLike, data_root: Optional[PathLike] = None) -> Path:
    path = resolve_video_path(video_path, data_root=data_root)
    return path.parent.parent


def derive_sibling_file(
    video_path: PathLike,
    subdir: str,
    *,
    suffix: Optional[str] = None,
    data_root: Optional[PathLike] = None,
) -> Path:
    path = resolve_video_path(video_path, data_root=data_root)
    filename = path.name if suffix is None else f"{path.stem}{suffix}"
    return get_action_root(path) / subdir / filename


def get_context_names(video_path: PathLike, data_root: Optional[PathLike] = None) -> Tuple[str, str]:
    action_root = get_action_root(video_path, data_root=data_root)
    return action_root.parent.name, action_root.name


def build_output_stem(video_path: PathLike, data_root: Optional[PathLike] = None) -> str:
    collection_name, sample_name = get_context_names(video_path, data_root=data_root)
    return f"{collection_name}_{sample_name}_{Path(video_path).stem}"


def resolve_taste_rob_fixed_raymap(data_root: PathLike) -> Path:
    data_root = Path(data_root)
    candidates = [
        Path("data/taste_rob/cam_params/fixed_raymap.npz"),
        data_root / "cam_params" / "fixed_raymap.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
