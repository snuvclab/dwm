import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


CANONICAL_PIPELINE_TYPE = "cogvideox_fun_static_hand_concat"
CANONICAL_PIPELINE_CLASS = "CogVideoXFunStaticHandConcatPipeline"
CANONICAL_TRANSFORMER_CLASS = "CogVideoXFunStaticHandConcatTransformer3DModel"

PIPELINE_TYPE_ALIASES = {
    CANONICAL_PIPELINE_TYPE: CANONICAL_PIPELINE_TYPE,
    "cogvideox_fun_static_to_video_pose_concat": CANONICAL_PIPELINE_TYPE,
}


class ExperimentConfigLoader:
    """Load the standalone DWM CogVideoX static-hand-concat config tree."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config_dir = Path(config_dir) if config_dir is not None else self.repo_root / "configs" / "pipelines"

    def load_pipeline_config(self, pipeline_type: str) -> Dict[str, Any]:
        canonical_type = self.normalize_pipeline_type(pipeline_type)
        config_path = self.config_dir / f"{canonical_type}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def load_experiment_config(self, experiment_file: str, overrides: Optional[list[str]] = None) -> Dict[str, Any]:
        experiment_path = Path(experiment_file)
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

        with experiment_path.open("r", encoding="utf-8") as handle:
            experiment_config = yaml.safe_load(handle) or {}

        pipeline_config = self.load_pipeline_config(experiment_config["pipeline"]["type"])
        merged_config = self._merge_configs(pipeline_config, experiment_config)
        merged_config = self.apply_overrides(merged_config, overrides)
        merged_config = self.resolve_repo_relative_paths(merged_config)
        self.validate_config(merged_config)
        return merged_config

    def normalize_pipeline_type(self, pipeline_type: str) -> str:
        try:
            return PIPELINE_TYPE_ALIASES[pipeline_type]
        except KeyError as exc:
            supported = ", ".join(sorted(PIPELINE_TYPE_ALIASES))
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}. Supported types: {supported}") from exc

    def _merge_configs(self, pipeline_config: Dict[str, Any], experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(pipeline_config)

        if "pipeline" in experiment_config:
            merged["pipeline"] = {**merged.get("pipeline", {}), **experiment_config["pipeline"]}
        if "training" in experiment_config:
            merged["training"] = {**merged.get("training", {}), **experiment_config["training"]}

        merged["pipeline"]["type"] = self.normalize_pipeline_type(merged["pipeline"]["type"])
        merged["pipeline"]["class"] = CANONICAL_PIPELINE_CLASS
        merged["pipeline"]["transformer_class"] = CANONICAL_TRANSFORMER_CLASS

        for section in ("experiment", "data", "model", "logging", "slurm", "environment"):
            merged[section] = experiment_config.get(section, {})

        return merged

    def validate_config(self, config: Dict[str, Any]) -> None:
        for required in ("pipeline", "training", "model", "data"):
            if required not in config:
                raise ValueError(f"Missing required section: {required}")

        pipeline_type = config["pipeline"]["type"]
        if pipeline_type != CANONICAL_PIPELINE_TYPE:
            raise ValueError(
                f"Expected canonical pipeline type {CANONICAL_PIPELINE_TYPE}, got {pipeline_type}"
            )

    def resolve_repo_relative_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        resolved = copy.deepcopy(config)
        data_config = resolved.get("data", {})
        data_root = data_config.get("data_root")
        if isinstance(data_root, str) and data_root and not Path(data_root).is_absolute():
            data_config["data_root"] = str((self.repo_root / data_root).resolve())

        for key in ("dataset_file", "validation_set"):
            value = data_config.get(key)
            if value is None:
                continue
            data_config[key] = self._resolve_repo_relative_dataset_files(value)
        return resolved

    def _resolve_repo_relative_dataset_files(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._resolve_repo_relative_dataset_file(value)
        if isinstance(value, list):
            return [self._resolve_repo_relative_dataset_file(item) if isinstance(item, str) else item for item in value]
        return value

    def _resolve_repo_relative_dataset_file(self, dataset_file: str) -> str:
        path = Path(dataset_file)
        if path.is_absolute():
            return str(path)
        repo_candidate = (self.repo_root / path).resolve()
        if repo_candidate.exists():
            return str(repo_candidate)
        return dataset_file

    def apply_overrides(self, config: Dict[str, Any], overrides: Optional[list[str]]) -> Dict[str, Any]:
        if not overrides:
            return config

        updated = copy.deepcopy(config)
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: {override}. Use key=value")

            key, raw_value = override.split("=", 1)
            cursor = updated
            path = key.split(".")
            for part in path[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[path[-1]] = self._coerce_value(raw_value)

        return updated

    def _coerce_value(self, value: str) -> Any:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in value or "e" in lowered:
                return float(value)
            return int(value)
        except ValueError:
            return value


def load_experiment_config(experiment_file: str, overrides: Optional[list[str]] = None) -> Dict[str, Any]:
    loader = ExperimentConfigLoader()
    return loader.load_experiment_config(experiment_file, overrides)
