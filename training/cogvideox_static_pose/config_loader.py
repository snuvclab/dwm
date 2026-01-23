import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentConfigLoader:
    """
    YAML-based configuration loader for CogVideoX pose training experiments.
    
    Supports:
    - Pipeline-specific configurations
    - Experiment-specific configurations
    - Configuration merging and validation
    - Runtime overrides
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_pipeline_config(self, pipeline_type: str) -> Dict[str, Any]:
        """Load pipeline configuration from YAML."""
        config_path = self.config_dir / "pipelines" / f"{pipeline_type}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"📋 Loaded pipeline config: {pipeline_type}")
        return config
    
    def load_experiment_config(self, experiment_file: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        experiment_path = Path(experiment_file)
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {experiment_path}")
            
        with open(experiment_path, 'r') as f:
            experiment_config = yaml.safe_load(f)
        
        print(f"📋 Loaded experiment config: {experiment_path}")
        
        # Load and merge pipeline config
        pipeline_type = experiment_config["pipeline"]["type"]
        pipeline_config = self.load_pipeline_config(pipeline_type)
        
        # Merge configurations
        merged_config = self._merge_configs(pipeline_config, experiment_config)
        
        # Validate merged configuration
        self.validate_config(merged_config)
        
        return merged_config
    
    def _merge_configs(self, pipeline_config: Dict, experiment_config: Dict) -> Dict:
        """Merge pipeline and experiment configurations."""
        merged = pipeline_config.copy()
        
        # Override with experiment-specific settings
        if "training" in experiment_config:
            merged["training"] = {**merged.get("training", {}), **experiment_config["training"]}
        
        if "adapter" in experiment_config:
            merged["adapter"] = {**merged.get("adapter", {}), **experiment_config["adapter"]}
        
        # Merge pipeline settings (experiment config can override pipeline defaults)
        if "pipeline" in experiment_config:
            merged["pipeline"] = {**merged.get("pipeline", {}), **experiment_config["pipeline"]}
        
        # Add experiment metadata
        merged["experiment"] = experiment_config.get("experiment", {})
        merged["data"] = experiment_config.get("data", {})
        merged["model"] = experiment_config.get("model", {})
        merged["logging"] = experiment_config.get("logging", {})
        
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        print("🔍 Validating configuration...")
        
        # Check required sections
        required_sections = ["pipeline", "training"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate pipeline type
        pipeline_type = config["pipeline"]["type"]
        supported_pipeline_types = [
            "cogvideox_i2v",
            "cogvideox_pose_concat",
            "cogvideox_pose_adapter",
            "cogvideox_pose_adaln",
            "cogvideox_pose_adaln_perframe",
            "cogvideox_static_to_video",
            "cogvideox_static_to_video_pose_concat",
            "cogvideox_fun_static_to_video",
            "cogvideox_fun_static_to_video_pose_concat",
            "cogvideox_fun_static_to_video_posmap_concat",
            "cogvideox_fun_static_to_video_raymap_pose_concat",
            "cogvideox_fun_static_to_video_posmap_adapter",
            "cogvideox_fun_static_to_video_pose_adapter",
            "cogvideox_fun_static_to_video_pose_cond_token",
            "cogvideox_fun_static_to_video_cross_pose_adapter",
            "cogvideox_fun_static_to_video_pose_adaln",
            "cogvideox_fun_static_to_video_pose_adaln_perframe",
            "cogvideox_fun_static_to_video_joint_generation",
        ]
        if pipeline_type not in supported_pipeline_types:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}. Supported types: {supported_pipeline_types}")
        
        print("✅ Configuration validation passed")
        return True
    
    def apply_overrides(self, config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
        """Apply runtime overrides to configuration."""
        if not overrides:
            return config
            
        print("🔧 Applying runtime overrides...")
        
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: {override}. Use key=value")
            
            key, value = override.split("=", 1)
            
            # Parse nested keys (e.g., "training.learning_rate=1e-4")
            keys = key.split(".")
            current = config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            target_key = keys[-1]
            
            # Try to convert value to appropriate type
            try:
                # Try to convert to float
                if "." in value or "e" in value.lower():
                    current[target_key] = float(value)
                # Try to convert to int
                elif value.isdigit():
                    current[target_key] = int(value)
                # Try to convert to bool
                elif value.lower() in ["true", "false"]:
                    current[target_key] = value.lower() == "true"
                else:
                    current[target_key] = value
            except ValueError:
                current[target_key] = value
                
            print(f"   {key} = {current[target_key]}")
        
        return config
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print a summary of the configuration."""
        print("\n" + "="*80)
        print("🎯 EXPERIMENT CONFIGURATION SUMMARY")
        print("="*80)
        
        # Experiment info
        experiment = config.get("experiment", {})
        print(f"📝 Experiment: {experiment.get('name', 'Unknown')}")
        print(f"   Description: {experiment.get('description', 'No description')}")
        print(f"   Author: {experiment.get('author', 'Unknown')}")
        print(f"   Date: {experiment.get('date', 'Unknown')}")
        
        # Pipeline info
        pipeline = config.get("pipeline", {})
        print(f"\n🔧 Pipeline: {pipeline.get('type', 'Unknown')}")
        print(f"   Class: {pipeline.get('class', 'Unknown')}")
        
        # Training info
        training = config.get("training", {})
        print(f"\n🚀 Training Mode: {training.get('mode', 'Unknown')}")
        print(f"   Learning Rate: {training.get('learning_rate', 'Unknown')}")
        print(f"   Batch Size: {training.get('batch_size', 'Unknown')}")
        if 'max_train_steps' in training:
            print(f"   Max Train Steps: {training.get('max_train_steps', 'Unknown')}")
        else:
            print(f"   Epochs: {training.get('num_epochs', 'Unknown')}")
        
        # Adapter info (if applicable)
        if "adapter" in config:
            adapter = config["adapter"]
            print(f"\n🔗 Adapter Settings:")
            print(f"   Norm: {adapter.get('norm', 'Unknown')}")
            print(f"   Groups: {adapter.get('groups', 'Unknown')}")
            print(f"   Freeze Hand: {adapter.get('freeze_hand_branch', 'Unknown')}")
            print(f"   Freeze Static: {adapter.get('freeze_static_branch', 'Unknown')}")
        
        # Data info
        data = config.get("data", {})
        print(f"\n📊 Data:")
        print(f"   Dataset: {data.get('dataset_file', 'Unknown')}")
        print(f"   Validation: {data.get('validation_set', 'Unknown')}")
        print(f"   Data Root: {data.get('data_root', 'Unknown')}")
        
        # Model info
        model = config.get("model", {})
        print(f"\n🤖 Model:")
        print(f"   Base Model: {model.get('pretrained_model_name_or_path', 'Unknown')}")
        print(f"   Output Dir: {model.get('output_dir', 'Unknown')}")
        
        print("="*80 + "\n")


def load_experiment_config(experiment_file: str, overrides: Optional[list] = None) -> Dict[str, Any]:
    """
    Convenience function to load and validate experiment configuration.
    
    Args:
        experiment_file: Path to experiment YAML file
        overrides: List of runtime overrides (key=value format)
    
    Returns:
        Merged and validated configuration dictionary
    """
    config_loader = ExperimentConfigLoader()
    config = config_loader.load_experiment_config(experiment_file)
    
    if overrides:
        config = config_loader.apply_overrides(config, overrides)
    
    config_loader.print_config_summary(config)
    return config
