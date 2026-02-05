import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentConfigLoader:
    """
    YAML-based configuration loader for WAN training experiments.
    
    Supports:
    - Pipeline-specific configurations (base config)
    - Experiment-specific configurations (overrides)
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
        
        # Load and merge pipeline config if pipeline type is specified
        if "pipeline" in experiment_config and "type" in experiment_config["pipeline"]:
            pipeline_type = experiment_config["pipeline"]["type"]
            try:
                pipeline_config = self.load_pipeline_config(pipeline_type)
                # Merge configurations (pipeline as base, experiment overrides)
                merged_config = self._merge_configs(pipeline_config, experiment_config)
            except FileNotFoundError:
                print(f"⚠️  Pipeline config for '{pipeline_type}' not found, using experiment config only")
                merged_config = experiment_config
        else:
            merged_config = experiment_config
        
        # Validate configuration
        self.validate_config(merged_config)
        
        return merged_config
    
    def _merge_configs(self, pipeline_config: Dict, experiment_config: Dict) -> Dict:
        """
        Merge pipeline and experiment configurations.
        Pipeline config is the base, experiment config overrides.
        """
        merged = {}
        
        # Deep merge pipeline config first
        for key, value in pipeline_config.items():
            merged[key] = value
        
        # Override/merge with experiment config
        for key, value in experiment_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Deep merge for dict values
                merged[key] = {**merged[key], **value}
            else:
                # Direct override for non-dict values
                merged[key] = value
        
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        print("🔍 Validating configuration...")
        
        # Check required sections
        required_sections = ["experiment", "training"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate pipeline type if specified
        if "pipeline" in config and "type" in config["pipeline"]:
            pipeline_type = config["pipeline"]["type"]
            supported_pipeline_types = [
                # WAN pipelines
                "wan2.1_fun_inpaint",
                "wan2.1_fun_inp_hand_concat",
                "wan2.2_fun_inpaint",
                "wan2.2_fun_inp_hand_concat",
            ]
            if pipeline_type not in supported_pipeline_types:
                print(f"⚠️  Pipeline type '{pipeline_type}' not in predefined list, but continuing...")
        
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
        if pipeline:
            print(f"\n🔧 Pipeline: {pipeline.get('type', 'Unknown')}")
            print(f"   Class: {pipeline.get('class', 'Unknown')}")
            if 'base_model_name_or_path' in pipeline:
                print(f"   Base Model: {pipeline.get('base_model_name_or_path', 'Unknown')}")
        
        # Transformer info
        transformer = config.get("transformer", {})
        if transformer:
            print(f"\n🔩 Transformer:")
            print(f"   Class: {transformer.get('class', 'Unknown')}")
            if 'condition_channels' in transformer:
                print(f"   Condition Channels: {transformer.get('condition_channels', 'Unknown')}")
        
        # Training info
        training = config.get("training", {})
        print(f"\n🚀 Training Mode: {training.get('mode', 'Unknown')}")
        print(f"   Learning Rate: {training.get('learning_rate', 'Unknown')}")
        print(f"   Batch Size: {training.get('batch_size', 'Unknown')}")
        if 'max_train_steps' in training:
            print(f"   Max Train Steps: {training.get('max_train_steps', 'Unknown')}")
        if 'lora_rank' in training:
            print(f"   LoRA Rank: {training.get('lora_rank', 'Unknown')}")
        if 'lora_alpha' in training:
            print(f"   LoRA Alpha: {training.get('lora_alpha', 'Unknown')}")
        
        # Data info
        data = config.get("data", {})
        print(f"\n📊 Data:")
        print(f"   Dataset: {data.get('dataset_file', 'Unknown')}")
        print(f"   Data Root: {data.get('data_root', 'Unknown')}")
        
        print("="*80 + "\n")


def load_experiment_config(experiment_file: str, overrides: Optional[list] = None) -> Dict[str, Any]:
    """
    Convenience function to load and validate experiment configuration.
    
    Loads pipeline config (if specified) as base, then merges experiment config on top.
    
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
