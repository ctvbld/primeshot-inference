#!/usr/bin/env python3
"""
ComfyUI Workflow Configuration Generator

Automatically generates workflow configuration entries for workflow_config.json
from ComfyUI workflow JSON files.

Usage:
    python scripts/generate_workflow_config.py <workflow_file.json> <workflow_name>
    
Example:
    python scripts/generate_workflow_config.py workflows/flux_lora.json flux_lora
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Node type mappings for logical categorization
NODE_TYPE_MAPPINGS = {
    # Text/Prompts
    "CLIPTextEncode": "prompt",
    "CLIPTextEncodeSDXL": "prompt", 
    "CLIPTextEncodeFlux": "prompt",
    
    # Sampling/Generation
    "KSampler": "sampler",
    "KSamplerAdvanced": "sampler",
    "SamplerCustom": "sampler",
    
    # Latent/Image Size
    "EmptyLatentImage": "latent",
    "EmptySD3LatentImage": "latent",
    "LatentUpscale": "latent",
    
    # LoRA/Models
    "LoraLoader": "lora",
    "CheckpointLoaderSimple": "checkpoint",
    "UNETLoader": "model",
    
    # ControlNet
    "ControlNetLoader": "controlnet",
    "ControlNetApply": "controlnet_apply",
    "ControlNetApplyAdvanced": "controlnet_apply",
    
    # Upscaling
    "UpscaleModelLoader": "upscale",
    "ImageUpscaleWithModel": "upscale_apply",
    "ESRGAN_4x": "upscale_apply",
    
    # Image Operations
    "LoadImage": "image_input",
    "SaveImage": "image_output",
    "PreviewImage": "image_output",
    "ImageResize": "image_transform",
    "ImageScale": "image_transform",
    
    # VAE
    "VAEDecode": "vae",
    "VAEEncode": "vae",
    "VAELoader": "vae_loader",
    
    # Inpainting
    "InpaintModelConditioning": "inpaint",
    "MaskToImage": "mask",
    "ImageToMask": "mask"
}

# Input name to parameter name mappings
INPUT_TO_PARAM_NAME = {
    # Common inputs
    "text": "prompt",  # Will be refined to style_prompt/negative_prompt based on context
    "seed": "seed",
    "steps": "steps", 
    "cfg": "cfg",
    "width": "resolution",  # Will be combined with height
    "height": "resolution", # Will be combined with width
    "batch_size": "batch_size",
    "denoise": "denoising_strength",
    
    # LoRA specific
    "lora_name": "lora_path",
    "strength_model": "lora_strength",
    "strength_clip": "lora_strength",
    
    # ControlNet specific  
    "image": "control_image",  # Context dependent
    "strength": "strength",
    "start_percent": "guidance_start",
    "end_percent": "guidance_end",
    
    # Upscaling specific
    "model_name": "upscale_model",
    "tile": "tile_size",
    "upscale_factor": "upscale_factor",
    
    # Sampling specific
    "sampler_name": "sampler",
    "scheduler": "scheduler",
    
    # Model specific
    "ckpt_name": "checkpoint_model",
    "filename_prefix": "output_prefix"
}

# Type inference based on input names and values
def infer_parameter_type(input_name: str, input_value: Any) -> str:
    """Infer parameter type from input name and value."""
    
    # String parameters
    string_indicators = ["text", "ckpt_name", "lora_name", "model_name", "sampler_name", "scheduler", "image", "filename_prefix"]
    if any(indicator in input_name for indicator in string_indicators):
        return "string"
    
    # Boolean parameters  
    if isinstance(input_value, bool):
        return "boolean"
    
    # Integer parameters
    integer_indicators = ["seed", "steps", "width", "height", "batch_size", "tile", "upscale_factor"]
    if any(indicator in input_name for indicator in integer_indicators):
        return "integer"
    
    # Float parameters
    float_indicators = ["cfg", "denoise", "strength", "start_percent", "end_percent"]
    if any(indicator in input_name for indicator in float_indicators) or isinstance(input_value, float):
        return "float"
    
    # Default to string for unknown types
    return "string"

def generate_parameter_description(param_name: str, param_type: str, node_type: str) -> str:
    """Generate a human-readable description for a parameter."""
    
    descriptions = {
        "style_prompt": "Positive prompt describing the desired image",
        "negative_prompt": "Negative prompt to avoid unwanted elements",
        "seed": "Random seed for generation (-1 for random)",
        "steps": "Number of denoising steps",
        "cfg": "Classifier-free guidance scale",
        "resolution": "Image resolution (width x height)",
        "batch_size": "Number of images to generate",
        "denoising_strength": "Denoising strength for image-to-image",
        "lora_path": "Path to LoRA file",
        "lora_strength": "LoRA influence strength",
        "controlnet_strength": "ControlNet influence strength",
        "control_image": "Control image for ControlNet",
        "guidance_start": "Guidance start percentage",
        "guidance_end": "Guidance end percentage",
        "upscale_model": "Upscaling model to use",
        "tile_size": "Tile size for processing",
        "sampler": "Sampling method",
        "scheduler": "Noise scheduler",
        "checkpoint_model": "Main model checkpoint",
        "output_prefix": "Output filename prefix"
    }
    
    return descriptions.get(param_name, f"{param_name.replace('_', ' ').title()} parameter")

def get_parameter_constraints(param_name: str, param_type: str, input_value: Any) -> Dict[str, Any]:
    """Get parameter constraints (min, max, options, etc.)."""
    constraints = {}
    
    # Common constraints based on parameter name
    constraint_mappings = {
        "steps": {"min": 1, "max": 50},
        "cfg": {"min": 1.0, "max": 10.0},
        "batch_size": {"min": 1, "max": 15},
        "denoising_strength": {"min": 0.0, "max": 1.0},
        "lora_strength": {"min": 0.0, "max": 2.0},
        "controlnet_strength": {"min": 0.0, "max": 2.0},
        "guidance_start": {"min": 0.0, "max": 1.0},
        "guidance_end": {"min": 0.0, "max": 1.0}
    }
    
    if param_name in constraint_mappings:
        constraints.update(constraint_mappings[param_name])
    
    # Set default based on input value
    if input_value is not None and not isinstance(input_value, list):
        constraints["default"] = input_value
    elif param_name == "seed":
        constraints["default"] = -1
    elif param_name == "resolution":
        constraints["default"] = "1024x1024"
        constraints["options"] = ["512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"]
    
    return constraints

def analyze_workflow(workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze workflow and extract parameter mappings."""
    
    parameters = {}
    parameter_mappings = {}
    node_mappings = {}
    
    # Track resolution parameters (width/height) for special handling
    resolution_nodes = {}
    lora_nodes = {}
    prompt_nodes = {"positive": [], "negative": []}
    
    # Analyze each node
    for node_id, node_data in workflow_data.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        meta = node_data.get("_meta", {})
        title = meta.get("title", "")
        
        # Map to logical category
        logical_category = NODE_TYPE_MAPPINGS.get(class_type, "unknown")
        if logical_category != "unknown":
            node_mappings[logical_category] = node_id
        
        # Analyze inputs
        for input_name, input_value in inputs.items():
            # Skip connection inputs (lists with node references)
            if isinstance(input_value, list):
                continue
                
            # Generate parameter name
            param_name = INPUT_TO_PARAM_NAME.get(input_name, input_name)
            
            # Special handling for prompts (positive vs negative)
            if input_name == "text" and class_type == "CLIPTextEncode":
                if "negative" in title.lower() or "negative" in str(input_value).lower():
                    param_name = "negative_prompt"
                    prompt_nodes["negative"].append(node_id)
                else:
                    param_name = "style_prompt"
                    prompt_nodes["positive"].append(node_id)
            
            # Special handling for resolution
            if input_name in ["width", "height"]:
                if node_id not in resolution_nodes:
                    resolution_nodes[node_id] = {}
                resolution_nodes[node_id][input_name] = input_value
                continue
            
            # Special handling for LoRA
            if input_name == "lora_name":
                param_name = "lora_path"
                lora_nodes[node_id] = True
                
            # Generate parameter definition
            param_type = infer_parameter_type(input_name, input_value)
            param_desc = generate_parameter_description(param_name, param_type, class_type)
            param_constraints = get_parameter_constraints(param_name, param_type, input_value)
            
            # Check if parameter already exists (for secondary mappings)
            if param_name in parameters:
                # Add secondary mapping for existing parameter
                if "secondary_mappings" not in parameter_mappings[param_name]:
                    parameter_mappings[param_name]["secondary_mappings"] = []
                parameter_mappings[param_name]["secondary_mappings"].append({
                    "node": node_id,
                    "input_key": input_name
                })
            else:
                # Add new parameter
                parameters[param_name] = {
                    "type": param_type,
                    "description": param_desc,
                    **param_constraints
                }
                
                # Add parameter mapping
                mapping = {
                    "node": node_id,
                    "input_key": input_name
                }
                
                # Add special handlers
                if param_name == "seed":
                    mapping["special_handler"] = "random_seed"
                elif param_name == "lora_path":
                    mapping["special_handler"] = "lora_file"
                elif input_name == "filename_prefix":
                    mapping["special_handler"] = "job_output"
                
                parameter_mappings[param_name] = mapping
    
    # Handle resolution special case
    if resolution_nodes:
        # Find the best resolution node (one with both width and height)
        resolution_node = None
        for node_id, dimensions in resolution_nodes.items():
            if "width" in dimensions and "height" in dimensions:
                resolution_node = node_id
                break
        
        if resolution_node:
            dimensions = resolution_nodes[resolution_node]
            parameters["resolution"] = {
                "type": "string",
                "default": f"{dimensions.get('width', 1024)}x{dimensions.get('height', 1024)}",
                "options": ["512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"],
                "description": "Image resolution (width x height)"
            }
            
            parameter_mappings["resolution"] = {
                "special_handler": "resolution",
                "width_mapping": {
                    "node": resolution_node,
                    "input_key": "width"
                },
                "height_mapping": {
                    "node": resolution_node,
                    "input_key": "height"
                }
            }
    
    # Ensure output prefix is included
    if "output_prefix" not in parameters:
        parameters["output_prefix"] = {
            "type": "string",
            "description": "Output filename prefix (automatically set per job)"
        }
        parameter_mappings["output_prefix"] = {
            "special_handler": "job_output",
            "node": "8",  # Common SaveImage node
            "input_key": "filename_prefix"
        }
    
    return {
        "parameters": parameters,
        "parameter_mappings": parameter_mappings,
        "node_mappings": node_mappings
    }

def generate_workflow_config(workflow_file: str, workflow_name: str) -> Dict[str, Any]:
    """Generate complete workflow configuration."""
    
    # Load workflow
    workflow_path = Path(workflow_file)
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_file}")
    
    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)
    
    # Analyze workflow
    analysis = analyze_workflow(workflow_data)
    
    # Generate configuration
    config = {
        workflow_name: {
            "file": workflow_path.name,
            "name": workflow_name.replace('_', ' ').title(),
            "description": f"Generated workflow configuration for {workflow_name}",
            **analysis
        }
    }
    
    return config

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_workflow_config.py <workflow_file.json> <workflow_name>")
        print("Example: python generate_workflow_config.py workflows/flux_lora.json flux_lora")
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    workflow_name = sys.argv[2]
    
    try:
        config = generate_workflow_config(workflow_file, workflow_name)
        
        print("Generated workflow configuration:")
        print("=" * 50)
        print(json.dumps(config, indent=2))
        print("=" * 50)
        print(f"\nTo add this workflow:")
        print(f"1. Copy the above JSON into workflows/workflow_config.json")
        print(f"2. Make sure {workflow_file} is in the workflows/ directory")
        print(f"3. Deploy the updated configuration")
        
    except Exception as e:
        print(f"Error generating configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 