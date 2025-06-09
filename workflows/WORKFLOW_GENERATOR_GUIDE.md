# ComfyUI Workflow Configuration Generator Guide

## Overview

The workflow configuration generator automatically creates workflow configurations for `workflow_config.json` from ComfyUI workflow JSON files. This eliminates the need to manually map parameters and speeds up workflow integration.

## Quick Start

### 1. Generate Configuration from Workflow File

```bash
python scripts/generate_workflow_config.py workflows/your_workflow.json workflow_name
```

Example:
```bash
python scripts/generate_workflow_config.py workflows/controlnet_canny.json controlnet_canny
```

### 2. Copy Generated Configuration

The script outputs a complete workflow configuration that you can copy directly into `workflows/workflow_config.json`.

### 3. Deploy

Deploy your updated configuration to make the new workflow available via API.

## What the Generator Does

### Automatic Analysis
- **Node Recognition**: Identifies ComfyUI node types (KSampler, LoraLoader, CLIPTextEncode, etc.)
- **Parameter Extraction**: Finds all configurable inputs in each node
- **Type Inference**: Determines parameter types (string, integer, float, boolean)
- **Logical Mapping**: Creates human-readable parameter names

### Smart Features
- **Resolution Handling**: Combines width/height into single resolution parameter
- **LoRA Processing**: Automatically sets up LoRA file linking
- **Prompt Detection**: Distinguishes positive/negative prompts
- **Multi-Node Mapping**: Links parameters that affect multiple nodes (like LoRA strength)
- **Special Handlers**: Adds appropriate special handling for seeds, output files, etc.

### Generated Structure
- **Parameter Definitions**: Type, constraints, defaults, descriptions
- **Parameter Mappings**: How parameters map to workflow nodes
- **Node Mappings**: Logical names for workflow components
- **Validation Rules**: Min/max values and options where appropriate

## Example Output

For a ControlNet workflow, the generator might produce:

```json
{
  "controlnet_canny": {
    "file": "controlnet_canny.json",
    "name": "Controlnet Canny",
    "description": "Generated workflow configuration for controlnet_canny",
    "parameters": {
      "style_prompt": {
        "type": "string",
        "description": "Positive prompt describing the desired image",
        "default": "high quality photo"
      },
      "control_image": {
        "type": "string", 
        "description": "Control image for ControlNet",
        "required": true
      },
      "controlnet_strength": {
        "type": "float",
        "description": "ControlNet influence strength",
        "min": 0.0,
        "max": 2.0,
        "default": 1.0
      },
      "resolution": {
        "type": "string",
        "default": "1024x1024",
        "options": ["512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"],
        "description": "Image resolution (width x height)"
      }
    },
    "parameter_mappings": {
      "style_prompt": {
        "node": "6",
        "input_key": "text"
      },
      "control_image": {
        "node": "9",
        "input_key": "image"
      },
      "controlnet_strength": {
        "node": "10",
        "input_key": "strength"
      },
      "resolution": {
        "special_handler": "resolution",
        "width_mapping": {"node": "5", "input_key": "width"},
        "height_mapping": {"node": "5", "input_key": "height"}
      }
    },
    "node_mappings": {
      "checkpoint": "4",
      "latent": "5", 
      "prompt": "6",
      "sampler": "3",
      "controlnet": "10"
    }
  }
}
```

## Workflow Types Supported

### Standard Workflows
- **Text-to-Image**: Basic generation with prompts
- **LoRA Integration**: Custom model fine-tuning
- **ControlNet**: Structure-guided generation
- **Upscaling**: Image enhancement and scaling
- **Inpainting**: Area-specific image editing

### Advanced Workflows  
- **Multi-ControlNet**: Multiple control inputs
- **Custom Nodes**: Any ComfyUI extension nodes
- **Complex Pipelines**: Multi-stage processing workflows
- **Batch Processing**: Multiple image generation

## Parameter Name Mapping

The generator intelligently maps ComfyUI input names to user-friendly parameter names:

| ComfyUI Input | Generated Parameter | Description |
|---------------|-------------------|-------------|
| `text` | `style_prompt` / `negative_prompt` | Context-aware prompt naming |
| `seed` | `seed` | Random seed with automatic -1 handling |
| `steps` | `steps` | Denoising steps with reasonable limits |
| `cfg` | `cfg` | Guidance scale with typical ranges |
| `width`, `height` | `resolution` | Combined into single resolution parameter |
| `lora_name` | `lora_path` | LoRA file with path validation |
| `strength_model`, `strength_clip` | `lora_strength` | Combined LoRA strength |
| `sampler_name` | `sampler` | Sampling method |
| `filename_prefix` | `output_prefix` | Job-specific output naming |

## Special Handlers

The generator automatically detects and configures special handlers:

### Resolution Handler
```json
"resolution": {
  "special_handler": "resolution",
  "width_mapping": {"node": "5", "input_key": "width"},
  "height_mapping": {"node": "5", "input_key": "height"}
}
```

### Random Seed Handler
```json
"seed": {
  "special_handler": "random_seed",
  "node": "6",
  "input_key": "seed"
}
```

### LoRA File Handler
```json
"lora_path": {
  "special_handler": "lora_file", 
  "node": "2",
  "input_key": "lora_name"
}
```

### Job Output Handler
```json
"output_prefix": {
  "special_handler": "job_output",
  "node": "8", 
  "input_key": "filename_prefix"
}
```

## Common Workflows

### Adding a ControlNet Workflow

1. **Export from ComfyUI**:
   - Create your ControlNet workflow in ComfyUI
   - Use "Save (API Format)" to export as JSON
   - Place in `workflows/` directory

2. **Generate Configuration**:
   ```bash
   python scripts/generate_workflow_config.py workflows/controlnet_canny.json controlnet_canny
   ```

3. **Review and Customize**:
   - Check parameter names and descriptions
   - Adjust min/max values if needed
   - Add any missing required parameters

4. **Add to Config**:
   - Copy generated JSON into `workflow_config.json`
   - Update the workflows object

### Adding an Upscaling Workflow

1. **Create Upscaling Workflow**:
   - Add ESRGAN or other upscaling nodes
   - Configure input/output connections
   - Export as JSON

2. **Generate and Customize**:
   ```bash
   python scripts/generate_workflow_config.py workflows/upscale_4x.json upscale_4x
   ```

3. **Test Parameters**:
   - Verify upscale factor ranges
   - Check tile size limits
   - Validate model names

## Customization After Generation

### Parameter Refinement
```json
"steps": {
  "type": "integer",
  "default": 20,        // Adjust default
  "min": 5,            // Lower minimum  
  "max": 100,          // Higher maximum
  "description": "Number of denoising steps (more = higher quality, slower)"
}
```

### Adding Options
```json
"sampler": {
  "type": "string",
  "default": "euler",
  "options": ["euler", "euler_ancestral", "heun", "dpm_2", "lms"],
  "description": "Sampling algorithm"
}
```

### Required Parameters
```json
"control_image": {
  "type": "string",
  "required": true,     // Mark as required
  "description": "Input image for ControlNet guidance"
}
```

## Validation and Testing

### Generated Config Validation
```bash
# Test workflow configuration
curl -X GET http://your-app.modal.run/workflows
```

### Parameter Testing  
```bash
# Test API with new workflow
curl -X POST http://your-app.modal.run/api \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "your_new_workflow",
    "user_id": "test_user",
    "parameters": {
      "style_prompt": "test image",
      "resolution": "1024x1024"
    }
  }'
```

## Best Practices

### 1. Workflow Design
- Keep workflows focused on single tasks
- Use clear, descriptive node titles
- Minimize complex node connections
- Test workflows in ComfyUI first

### 2. Parameter Design
- Use intuitive parameter names
- Set reasonable defaults and limits
- Group related parameters logically
- Add helpful descriptions

### 3. Configuration Review
- Always review generated configs
- Test with various parameter values
- Validate required vs optional parameters
- Check special handler assignments

### 4. Documentation
- Document custom parameters
- Explain workflow purpose and use cases
- Provide example API calls
- Note any special requirements

## Troubleshooting

### Common Issues

**Generator doesn't detect parameters**:
- Check if workflow uses unsupported node types
- Verify JSON format is valid
- Ensure inputs aren't all connections (no literal values)

**Wrong parameter types**:
- Review and manually adjust type assignments
- Check value ranges and constraints
- Update descriptions for clarity

**Missing special handlers**:
- Manually add handlers for resolution, seeds, LoRA files
- Check node mappings are correct
- Verify handler implementation exists in code

**API validation errors**:
- Ensure required parameters are marked correctly
- Check parameter constraints match usage
- Validate node IDs exist in workflow

### Manual Fixes

If the generator misses something, you can manually edit the configuration:

```json
{
  "your_workflow": {
    // Add missing parameters
    "parameters": {
      "custom_param": {
        "type": "float",
        "default": 1.5,
        "min": 0.1,
        "max": 3.0,
        "description": "Custom parameter for special effect"
      }
    },
    // Add missing mappings  
    "parameter_mappings": {
      "custom_param": {
        "node": "15",
        "input_key": "strength"
      }
    }
  }
}
```

## Future Enhancements

The generator will be enhanced to support:
- **Conditional Parameters**: Parameters that only apply when others are set
- **Dynamic Options**: Parameter options based on available models/files
- **Workflow Validation**: Checking workflow integrity and node compatibility
- **Interactive Configuration**: Web UI for workflow configuration
- **Template Library**: Pre-built configurations for common workflow patterns 