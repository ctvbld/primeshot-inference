# Generic Workflow System Examples

The new generic parameter injection system can handle **any workflow** with **any custom nodes** without requiring Python code changes.

## Example 1: ControlNet Pose Workflow

### Workflow Configuration
```json
{
  "workflows": {
    "controlnet_pose": {
      "file": "controlnet_pose.json",
      "name": "ControlNet Pose Generation",
      "description": "Generate images using pose reference with ControlNet",
      "parameters": {
        "control_image": {
          "type": "string",
          "required": true,
          "description": "Path to pose reference image"
        },
        "controlnet_strength": {
          "type": "float", 
          "default": 0.8,
          "min": 0.0,
          "max": 2.0,
          "description": "ControlNet influence strength"
        },
        "guidance_start": {
          "type": "float",
          "default": 0.0,
          "min": 0.0,
          "max": 1.0,
          "description": "Guidance start step"
        },
        "guidance_end": {
          "type": "float",
          "default": 1.0,
          "min": 0.0,
          "max": 1.0,
          "description": "Guidance end step"
        },
        "style_prompt": {
          "type": "string",
          "required": true,
          "description": "Positive prompt"
        },
        "resolution": {
          "type": "string",
          "default": "768x768",
          "options": ["512x512", "768x768", "1024x1024"],
          "description": "Image resolution"
        }
      },
      "parameter_mappings": {
        "control_image": {
          "node": "15",
          "input_key": "image"
        },
        "controlnet_strength": {
          "node": "15",
          "input_key": "strength"
        },
        "guidance_start": {
          "node": "15",
          "input_key": "start_percent"
        },
        "guidance_end": {
          "node": "15",
          "input_key": "end_percent"
        },
        "style_prompt": {
          "node": "6",
          "input_key": "text"
        },
        "resolution": {
          "special_handler": "resolution",
          "width_mapping": {
            "node": "5",
            "input_key": "width"
          },
          "height_mapping": {
            "node": "5", 
            "input_key": "height"
          }
        }
      }
    }
  }
}
```

### API Usage
```json
{
  "workflow_name": "controlnet_pose",
  "user_id": "user_123",
  "parameters": {
    "control_image": "/data/user_123/poses/dancing_pose.jpg",
    "controlnet_strength": 0.9,
    "guidance_start": 0.0,
    "guidance_end": 0.8,
    "style_prompt": "elegant dancer in flowing dress, studio photography, dramatic lighting",
    "resolution": "768x768"
  }
}
```

## Example 2: Custom Upscaler Workflow

### Workflow Configuration
```json
{
  "workflows": {
    "esrgan_upscale": {
      "file": "esrgan_upscale.json",
      "name": "ESRGAN 4x Upscaling",
      "description": "Upscale images using ESRGAN with face enhancement",
      "parameters": {
        "input_image": {
          "type": "string",
          "required": true,
          "description": "Path to input image"
        },
        "upscale_model": {
          "type": "string",
          "default": "RealESRGAN_x4plus.pth",
          "options": ["RealESRGAN_x4plus.pth", "ESRGAN_4x.pth", "RealESRNet_x4plus.pth"],
          "description": "Upscale model to use"
        },
        "tile_size": {
          "type": "integer",
          "default": 512,
          "min": 128,
          "max": 1024,
          "description": "Tile size for processing"
        },
        "face_enhance": {
          "type": "boolean",
          "default": true,
          "description": "Enable face enhancement"
        },
        "face_visibility": {
          "type": "float",
          "default": 0.85,
          "min": 0.0,
          "max": 1.0,
          "description": "Face enhancement visibility"
        }
      },
      "parameter_mappings": {
        "input_image": {
          "node": "1",
          "input_key": "image"
        },
        "upscale_model": {
          "node": "2",
          "input_key": "model_name"
        },
        "tile_size": {
          "node": "2",
          "input_key": "tile"
        },
        "face_enhance": {
          "node": "3",
          "input_key": "enabled"
        },
        "face_visibility": {
          "node": "3",
          "input_key": "visibility"
        }
      }
    }
  }
}
```

### API Usage
```json
{
  "workflow_name": "esrgan_upscale",
  "user_id": "user_123", 
  "parameters": {
    "input_image": "/data/user_123/photos/portrait.jpg",
    "upscale_model": "RealESRGAN_x4plus.pth",
    "tile_size": 256,
    "face_enhance": true,
    "face_visibility": 0.9
  }
}
```

## Example 3: Multi-Stage Inpainting Workflow

### Workflow Configuration
```json
{
  "workflows": {
    "advanced_inpaint": {
      "file": "advanced_inpaint.json",
      "name": "Advanced Inpainting",
      "description": "Multi-stage inpainting with object removal and seamless blending",
      "parameters": {
        "input_image": {
          "type": "string",
          "required": true,
          "description": "Original image path"
        },
        "mask_image": {
          "type": "string", 
          "required": true,
          "description": "Mask image path"
        },
        "inpaint_prompt": {
          "type": "string",
          "required": true,
          "description": "What to inpaint"
        },
        "mask_blur": {
          "type": "integer",
          "default": 4,
          "min": 0,
          "max": 20,
          "description": "Mask blur radius"
        },
        "denoising_strength": {
          "type": "float",
          "default": 0.75,
          "min": 0.1,
          "max": 1.0,
          "description": "Denoising strength"
        },
        "inpaint_area": {
          "type": "string",
          "default": "masked_only",
          "options": ["masked_only", "whole_picture"],
          "description": "Inpainting area"
        },
        "blend_mode": {
          "type": "string",
          "default": "seamless",
          "options": ["seamless", "additive", "multiply"],
          "description": "Blending mode"
        }
      },
      "parameter_mappings": {
        "input_image": {
          "node": "10",
          "input_key": "image"
        },
        "mask_image": {
          "node": "11", 
          "input_key": "image"
        },
        "inpaint_prompt": {
          "node": "6",
          "input_key": "text"
        },
        "mask_blur": {
          "node": "12",
          "input_key": "blur_radius"
        },
        "denoising_strength": {
          "node": "13",
          "input_key": "denoise"
        },
        "inpaint_area": {
          "node": "13",
          "input_key": "inpaint_area"
        },
        "blend_mode": {
          "node": "14",
          "input_key": "blend_mode"
        }
      }
    }
  }
}
```

### API Usage
```json
{
  "workflow_name": "advanced_inpaint",
  "user_id": "user_123",
  "parameters": {
    "input_image": "/data/user_123/photos/beach_scene.jpg",
    "mask_image": "/data/user_123/masks/remove_person.png", 
    "inpaint_prompt": "empty beach with clear blue water",
    "mask_blur": 6,
    "denoising_strength": 0.8,
    "inpaint_area": "masked_only",
    "blend_mode": "seamless"
  }
}
```

## Key Benefits

🎯 **Zero Code Changes**: Add any workflow by editing JSON only
🔧 **Any Custom Nodes**: Works with any ComfyUI custom nodes
📝 **Full Validation**: Parameter types, ranges, and options are validated
🔒 **Security Built-in**: User isolation and path validation
⚡ **Performance**: Only load what's needed per job
📚 **Self-Documenting**: Complete parameter documentation in config

## Adding Your Own Workflow

1. **Export your ComfyUI workflow** as JSON
2. **Add configuration** to `workflow_config.json` with parameter mappings
3. **Deploy immediately** - no code changes needed!

The generic system handles:
- Parameter validation
- Type conversion
- Default values
- Multi-node parameter injection
- Special handlers (resolution, seeds, etc.)
- User security and isolation 