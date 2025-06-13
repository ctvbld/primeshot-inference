# ComfyUI Workflows

This directory contains ComfyUI workflow templates for AI image generation and upscaling.

## Available Workflows

### 1. Flux + LoRA (`flux_lora`)
Basic image generation using Flux.1-dev with custom LoRAs.

**Output**: Original resolution images (e.g., 1024x1024)

### 2. Flux + LoRA + SUPIR Upscaling (`flux_lora_supir`)
Enhanced workflow that generates images and then upscales them using SUPIR for superior quality.

**Output**: High-resolution upscaled images (e.g., 2048x2048 with 2x upscaling)

### 3. Flux + LoRA + ControlNet Upscaling (`flux_lora_controlnet_upscale`)
Native Flux upscaling using [Jasper AI's Flux.1-dev ControlNet Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler).

**Key Features**:
- **Flux-Native**: Specifically trained for Flux.1-dev output
- **High Quality**: Uses diffusion-based upscaling instead of traditional interpolation
- **4x Upscaling**: Produces 4096x4096 images from 1024x1024 input
- **ControlNet Architecture**: Maintains coherence with original Flux generation style

**Workflow Flow**:
1. Flux + LoRA generation → Initial image (e.g., 1024x1024)
2. Image scaling → Prepare control image (4x scaled)
3. **Flux ControlNet Upscaler** → High-quality 4K output

**Benefits**:
- Better quality than traditional upscalers for Flux images
- Maintains Flux's unique aesthetic characteristics
- Professional-grade upscaling using the same diffusion process

### 4. Flux + LoRA + Latent Interposer + SUPIR (`flux_lora_supir_interposer`) ⚡ **EXPERIMENTAL**
Experimental workflow using [SD-Latent-Interposer](https://github.com/city96/SD-Latent-Interposer) to convert Flux latents directly to SDXL latents for SUPIR processing.

**Key Innovation**: Instead of decoding Flux latents to pixels and then re-encoding to SDXL latents (lossy process), this workflow uses a neural network to convert latents directly, potentially preserving more image quality.

**Workflow Flow**:
1. Flux generation → Flux latents
2. **Latent Interposer** → SDXL-compatible latents  
3. SUPIR processing → Upscaled images

**Potential Benefits**:
- Reduced quality loss from decode/encode cycle
- Faster processing (no pixel conversion step)
- Better preservation of Flux generation characteristics

**Note**: This is experimental technology. The interposer may introduce artifacts or color shifts as mentioned in the [project documentation](https://github.com/city96/SD-Latent-Interposer#interposer-v40).

### 5. Pixelwave Flux 8x Upscale (`flux_lora_pixelwave_8x`) 🚀 **NEW**
Advanced 8x upscaling workflow using the specialized [Pixelwave Flux model](https://huggingface.co/mikeyandfriends/PixelWave_FLUX.1-dev_03) with TTP Tiling technology for ultra-realistic skin texture preservation.

**Key Features**:
- **8x Upscaling**: Produces ultra-high resolution images (e.g., 8192x8192 from 1024x1024)
- **Pixelwave Model**: Fine-tuned Flux model optimized for realistic skin textures
- **TTP Tiling**: Advanced tiling system prevents seams and artifacts
- **Quality Preservation**: Maintains fine details without quality loss

**Workflow Flow**:
1. Pixelwave Flux generation → High-quality base image
2. **TTP Tile Processing** → Image divided into optimal tiles
3. **Advanced Upscaling** → Each tile processed individually
4. **Seamless Assembly** → Tiles merged without visible boundaries

**Benefits**:
- Superior skin texture rendering
- No visible tile boundaries
- Extreme detail preservation
- Professional-grade 8K output

**Best For**: Portraits, fashion photography, detailed character art

## SUPIR Upscaling Features

- **High Quality**: Professional-grade upscaling using SUPIR-v0Q model
- **Denoising**: First-stage denoising for cleaner results
- **Tiled Processing**: Memory-efficient processing for large images
- **Configurable**: Adjustable upscale factor, steps, and guidance

## API Usage

### Basic Flux + LoRA Generation

```json
{
  "user_id": "user123",
  "workflow_name": "flux_lora",
  "parameters": {
    "lora_path": "/data/user123/loras/style.safetensors",
    "style_prompt": "professional portrait, high quality, sharp focus",
    "negative_prompt": "blurry, low quality, distorted",
    "batch_size": 4,
    "resolution": "1024x1024",
    "steps": 28,
    "cfg": 3.5,
    "lora_strength": 1.0
  }
}
```

### Flux + LoRA + SUPIR Upscaling

```json
{
  "user_id": "user123", 
  "workflow_name": "flux_lora_supir",
  "parameters": {
    "lora_path": "/data/user123/loras/style.safetensors",
    "style_prompt": "professional portrait, high quality, sharp focus",
    "negative_prompt": "blurry, low quality, distorted",
    "batch_size": 4,
    "resolution": "1024x1024",
    "steps": 28,
    "cfg": 3.5,
    "lora_strength": 1.0,
    "upscale_factor": 2.0,
    "supir_steps": 20,
    "supir_cfg": 4.0,
    "supir_denoise_strength": 50,
    "use_tiled_processing": true
  }
}
```

### Flux + LoRA + ControlNet Upscaling (Flux-Native)

```json
{
  "user_id": "user123", 
  "workflow_name": "flux_lora_controlnet_upscale",
  "parameters": {
    "lora_path": "/data/user123/loras/style.safetensors",
    "style_prompt": "professional portrait, high quality, sharp focus",
    "negative_prompt": "blurry, low quality, distorted",
    "batch_size": 1,
    "resolution": "1024x1024",
    "steps": 28,
    "upscale_steps": 28,
    "cfg": 3.5,
    "controlnet_strength": 0.6,
    "lora_strength": 1.0
  }
}
```

### Flux + LoRA + Latent Interposer + SUPIR (Experimental)

```json
{
  "user_id": "user123", 
  "workflow_name": "flux_lora_supir_interposer",
  "parameters": {
    "lora_path": "/data/user123/loras/style.safetensors",
    "style_prompt": "professional portrait, high quality, sharp focus",
    "negative_prompt": "blurry, low quality, distorted",
    "batch_size": 4,
    "resolution": "1024x1024",
    "steps": 28,
    "cfg": 3.5,
    "lora_strength": 1.0,
    "upscale_factor": 2.0,
    "supir_steps": 20,
    "supir_cfg": 4.0,
    "supir_denoise_strength": 50,
    "use_tiled_processing": true
  }
}
```

### Pixelwave Flux 8x Upscale (Ultra-Realistic)

```json
{
  "user_id": "user123", 
  "workflow_name": "flux_lora_pixelwave_8x",
  "parameters": {
    "style_prompt": "ultra realistic portrait, professional photography, sharp details, perfect skin texture",
    "negative_prompt": "blurry, low quality, distorted, artifacts, pixelated",
    "batch_size": 1,
    "resolution": "1024x1024",
    "steps": 28,
    "cfg": 3.5,
    "tile_width": 1024,
    "tile_height": 1024,
    "denoise": 0.35,
    "lora_path": "realistic_skin_v2.safetensors",
    "lora_strength": 0.8,
    "lora_clip_strength": 0.8
  }
}
```

## SUPIR Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `upscale_factor` | float | 2.0 | 1.0-4.0 | Upscaling multiplier |
| `supir_steps` | integer | 20 | 5-50 | SUPIR sampling steps |
| `supir_cfg` | float | 4.0 | 1.0-10.0 | SUPIR guidance scale |
| `supir_denoise_strength` | integer | 50 | 0-100 | First-stage denoising strength |
| `use_tiled_processing` | boolean | true | - | Enable tiled processing for memory efficiency |

## Pixelwave Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `tile_width` | integer | 1024 | 512-2048 | Width of each processing tile |
| `tile_height` | integer | 1024 | 512-2048 | Height of each processing tile |
| `denoise` | float | 0.35 | 0.1-1.0 | Denoising strength for upscaling tiles |
| `lora_path` | string | None | - | Path to LoRA model file (relative to /data/user_id/loras/) |
| `lora_strength` | float | 1.0 | 0.0-2.0 | LoRA model strength |
| `lora_clip_strength` | float | 1.0 | 0.0-2.0 | LoRA CLIP strength |

## Performance Considerations

### Memory Requirements
- **Basic Generation**: ~8GB VRAM for 1024x1024
- **SUPIR Upscaling**: ~12-16GB VRAM for 2x upscaling
- **Pixelwave 8x Upscale**: ~16-24GB VRAM for 8x upscaling
- **Tiled Processing**: Reduces memory usage for large images

### Processing Time
- **Basic Generation**: ~30-60 seconds for 4 images
- **With SUPIR Upscaling**: ~90-180 seconds for 4 upscaled images
- **Pixelwave 8x Upscale**: ~180-300 seconds for 1 ultra-high resolution image

### Quality vs Speed
- Higher `supir_steps` = Better quality, longer processing time
- Lower `supir_denoise_strength` = Faster processing, potentially less cleaning
- `use_tiled_processing` = Memory efficient but slightly slower

## Best Practices

1. **Start with lower resolution** (512x512 or 768x768) for faster testing
2. **Use tiled processing** for upscaling factors >2x
3. **Adjust denoise strength** based on input image quality
4. **Monitor batch size** - SUPIR workflows use more memory per image

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce batch size
- Enable tiled processing
- Lower base resolution

**Poor Upscaling Quality**
- Increase `supir_steps` (20-30)
- Adjust `supir_denoise_strength` (30-70)
- Check input image quality

**Slow Processing**
- Reduce `supir_steps` to 15-20
- Lower `supir_denoise_strength` to 30-40
- Use smaller batch sizes

## Workflow Discovery

Use the `/workflows` endpoint to discover available workflows and their parameters:

```bash
curl -X GET https://your-modal-app.modal.run/workflows
```

This returns the complete workflow configuration including:
- Available workflows
- Parameter definitions with types and defaults
- Validation rules (min/max values, options)
- Node mappings for workflow customization

## Adding New Workflows

🎉 **No Code Changes Required!** The system now uses **generic parameter injection**.

1. **Create workflow JSON file** in this directory (e.g., `controlnet_pose.json`)
2. **Add workflow definition** to `workflow_config.json`:
   ```json
   {
     "workflows": {
       "controlnet_pose": {
         "file": "controlnet_pose.json",
         "name": "ControlNet Pose Generation",
         "description": "Generate images using pose ControlNet",
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
           "style_prompt": {
             "type": "string",
             "required": true,
             "description": "Positive prompt"
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
           "style_prompt": {
             "node": "3",
             "input_key": "text"
           }
         }
       }
     }
   }
   ```
3. **Deploy immediately** - the generic system handles parameter injection automatically!

### Parameter Mapping Types

**Standard Mapping:**
```json
"param_name": {
  "node": "12",
  "input_key": "strength"
}
```

**Multi-Node Mapping:**
```json
"lora_strength": {
  "node": "2",
  "input_key": "strength_model",
  "secondary_mappings": [
    {
      "node": "2", 
      "input_key": "strength_clip"
    }
  ]
}
```

**Special Handlers:**
- `resolution` - Parses "1024x1024" into width/height
- `random_seed` - Generates random seed if value is -1
- `lora_file` - Handles LoRA file linking
- `job_output` - Sets job-specific output prefix

## File Structure

```
workflows/
├── workflow_config.json    # Configuration and metadata
├── flux_lora.json         # Flux + LoRA workflow template  
└── README.md              # This documentation
```

## Security Features

- **User Isolation**: LoRA paths are validated to ensure users can only access their own files
- **Parameter Validation**: All parameters are validated against their definitions
- **Dynamic Loading**: Only requested workflows and LoRAs are loaded
- **Path Sanitization**: File paths are checked for security vulnerabilities

## Performance Benefits

- **Faster Startup**: No bulk loading of all LoRAs
- **Lower Memory**: Only load what's needed per job
- **Scalable**: Performance doesn't degrade with user count
- **Clean State**: Automatic cleanup after each job 