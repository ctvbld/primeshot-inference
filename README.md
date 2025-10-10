# ComfyUI Photography Platform (Refactored)

A serverless ComfyUI-based image generation platform built following Modal's best practices. Creates high-quality, personalized photography using Flux.1-dev + custom LoRAs with S3 integration.

## ✨ Features

- **🚀 Modal Best Practices**: Built following the official Modal ComfyUI example structure
- **🎨 Flux.1-dev Integration**: High-quality image generation with FP8 optimization for L40S
- **🔧 Custom LoRA Support**: Dynamic loading of user-trained LoRAs from S3
- **📱 Dual Interface**: Development UI + Production API
- **💾 Persistent Caching**: Models cached in Modal volumes for fast cold starts
- **🔍 Health Monitoring**: Automatic server health checks and recovery
- **🎨 SUPIR Upscaling**: Professional 4K upscaling with denoising
- **🌟 Latent Interposer** ⚡: Experimental Flux→SDXL latent conversion
- **📦 Batch Processing**: Generate up to 15 images simultaneously
- **🖥️ Dual GPU Setup**: A10G for development, L40S for production
- **🔗 S3 Integration**: Seamless asset management and output delivery

## 🏗️ Architecture

```
ComfyUI Photography Platform (v2.0)
├── 🖥️ Development Server (A10G)
│   ├── ComfyUI Web UI on port 8000
│   ├── S3 LoRA auto-linking
│   └── Interactive workflow testing
├── ⚡ Production Class (L40S)
│   ├── Background ComfyUI server
│   ├── Flux + LoRA workflow execution
│   ├── Health monitoring & recovery
│   └── S3 result uploading
└── 🌐 API Endpoints
    ├── POST /api - Generate images
    ├── GET /job-status - Check status
    └── GET /health - Health check
```

## 🚀 Quick Start

### 1. Deploy to Modal

```bash
# Deploy the platform
modal deploy comfyui_app.py

# Start development server for workflow testing
modal run comfyui_app.py::dev_server
```

### 2. Access Development Interface

The development server provides a ComfyUI web interface where you can:
- ✅ Test Flux.1-dev workflows
- ✅ Load LoRAs from S3 automatically
- ✅ Create and export custom workflows
- ✅ Debug generation issues

### 3. Test the API

```bash
# Test with the provided client
python test_client.py

# Or use curl directly
curl -X POST "https://your-app.modal.run/api" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "lora_path": "path/to/lora.safetensors",
    "style_prompt": "professional portrait, studio lighting",
    "batch_size": 5,
    "resolution": "1024x1024"
  }'
curl -X POST "https://your-modal-app.modal.run/generate-images" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "lora_path": "s3://primeshot-uploads-01/user-images/user123/loras/portrait_lora.safetensors",
    "style_prompt": "professional headshot, studio lighting, high quality",
    "negative_prompt": "blurry, low quality, distorted",
    "batch_size": 15,
    "resolution": "1024x1024",
    "upscale_target": "4K"
  }'
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "submitted",
  "estimated_time": "120s",
  "cost_estimate": "$1.50"
}
```

#### Check Job Status

```bash
curl "https://your-modal-app.modal.run/job-status/your-job-id"
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 1.0,
  "images_completed": 15,
  "total_images": 15,
  "estimated_remaining": "0s",
  "output_urls": [
    "s3://primeshot-uploads-01/user-images/user123/generated/job-id/image_001.png",
    "..."
  ],
  "execution_time": 95.2,
  "cost_estimate": 1.50
}
```

## Configuration

### Node Bypass and Settings Override

The platform supports fine-grained control over ComfyUI nodes through the `settings_override` parameter. You can both modify node parameters and bypass entire nodes using special control keys.

#### Basic Parameter Override

Override specific node parameters by node title:

```json
{
  "settings_override": {
    "FilmGrain": {
      "grain_intensity": 0.1,
      "grain_size": 1.5
    },
    "VibSat": {
      "vibrance": 0.3,
      "saturation": 0.2
    }
  }
}
```

#### Bypassing Nodes

Bypass nodes to skip their processing using the `__bypass__` control key:

```json
{
  "settings_override": {
    "VibSat": {
      "__bypass__": true
    },
    "LightLeaks": {
      "__bypass__": true,
      "__passthrough_key__": "image"
    }
  }
}
```

**Bypass Control Keys:**
- `__bypass__`: Set to `true` to bypass this node (required)
- `__passthrough_key__`: Which input to forward through (optional, defaults to "image")
- `__output_index__`: Which output slot to replace (optional, defaults to 0)

#### Combined Settings and Bypass

You can combine parameter overrides and bypasses in the same configuration:

```json
{
  "settings_override": {
    "FilmGrain": {
      "grain_intensity": 0.15
    },
    "VibSat": {
      "__bypass__": true
    },
    "ChannelMixer": {
      "red_adjust": 1.1,
      "blue_adjust": 0.9
    }
  }
}
```

#### Multi-Output Node Bypass

For nodes with multiple outputs (like LoRA loaders), you may need multiple bypass operations:

```json
{
  "settings_override": {
    "StyleLora": {
      "__bypass__": true,
      "__passthrough_key__": "model",
      "__output_index__": 0
    }
  }
}
```

**Note:** Keys starting with `__` are reserved for control purposes and won't be applied as node parameters.

### Environment Variables

The platform uses these Modal secrets and volumes:

#### Modal Secrets

- **aws-secret**: AWS credentials for S3 access
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`

- **inference-secret**: Inference configuration
  - `COMFY_WORKFLOW_ENDPOINT` - ComfyUI API endpoint
  - `PROGRESS_WS_URL` - WebSocket URL for progress updates
  - `WEBHOOK_SECRET` - Secret for webhook authentication
  - `WEBHOOK_URL` - URL for webhook callbacks
  - **`WORKFLOW_VERSION`** - Workflow version to use (e.g., `1.2`, `1.3`)
    - Default: `1.2` if not set
    - Controls which workflow files are loaded: `V{version}_1K.json` and `V{version}.json`
    - Can be updated without redeploying the Modal app

- **supabase-secret**: Database credentials for multiple environments
  - `SUPABASE_URL_DEV`, `SUPABASE_SERVICE_ROLE_KEY_DEV`
  - `SUPABASE_URL_STAGING`, `SUPABASE_SERVICE_ROLE_KEY_STAGING`
  - `SUPABASE_URL_PROD`, `SUPABASE_SERVICE_ROLE_KEY_PROD`

#### Modal Volumes

- **models-vol**: Persistent volume for model storage

### S3 Bucket Structure

```
primeshot-uploads-01/
├── user-images/
│   └── {user_id}/
│       ├── source/           # Raw uploaded images
│       ├── loras/            # Trained LoRA models
│       └── generated/        # Generated images organized by job
│           └── {job_id}/     # Job-specific outputs
│               ├── image_001.png
│               ├── image_002.png
│               └── ...
```

### Updating Workflow Version

To update to a new workflow version without redeploying:

1. **Upload new workflow files to S3** (if using S3 workflow storage):
   ```bash
   # Upload new workflow files to S3
   aws s3 cp V1.3_1K.json s3://your-bucket/workflows/
   aws s3 cp V1.3.json s3://your-bucket/workflows/
   ```

2. **Update the Modal secret**:
   - Go to Modal Dashboard → Secrets
   - Open `inference-secret`
   - Update `WORKFLOW_VERSION` value (e.g., from `1.2` to `1.3`)
   - Save changes

3. **Verify the change**:
   - Monitor logs for: `📋 Using workflow version: 1.3`
   - Check that correct workflow files are loaded: `🎯 Using quality-selected workflow: V1.3_1K.json`

**Benefits:**
- ✅ No code deployment required
- ✅ Instant updates across all containers
- ✅ Easy rollback by reverting the environment variable
- ✅ Test new versions quickly

**Workflow File Naming:**
- For 1K quality: `V{WORKFLOW_VERSION}_1K.json`
- For 2K/4K quality: `V{WORKFLOW_VERSION}.json`

### Model Storage

Models are cached in the persistent Modal volume:

```
/models/
├── checkpoints/
│   └── flux1-dev/            # Flux.1-dev model
├── loras/                    # User LoRAs (symlinked from S3)
├── upscale_models/           # SUPIR models
├── clip/                     # CLIP text encoders
└── vae/                      # VAE models
```

## Workflow Development

### 1. Create Custom Workflows

Use the ComfyUI development interface to create workflows:

1. Start the dev server: `modal run comfyui_app.py::comfyui_dev_server`
2. Access the interface at the provided URL
3. Design your workflow with:
   - Flux.1-dev checkpoint loader
   - LoRA loader with S3 path
   - Text prompt encoding
   - KSampler for generation
   - SUPIR upscaler
   - Image save node

### 2. Export Workflows

Save workflows as JSON files in the ComfyUI interface, then copy them to the `scripts/` directory for production use.

### 3. Customize for Production

Update the workflow manager in `scripts/comfyui_server.py` to use your custom workflow:

```python
def create_custom_workflow(self, **params):
    # Your custom workflow logic
    return workflow_dict
```

## Performance Optimization

### Target Metrics

- **Generation Time**: <120 seconds for 15-image batch at 4K
- **Cold Start**: <60 seconds container initialization  
- **Cost**: $0.10 per image ($1.50 per 15-image batch)
- **Success Rate**: >99% successful generation rate

### GPU Configuration

- **Development**: A10G (~$0.30/hour) for ComfyUI UI
- **Production**: L40S (~$1.10/hour) for batch inference
- **Scaling**: 3 max containers with 5-minute scaledown

### Cost Optimization

- Persistent model caching to avoid redownloads
- Efficient batch processing 
- Auto-scaling based on demand
- S3 lifecycle policies for generated images

## Integration with Existing LoRA Training

This platform integrates seamlessly with the existing LoRA training service (`modal_app.py`):

1. **Training**: Use `modal_app.py` to train custom LoRAs
2. **Storage**: LoRAs are automatically saved to S3 user directories  
3. **Generation**: Reference LoRA S3 paths in generation requests
4. **Workflow**: Train → Generate → Download results

Example workflow:
```python
# 1. Train LoRA
lora_result = modal_app.main.remote({
    "user_id": "user123",
    "trigger_word": "TOK",
    # ... training params
})

# 2. Generate images with trained LoRA
generation_result = requests.post("https://comfyui-app.modal.run/generate-images", json={
    "user_id": "user123",
    "lora_path": f"s3://primeshot-uploads-01/user-images/user123/loras/{lora_result['lora']}",
    "style_prompt": "TOK professional portrait, studio lighting",
    # ... generation params
})
```

## API Reference

### POST /generate-images

Generate a batch of images using Flux.1-dev + LoRA + upscaling.

**Parameters:**
- `user_id` (required): User identifier for organizing outputs
- `lora_path` (required): S3 path to LoRA model
- `style_prompt` (required): Text prompt for generation
- `negative_prompt` (optional): Negative text prompt
- `batch_size` (optional): Number of images (default: 15)
- `resolution` (optional): Base resolution (default: "1024x1024")
- `upscale_target` (optional): Target upscale (default: "4K")
- `seed` (optional): Random seed (-1 for random)

### GET /job-status/{job_id}

Get the status of a generation job.

**Response Fields:**
- `job_id`: Unique job identifier
- `status`: "submitted" | "processing" | "completed" | "failed"
- `progress`: Float 0.0-1.0
- `images_completed`: Number of images finished
- `total_images`: Total images in batch
- `estimated_remaining`: Estimated time remaining
- `output_urls`: Array of S3 URLs for completed images
- `execution_time`: Total execution time in seconds
- `cost_estimate`: Estimated cost in USD

### GET /health

Health check endpoint.

## Troubleshooting

### Common Issues

1. **LoRA not found**: Ensure LoRA path is correct and accessible in S3
2. **Model download timeouts**: Models are cached after first download
3. **Out of memory**: Reduce batch size or resolution
4. **Generation timeouts**: Check ComfyUI logs in Modal dashboard

### Monitoring

- Check Modal dashboard for container logs
- Monitor job tracker storage for failed jobs
- Use health endpoint for uptime monitoring
- Track costs in Modal usage dashboard

## Development Roadmap & Task Status

### Phase 1: Foundation Setup ✅ COMPLETED
- [x] **ComfyUI Development Environment**
  - [x] Set up Modal function with A10G GPU for ComfyUI UI development
  - [x] Install ComfyUI with all required custom nodes
  - [x] Configure Flux.1-dev base model integration
  - [x] Configure S3 volume mounting for LoRA access
  - [x] SUPIR upscaling integration

- [x] **Base Infrastructure**
  - [x] Create separate Modal app for ComfyUI service
  - [x] Configure development vs production GPU environments (A10G/L40S)
  - [x] Set up model caching with persistent volumes
  - [x] Implement basic workflow testing framework

### Phase 2: Core Workflow Development ✅ COMPLETED  
- [x] **ComfyUI Workflow Creation**
  - [x] Design LoRA loading workflow from S3 paths
  - [x] Create batch image generation workflow (15 images)
  - [x] Implement 4K upscaling pipeline (SUPIR)
  - [x] Optimize for 15-image batch processing

- [x] **Python Conversion**
  - [x] Convert ComfyUI workflow to Python for production
  - [x] Optimize for L40S GPU performance
  - [x] Implement batch processing logic
  - [x] Add performance monitoring and logging

### Phase 3: API Development ✅ COMPLETED
- [x] **REST API Implementation**
  - [x] Implement `/generate-images` POST endpoint
  - [x] Create job queuing system with unique IDs
  - [x] Implement `/job-status/{job_id}` endpoint
  - [x] Add S3 integration for input/output handling
  - [x] User-based path organization (`user_id/generated/job_id/`)

- [x] **Error Handling & Optimization**
  - [x] Add comprehensive error handling
  - [x] Implement job tracking and status updates
  - [x] Cost monitoring and estimation ($0.10/image target)
  - [x] Performance framework for <120s target

### Phase 4: Integration & Testing ⚠️ PARTIALLY COMPLETED
- [ ] **React/NextJS Integration** ❌ NOT STARTED
  - [ ] Create TypeScript SDK for API consumption
  - [ ] Client-side job status polling
  - [ ] File upload integration for LoRAs
  - [ ] Error handling and user feedback

- [⚠️] **Production Deployment** ⚠️ PARTIALLY COMPLETED
  - [x] Documentation and deployment guides
  - [x] Test scripts and validation tools
  - [ ] Scale testing with concurrent requests
  - [ ] End-to-end testing with real LoRA workflows
  - [ ] Performance validation (<120s for 15 images)
  - [ ] Cost validation (<$0.10/image)

### Additional Completed Features ✅
- [x] **Job Tracking System**
  - [x] File-based job persistence
  - [x] Progress tracking and time estimation
  - [x] Status management (submitted/processing/completed/failed)

- [x] **Workflow Management**
  - [x] Dynamic workflow customization
  - [x] Template-based workflow generation
  - [x] Parameter injection system

- [x] **Development Tools**
  - [x] Deployment automation script
  - [x] Comprehensive test suite
  - [x] Health check endpoints

### Next Priority Tasks 🎯
1. **React/NextJS Integration**: Build the frontend client for the ComfyUI platform
2. **React/NextJS SDK**: Create TypeScript client library
3. **Performance Testing**: Validate <120s and <$0.10 targets
4. **Load Testing**: Test concurrent request handling
5. **Monitoring**: Advanced analytics and error reporting

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Test with Modal: `modal run comfyui_app.py::comfyui_dev_server`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@primeshot.ai
- 💬 Discord: [Primeshot Community](https://discord.gg/primeshot)
- 📖 Documentation: [docs.primeshot.ai](https://docs.primeshot.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/primeshot/comfyui-modal/issues) 

### Advanced Features

#### SD-Latent-Interposer Integration ⚡ **EXPERIMENTAL**

This platform includes experimental support for [SD-Latent-Interposer](https://github.com/city96/SD-Latent-Interposer), enabling direct Flux→SDXL latent conversion for SUPIR upscaling.

**Traditional Workflow**:
```
Flux → Flux Latents → Decode to Pixels → Encode to SDXL Latents → SUPIR
```

**Interposer Workflow**:
```
Flux → Flux Latents → Direct Neural Conversion → SDXL Latents → SUPIR
```

**Potential Benefits**:
- Reduced quality loss from pixel conversion
- Faster processing (skips decode/encode cycle)
- Better preservation of Flux generation characteristics

**Usage**:
```json
{
  "workflow_name": "flux_lora_supir_interposer",
  "parameters": {
    // ... same parameters as flux_lora_supir
  }
}
```

**Note**: This is experimental technology that may introduce artifacts. Use the standard `flux_lora_supir` workflow for production use. 