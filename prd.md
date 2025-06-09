# Product Requirements Document (PRD)
## ComfyUI Serverless Photography Enhancement Platform

---

## 1. Executive Summary

**Product Vision**: A serverless ComfyUI-based image generation platform that creates high-quality, personalized photography using custom LoRAs and advanced upscaling, integrated with a React/NextJS web application.

**Business Objective**: Enable users to generate personalized, professional-quality images at scale through an internal API service that supports custom LoRA-based style transfer and SUPIR upscaling.

---

## 2. Product Overview

### 2.1 Core Value Proposition
- **Personalized Photography**: Generate custom images using user-specific LoRAs trained on uploaded photos
- **Premium Quality**: 4K output with SUPIR upscaling for professional-grade results  
- **Cost-Effective Scaling**: Serverless architecture optimized for $0.10/image target cost
- **Developer-Friendly**: Dual interface supporting both UI development and Python API production workflows

### 2.2 Target Architecture
Based on [Modal's ComfyUI example](https://modal.com/docs/examples/comfyapp), we'll implement:
- **Development Environment**: Interactive ComfyUI interface for workflow creation and testing
- **Production Environment**: Python-converted workflows for optimized API performance
- **Hybrid GPU Strategy**: Cost-effective GPUs for UI, high-performance GPUs for production inference

---

## 3. Functional Requirements

### 3.1 Core Features

#### 3.1.1 Image Generation Pipeline
- **Base Model**: Flux.1-dev integration for high-quality image generation
- **LoRA Integration**: Dynamic loading of custom LoRAs from S3-mounted storage
- **Upscaling**: SUPIR integration for 4K output enhancement
- **Batch Processing**: Support for 10-20 image batches per request

#### 3.1.2 Development Interface
- **Interactive ComfyUI**: Web-based workflow editor for development and testing
- **Workflow Export**: API-format JSON export capability
- **Real-time Preview**: Live workflow testing with immediate feedback

#### 3.1.3 Production API
- **REST Endpoints**: Clean API interface for React/NextJS integration
- **Python Workflow Execution**: Converted workflows as per [Modal's blog post](https://modal.com/blog/comfyui-prototype-to-production)
- **S3 Integration**: Direct access to user-uploaded LoRAs and generated images

### 3.2 API Specifications

#### 3.2.1 Primary Endpoint: `/generate-images`
```json
{
  "method": "POST",
  "payload": {
    "lora_path": "s3://bucket/user123/lora_model.safetensors",
    "style_prompt": "professional headshot, studio lighting",
    "negative_prompt": "blurry, low quality, distorted",
    "batch_size": 15,
    "resolution": "1024x1024",
    "upscale_target": "4K"
  },
  "response": {
    "job_id": "uuid",
    "estimated_time": "120s",
    "cost_estimate": "$1.20"
  }
}
```

#### 3.2.2 Status Endpoint: `/job-status/{job_id}`
```json
{
  "status": "processing|completed|failed",
  "progress": 0.75,
  "images_completed": 12,
  "total_images": 15,
  "download_urls": ["s3://output/..."]
}
```

---

## 4. Technical Requirements

### 4.1 Infrastructure Architecture

#### 4.1.1 Modal.com Configuration
```python
# Development Environment
@app.function(
    gpu="A10G",  # Cost-effective for UI development
    max_containers=1,
    volumes={"/cache": vol, "/s3-mount": s3_vol}
)

# Production Environment  
@app.cls(
    gpu="L40S",  # High-performance for batch inference
    scaledown_window=300,
    volumes={"/cache": vol, "/s3-mount": s3_vol}
)
```

#### 4.1.2 Model Management
- **Base Models**: Flux.1-dev cached in persistent Modal volumes
- **Custom Nodes**: SUPIR upscaler, LoRA loading nodes
- **S3 Integration**: Dynamic LoRA mounting from user uploads

### 4.2 Workflow Architecture

#### 4.2.1 Core Workflow Steps
1. **Input Processing**: Load user-specific LoRA from S3 path
2. **Base Generation**: Flux.1-dev generates initial images (1024x1024)
3. **Batch Processing**: Generate 10-20 images per request
4. **Upscaling**: SUPIR enhancement to 4K resolution
5. **Output Management**: Save to S3, return download URLs

#### 4.2.2 Performance Optimization
- **Model Caching**: Persistent volumes for model storage
- **Container Warm-up**: 5-minute scaledown window for batch requests
- **Concurrent Processing**: No Support for concurrent batches per container, 1 batch per container

---

## 5. Performance & Cost Requirements

### 5.1 Performance Targets
- **Generation Time**: <120 seconds for 15-image batch at 4K
- **Cold Start**: <60 seconds container initialization
- **Throughput**: Support for 5000-10000 images/month at scale
- **Concurrent Users**: Handle multiple simultaneous batch requests

### 5.2 Cost Optimization
- **Target Cost**: $0.10 per image ($1.50 per 15-image batch)
- **GPU Strategy**: 
  - Development: A10G (~$0.30/hour) for UI testing
  - Production: L40S (~$1.10/hour) for optimal price/performance
- **Volume Efficiency**: Persistent model caching to avoid redownloads

### 5.3 Quality Standards
- **Output Resolution**: 4K (3840x2160) minimum
- **Image Format**: PNG for lossless quality
- **Upscaling**: SUPIR for professional-grade enhancement
- **Consistency**: Reproducible results across batch generations

---

## 6. Integration Requirements

### 6.1 React/NextJS Integration
- **API Client**: TypeScript SDK for seamless integration
- **File Upload**: Direct S3 upload for LoRA files
- **Progress Tracking**: Real-time job status updates
- **Error Handling**: Comprehensive error responses and retry logic

### 6.2 Storage Integration
- **S3 Buckets**: 
  - Input: User-uploaded LoRA models
  - Output: Generated 4K images
  - Cache: Model weights and temporary files
- **Access Patterns**: Direct Modal-to-S3 integration using mounted volumes

---

## 7. Implementation Phases

### Phase 1: Foundation ✅ **COMPLETED**
- [x] Set up Modal.com infrastructure with basic ComfyUI
- [x] Implement Flux.1-dev base model integration
- [x] Create development UI environment
- [x] Basic S3 volume mounting

### Phase 2: Core Workflow ✅ **COMPLETED**
- [x] Integrate LoRA loading from S3 paths
- [x] Implement SUPIR upscaling pipeline
- [x] Create batch processing workflow
- [x] Convert workflow to Python using Modal's conversion process

### Phase 3: API Development ✅ **COMPLETED**
- [x] Develop REST API endpoints
- [x] Implement job queuing and status tracking
- [x] Add comprehensive error handling
- [x] Performance optimization and caching

### Phase 4: Integration & Testing ⚠️ **PARTIALLY COMPLETED**
- [ ] React/NextJS API integration *(Not started)*
- [⚠️] End-to-end testing with real LoRAs *(Basic testing implemented, full validation pending)*
- [⚠️] Cost optimization and monitoring *(Framework in place, validation pending)*
- [⚠️] Production deployment and scaling tests *(Deployment tools ready, scaling tests pending)*

---

### ✅ **Additional Implemented Features Beyond PRD Scope:**
- [x] **Job Tracking System**: File-based persistence with progress tracking
- [x] **Workflow Management**: Dynamic workflow customization and templates
- [x] **Development Tools**: Automated deployment script and comprehensive test suite
- [x] **User Organization**: S3 path structure with user_id organization
- [x] **Error Handling**: Comprehensive error tracking and reporting
- [x] **Documentation**: Complete API documentation and usage guides

### 🎯 **Current Status Summary:**
- **Core Platform**: Production-ready for ComfyUI image generation
- **API Layer**: Fully functional with job tracking and status monitoring
- **Infrastructure**: Modal deployment with dual GPU environments (A10G/L40S)
- **Integration**: Ready for React/NextJS client development
- **Testing**: Basic validation complete, performance testing pending

---

## 8. Success Metrics

### 8.1 Technical Metrics
- **API Response Time**: <120s for 15-image 4K batch
- **Cost per Image**: <$0.10 average
- **Success Rate**: >99% successful generation rate
- **Container Utilization**: >80% GPU utilization during processing

### 8.2 Business Metrics  
- **Monthly Volume**: Scale from <1000 to 5000-10000 images
- **User Satisfaction**: High-quality 4K outputs meeting professional standards
- **Cost Efficiency**: Maintain target pricing while scaling

---

## 9. Risk Mitigation

### 9.1 Technical Risks
- **Model Loading Performance**: Mitigated by persistent volume caching
- **S3 Access Latency**: Addressed through Modal's volume mounting
- **GPU Availability**: Handled by Modal's auto-scaling and multiple GPU options

### 9.2 Cost Risks
- **GPU Cost Overruns**: Monitored through detailed logging and auto-scaling limits
- **Storage Costs**: Managed through lifecycle policies and efficient caching

---

## 10. References

- [Modal ComfyUI Example](https://modal.com/docs/examples/comfyapp)
- [ComfyUI Prototype to Production](https://modal.com/blog/comfyui-prototype-to-production)
- Flux.1-dev Model Documentation
- SUPIR Upscaling Integration Guide
- Existing Modal volume with all models: models_volume = modal.Volume.from_name("models-vol")
- Existing Modal secret: aws_secret = modal.Secret.from_name("aws-secret")
- Mount user-images/ prefix at /data for all relevant functions:
```python
user_images_mount = {"/data": modal.CloudBucketMount(
    bucket_name="primeshot-uploads-01",
    key_prefix="user-images/",
    secret=aws_secret,
    read_only=False
)}
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: Phase 1 Completion 