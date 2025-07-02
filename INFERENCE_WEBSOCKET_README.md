# ComfyUI Inference WebSocket Progress Tracking

Real-time progress tracking for ComfyUI inference jobs using WebSockets. This system provides live progress updates during image generation workflows with accurate percentage completion and time estimation.

## Overview

The system consists of several components:
- **WebSocket Progress Server** (`inference_websocket_progress.py`) - Handles real-time progress broadcasting
- **Progress Tracker** (`inference_progress_tracker.py`) - Tracks workflow execution progress  
- **WebSocket Manager** (`inference_websocket_manager.py`) - Manages server lifecycle
- **Supabase Client** (`inference_supabase_client.py`) - Database integration for persistence
- **ComfyUI Integration** - Modified `generate_images` method with progress tracking

## WebSocket URL

```
wss://creativebuild--inference-websocket-progress.modal.run
```

## Usage

### 1. Starting an Inference Job with Progress Tracking

Include a `job_id` in your inference request to enable progress tracking:

```python
request_data = {
    "user_id": "user_123",
    "workflow_name": "flux_lora",
    "job_id": "inference_job_456",  # Required for progress tracking
    "parameters": {
        "prompt": "A beautiful landscape",
        "batch_size": 2,
        "steps": 20
    }
}

# Call ComfyUI inference endpoint
result = comfyui.generate_images(request_data)
```

### 2. Connecting to Progress Updates (Client Side)

```javascript
const jobId = "inference_job_456";
const ws = new WebSocket(`wss://creativebuild--inference-websocket-progress.modal.run/ws/progress/${jobId}`);

ws.onmessage = (event) => {
    const progressData = JSON.parse(event.data);
    console.log(`Progress: ${progressData.progress}%`);
    console.log(`Phase: ${progressData.phase}`);
    console.log(`ETA: ${progressData.estimated_remaining}s`);
    
    // Update UI with progress
    updateProgressBar(progressData.progress);
    updateStatusText(progressData.message);
    updateTimeRemaining(progressData.estimated_remaining);
};

ws.onopen = () => {
    console.log("Connected to inference progress stream");
};

ws.onclose = () => {
    console.log("Disconnected from inference progress stream");
};
```

### 3. Progress Data Format

The WebSocket sends JSON messages with the following structure:

```json
{
    "job_id": "inference_job_456",
    "progress": 45,
    "message": "Generating images",
    "timestamp": 1640995200.123,
    "estimated_remaining": 25,
    "elapsed_time": 15,
    "phase": "Generating...",
    "total_estimated_duration": 40,
    "total_images": 2,
    "status": "processing"
}
```

#### Fields:
- `job_id`: Unique identifier for the inference job
- `progress`: Completion percentage (0-100)
- `message`: Human-readable status message
- `timestamp`: Unix timestamp of the update
- `estimated_remaining`: Estimated seconds remaining
- `elapsed_time`: Seconds elapsed since start
- `phase`: Current workflow phase ("Initializing...", "Generating...", "Finalizing...")
- `total_estimated_duration`: Initial time estimate in seconds
- `total_images`: Number of images being generated
- `status`: Job status ("processing", "completed", "failed")

## Progress Phases

The inference workflow is divided into phases with specific progress ranges:

### Phase 1: Initialization (0-10%)
- Health check
- Workflow validation
- LoRA setup
- Parameter injection

### Phase 2: Generation (10-90%)
- Workflow execution
- Image sampling
- Latent processing
- Upscaling (if applicable)

### Phase 3: Finalization (90-100%)
- Result processing
- S3 upload
- Cleanup

## Database Integration

Progress is also stored in the `inference_jobs` Supabase table for persistence:

```sql
CREATE TABLE inference_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    workflow_name TEXT NOT NULL,
    parameters JSONB,
    status TEXT DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    output_urls TEXT[],
    images_generated INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Server Lifecycle Management

The WebSocket server automatically:
- Starts when the first inference job is registered
- Handles multiple concurrent jobs
- Shuts down after 3 minutes of inactivity
- Maintains persistent connections during job execution

## API Endpoints

### Health Check
```
GET https://creativebuild--inference-websocket-progress.modal.run/health
```

### Server Status
```
GET https://creativebuild--inference-websocket-progress.modal.run/status
```

### Job Status
```
GET https://creativebuild--inference-websocket-progress.modal.run/job/{job_id}/status
```

### Debug Connections
```
GET https://creativebuild--inference-websocket-progress.modal.run/debug/connections
```

## Error Handling

### Connection Failures
- Automatic reconnection with exponential backoff
- Message queuing during disconnections
- Graceful degradation if WebSocket unavailable

### Job Failures
- Error messages broadcast to connected clients
- Database status updated to 'failed'
- Cleanup of WebSocket resources

## Environment Variables

Required environment variables for database integration:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## Example Client Implementation

```javascript
class InferenceProgressTracker {
    constructor(jobId) {
        this.jobId = jobId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        const wsUrl = `wss://creativebuild--inference-websocket-progress.modal.run/ws/progress/${this.jobId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('Connected to inference progress stream');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleProgress(data);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from inference progress stream');
            this.attemptReconnect();
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleProgress(data) {
        // Update UI with progress data
        const progressPercent = data.progress;
        const statusMessage = data.message;
        const timeRemaining = data.estimated_remaining;
        
        // Update progress bar
        document.getElementById('progress-bar').style.width = `${progressPercent}%`;
        document.getElementById('status-text').textContent = statusMessage;
        document.getElementById('time-remaining').textContent = `${timeRemaining}s remaining`;
        
        // Handle completion
        if (data.status === 'completed') {
            this.onCompleted(data);
        } else if (data.status === 'failed') {
            this.onFailed(data);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000;
            setTimeout(() => this.connect(), delay);
        }
    }
    
    onCompleted(data) {
        console.log('Inference completed successfully');
        this.disconnect();
    }
    
    onFailed(data) {
        console.error('Inference failed:', data.error_message);
        this.disconnect();
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Usage
const tracker = new InferenceProgressTracker('inference_job_456');
tracker.connect();
```

## Deployment

The WebSocket server is deployed as a Modal function:

```bash
modal deploy inference_websocket_progress.py
```

The ComfyUI app automatically includes the progress tracking components when deployed.

## Monitoring

Monitor the WebSocket server health and active connections:

```bash
curl https://creativebuild--inference-websocket-progress.modal.run/health
curl https://creativebuild--inference-websocket-progress.modal.run/status
```