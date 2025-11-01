# VRAM Cleanup & Timeout Reduction Changes

## Problem Identified

The inference Modal container was crashing after completing the first job when starting a second job. Analysis revealed:

1. **Root Cause**: LoRA weights remained in VRAM after job completion, causing OOM when loading new LoRAs for the next job
2. **Symptom**: ComfyUI WebSocket closed after only 5 messages on second job, followed by complete health check failures
3. **Detection Time**: ~170 seconds due to excessive retry layers before detecting the crash

## Changes Made

### 1. Added VRAM Cleanup Function (Line 437-486)

```python
def cleanup_loras_from_vram(job_id: str, port: int = 8000) -> None:
```

This function:
- Interrupts ComfyUI to clear any lingering state
- Clears the ComfyUI queue
- Calls PyTorch CUDA cache cleanup (`torch.cuda.empty_cache()`)
- Synchronizes CUDA to ensure cleanup completes
- **Preserves base models** (UNET, VAE, CLIP, upscaler) for fast subsequent jobs

### 2. VRAM Cleanup Locations

Added cleanup calls in **3 strategic places**:

#### A. Before Starting New Job (Line 1128-1134)
- Only runs on warm starts (skips first job)
- Safety net to prevent OOM from any leftover state
- Non-blocking - continues even if cleanup fails

#### B. After Successful Job Completion (Line 1793-1797)
- Runs after symlink cleanup
- Ensures container is ready for next job
- Non-critical - logs error but doesn't fail the job

#### C. After Failed Job (Line 1951-1955)
- Runs even on failure to clean up for retry
- Prevents cascading OOM failures
- Non-critical - logs error but doesn't block retry

### 3. Reduced Timeouts for Fast Failure Detection

#### WebSocket Timeout
- **Before**: 120s → **After**: 60s
- Grace period: 15s → 5s

#### History API Retries
- **Before**: 6 attempts @ 10s timeout (3s backoff)
- **After**: 3 attempts @ 5s timeout (2s backoff)

#### Directory Scan
- **Before**: 5 attempts @ 2s delay
- **After**: 2 attempts @ 1s delay

#### Total Failure Detection Time
- **Before**: ~137-170s for two container attempts
- **After**: ~30-40s for two container attempts
- **Improvement**: ~75% reduction in failure detection time

## Expected Results

1. ✅ **No More OOM Crashes**: LoRAs purged between jobs
2. ✅ **Fast Failure Detection**: 30-40s instead of 170s
3. ✅ **Base Models Stay Warm**: Faster subsequent job starts
4. ✅ **Better Container Utilization**: Same container handles multiple jobs
5. ✅ **Improved User Experience**: Faster feedback on failures

## Testing Recommendations

1. **Sequential Jobs**: Run 2-3 jobs back-to-back with different LoRAs
2. **Monitor VRAM**: Check that VRAM usage drops after each job
3. **Failure Scenarios**: Intentionally fail a job and verify cleanup still happens
4. **Cold Start**: Verify first job still works (cleanup skipped)
5. **Performance**: Measure job-to-job latency (should be ~0.5s cleanup overhead)

## Monitoring

Look for these log messages:
- `🧹 Cleaning up LoRAs from VRAM for job {job_id}`
- `✅ Interrupted ComfyUI`
- `✅ Cleared queue`
- `✅ Cleared CUDA cache`
- `✅ LoRA VRAM cleanup completed for job {job_id}`

## Rollback Plan

If issues arise, you can disable VRAM cleanup by commenting out the three cleanup calls:
- Line 1128-1134 (before job)
- Line 1793-1797 (after success)
- Line 1951-1955 (after failure)

The timeout reductions can be reverted by changing the values back in `get_image_outputs()` function.

