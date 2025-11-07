# Idle Timeout Fix for ComfyUI Generation

## Problem

The previous timeout implementation used **total elapsed time** instead of **idle time** (time since last progress update). This caused generations to fail even when they were actively progressing.

### Example Failure Scenario

```
Time 0s:   Generation starts
Time 30s:  Progress update (30% complete)
Time 60s:  Progress update (60% complete)  
Time 90s:  Progress update (90% complete)
Time 91s:  TIMEOUT! ❌ (exceeded 60s total timeout)
```

Even though the job was continuously showing progress, it hit the total timeout and failed.

## Root Cause

In `ws_preview_relay.py`, line 1188:

```python
# ❌ BEFORE: Timeout based on total elapsed time
start_time = time.time()
while time.time() - start_time < timeout:
    # ... check for completion
    # Progress updates don't reset this timer!
```

The `manager.last_activity_time` was being tracked and updated with each progress message, but it **wasn't being used** for the timeout check. It was only used for the stuck node detection.

## Solution

Implemented **dual timeout logic**:

1. **MAX_TOTAL_TIME**: Absolute maximum (default: 120s for short jobs, 600s for batch jobs)
   - Prevents infinite loops if something goes wrong
   - Safety limit to catch runaway processes

2. **MAX_IDLE_TIME**: Time without progress (default: 90s, configurable via `COMFY_IDLE_TIMEOUT`)
   - Detects truly stuck/crashed generation
   - Resets on every progress update
   - Allows long-running jobs as long as they show progress

### New Logic

```python
# ✅ AFTER: Dual timeout with idle detection
start_time = time.time()
MAX_TOTAL_TIME = timeout  # Absolute safety limit (e.g., 600s)
MAX_IDLE_TIME = 90  # Time without progress = stuck

while True:
    elapsed_total = time.time() - start_time
    elapsed_idle = time.time() - manager.last_activity_time
    
    # Check absolute total timeout
    if elapsed_total > MAX_TOTAL_TIME:
        return False  # Exceeded absolute limit
    
    # Check idle timeout (resets with each progress update!)
    if elapsed_idle > MAX_IDLE_TIME:
        return False  # No progress, likely stuck
    
    # ... check for completion
```

## Expected Behavior

### Before Fix
```
Generation taking 150s with progress every 10s: ❌ TIMEOUT at 60s
Generation stuck at 50%: ❌ TIMEOUT at 60s
```

### After Fix
```
Generation taking 150s with progress every 10s: ✅ SUCCESS (idle time never exceeds 90s)
Generation stuck at 50%: ❌ TIMEOUT at 90s (no progress for 90s)
Generation taking 15 minutes total: ❌ TIMEOUT at 600s (absolute limit)
```

## Configuration

Set these environment variables in your Modal app:

### COMFY_IDLE_TIMEOUT (default: 90)
Maximum seconds without progress before considering job stuck.
```python
# In Modal app config
COMFY_IDLE_TIMEOUT=90  # Fail if no progress for 90s
```

**Recommended values:**
- 60s: Fast failure for stuck jobs
- 90s: Balanced (default)
- 120s: Very lenient, for complex workflows

### COMFY_GENERATION_TIMEOUT (default: 120)
Absolute maximum time for a single image generation.
```python
COMFY_GENERATION_TIMEOUT=120  # Safety limit
```

**Recommended values:**
- 120s: For single image generation
- 300s: For upscaling workflows
- 600s: For batch generation (15 images)

### COMFY_STUCK_NODE_TIMEOUT (default: 180)
Additional detection for when a specific node is stuck.
```python
COMFY_STUCK_NODE_TIMEOUT=180  # Fail if same node executes for 180s
```

## Changes Made

### 1. `lib/ws_preview_relay.py`

**Function**: `wait_for_prompt_completion()`

- ✅ Changed from single timeout to dual timeout (total + idle)
- ✅ Added `COMFY_IDLE_TIMEOUT` environment variable
- ✅ Added progress logging every 30s
- ✅ Improved stuck node detection to work with idle timeout
- ✅ Simplified inactivity fallback logic

### 2. `comfyui_app.py`

**Function**: `get_image_outputs()`

- ✅ Increased default timeout from 60s to 120s
- ✅ Updated documentation to explain dual timeout
- ✅ Increased grace period from 5s to 10s
- ✅ Increased history API attempts from 3 to 5
- ✅ Increased history API timeout from 5s to 10s per attempt
- ✅ Increased directory scan attempts from 2 to 4
- ✅ Increased directory scan delay from 1s to 2s

## Testing

### Test Case 1: Long but Active Generation
```python
# 150 second generation with progress every 10s
# BEFORE: ❌ Timeout at 60s
# AFTER:  ✅ Success - idle time never exceeds 90s
```

### Test Case 2: Stuck Generation
```python
# Generation gets stuck at 50% (node crashes)
# No progress updates for 91s
# BEFORE: ❌ Timeout at 60s (correct, but for wrong reason)
# AFTER:  ❌ Idle timeout at 90s (correct reason)
```

### Test Case 3: Very Long Generation
```python
# 15 minute generation (batch of 50 images)
# BEFORE: ❌ Timeout at 60s
# AFTER:  ❌ Absolute timeout at 600s (10 min limit)
#         Note: Set COMFY_GENERATION_TIMEOUT=900 for longer batches
```

## Monitoring

Look for these log messages:

### Successful Generation
```
⏳ Waiting for prompt {id} completion via WebSocket events
   • Max total time: 120s (absolute limit)
   • Max idle time: 90s (timeout without progress)
⏱️ Still waiting... 60s elapsed, 8s since last activity
✅ WebSocket reported completion (took 95.3s)
```

### Idle Timeout (Stuck Job)
```
⏳ Waiting for prompt {id} completion via WebSocket events
   • Max total time: 600s (absolute limit)
   • Max idle time: 90s (timeout without progress)
⏱️ Still waiting... 120s elapsed, 91s since last activity
🚨 Idle timeout (90s) exceeded - no progress for 91.2s
   Job {job_id} likely stuck or crashed
```

### Absolute Timeout (Job Too Long)
```
⏳ Waiting for prompt {id} completion via WebSocket events
   • Max total time: 120s (absolute limit)
   • Max idle time: 90s (timeout without progress)
⏱️ Still waiting... 120s elapsed, 12s since last activity
⏰ Absolute timeout (120s) exceeded for prompt {id}
```

## Rollback Instructions

If this causes issues, revert with:

```bash
cd /Users/ledave/Documents/Primeshot/App/modal_apps/inference
git diff HEAD~1 lib/ws_preview_relay.py comfyui_app.py
git checkout HEAD~1 -- lib/ws_preview_relay.py comfyui_app.py
```

Or set environment variables to previous behavior:
```python
COMFY_IDLE_TIMEOUT=60  # Match old total timeout
COMFY_GENERATION_TIMEOUT=60  # Old default
```

## Related Files

- `lib/ws_preview_relay.py` - Main timeout logic
- `comfyui_app.py` - Calls timeout function and handles fallbacks
- `TIMEOUT_FIX_SUMMARY.md` - Previous timeout optimization (now superseded)

## Author

Fixed based on user observation: "Why is the timeout not being reset every time there is a progress update?"

Date: November 7, 2025


