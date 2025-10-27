# Inference Timeout False Positive Fix

## Problem Analysis

The system was experiencing ~5% failure rate where jobs would "timeout" and retry, but the original container would complete successfully shortly after. This created a pattern where:

1. Generation starts and runs normally
2. System declares timeout at ~120-130s
3. Modal triggers container retry
4. Original generation completes 5-20s later (wasted)
5. Retry container starts fresh (sometimes fails to boot)

### Root Causes

1. **Overly aggressive stuck node detection**: 90s timeout per node, but some operations (high-res upscaling, multi-LoRA loading) legitimately take longer
2. **No grace period after timeout**: If WebSocket didn't signal completion within 120s, we immediately tried to retrieve outputs (which didn't exist yet)
3. **Race condition**: Generation completes → outputs written → but we'd already declared failure
4. **False positive stuck detection**: Node taking long ≠ stuck, if we're still receiving progress updates

## Changes Made

### 1. Smarter Stuck Node Detection (`ws_preview_relay.py`)

**Before:**
```python
stuck_timeout = 90  # Hard 90s limit
if time_on_same_node > stuck_timeout:
    return False  # Fail immediately
```

**After:**
```python
stuck_timeout = 180  # Increased to 180s
# Only fail if BOTH conditions true:
# 1. Same node > 180s
# 2. No progress messages in last 30s
if time_on_same_node > stuck_timeout and time_since_activity > 30:
    return False
elif time_on_same_node > stuck_timeout:
    # Node taking long but still receiving progress - OK!
    print(f"📊 Node executing for {time_on_same_node:.1f}s, but receiving progress - continuing...")
```

**Why this fixes it:**
- If we're receiving progress updates, the node is NOT stuck
- Some operations legitimately take 120-150s
- Only fail if truly stuck (no progress AND taking too long)

### 2. Grace Period After Timeout (`comfyui_app.py`)

**Before:**
```python
completed = wait_for_prompt_completion(job_id, prompt_id, timeout=120)
if completed:
    return completion_data["outputs"], "websocket"
# Immediately fail if timeout
```

**After:**
```python
completed = wait_for_prompt_completion(job_id, prompt_id, timeout=120)
if completed:
    return completion_data["outputs"], "websocket"
else:
    # Add 15s grace period for late completion
    print(f"⏳ WebSocket timeout - adding 15s grace period...")
    time.sleep(15)
    
    # Check if completion happened during grace period
    completion_data = get_prompt_completion_data(job_id, prompt_id)
    if completion_data and completion_data.get("outputs"):
        print(f"✅ Found completion data during grace period!")
        return completion_data["outputs"], "websocket_grace"
```

**Why this fixes it:**
- Catches completions that happen 1-20s after timeout
- Avoids premature failure when generation is actually finishing
- Minimal overhead (15s) vs cost of full container retry

### 3. More Aggressive Fallback Retries (`comfyui_app.py`)

**Before:**
```python
max_history_attempts = 3  # Only 3 tries
backoff = min(0.25 * (2 ** attempt), 1.0)  # Max 1s wait

found_images = find_generated_images(job_id)  # 3 attempts, 1s delays
```

**After:**
```python
max_history_attempts = 6  # 6 tries for history API
backoff = min(0.5 * (1.5 ** attempt), 3.0)  # Up to 3s between tries

found_images = find_generated_images(job_id, max_attempts=5, delay_seconds=2.0)
```

**Why this fixes it:**
- History API is very reliable once data is written
- Outputs may take 2-5s to fully write to disk after completion
- Better to wait a bit longer than trigger expensive container retry

## Expected Improvements

### Before Fix
- ~5% of generations timeout despite actually completing
- Wasted compute on retry containers
- User sees "4 errors in a row" (actually 1 job, 3 retry attempts)
- Failed generations often succeeded on retry (indicating false positive)

### After Fix
- Stuck node detection requires BOTH long runtime AND no progress
- 15s grace period catches late completions (most timeout cases)
- More aggressive fallback retries give outputs time to be written
- Should reduce false positive rate from ~5% to <1%

## Configuration

All timeouts are configurable via environment variables:

```bash
# Stuck node timeout (default: 180s, was 90s)
COMFY_STUCK_NODE_TIMEOUT=180

# WebSocket completion timeout (default: 120s)
COMFY_GENERATION_TIMEOUT=120

# No-progress timeout for inactivity detection (default: 60s)
COMFY_PROGRESS_TIMEOUT=60
```

## Monitoring

Key log messages to watch for:

### Good Signs (Fix Working)
```
📊 Node 22 executing for 95.3s, but receiving progress (last: 2.1s ago) - continuing...
✅ Found completion data during grace period!
✅ Got outputs from history (took 3.2s, attempt 4)
```

### Actual Stuck Nodes (Legitimate Failures)
```
🚨 Node 22 has been executing for 185.0s - likely stuck (timeout: 180s)
🚨 No progress for 35.2s
🚨 Failing job due to stuck node
```

## Testing Recommendations

1. Test with complex workflows (2+ LoRAs, high resolution)
2. Monitor logs for "grace period" messages
3. Check if retry rate decreases
4. Verify actual stuck nodes still fail appropriately
5. Monitor total generation time (should be similar or slightly longer due to grace period)

## Rollback Plan

If issues arise, can quickly revert via environment variables:

```bash
# Revert to old behavior
COMFY_STUCK_NODE_TIMEOUT=90  # Old aggressive timeout
COMFY_GENERATION_TIMEOUT=120  # Keep same
```

Or revert the code changes by removing the grace period and stuck node progress check.

