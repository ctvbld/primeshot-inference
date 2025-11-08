# Final Attempt Error Response Fix

## Problem

When generation fails on the **3rd/final attempt**, the frontend keeps showing "generating" indefinitely instead of "failed".

## Root Cause Analysis (The Real Issue)

**You were right to challenge the assumption!** The actual problem is more subtle:

### WebSocket Lifecycle Across Container Retries

1. **Container 1** starts → **WebSocket relay starts** (daemon thread)
2. **Container 1 fails** → exception raised → **Modal kills container**
3. **Daemon thread dies** when container exits (line 1046: `daemon=True`)
4. **Container 2** spawns → **NEW WebSocket relay starts**
5. **Container 2 fails** → same pattern
6. **Container 3** (final) → **NEW WebSocket relay starts**
7. **Container 3 fails** → sends failure message → **raises exception → container killed**

### The Critical Race Condition

When `send_custom_message_to_job()` is called (line 1892):

```python
# Line 1105-1123 in ws_preview_relay.py
def send_custom_message_to_job(job_id: str, message_data: dict) -> bool:
    manager.send_message(message_data)  # Line 1118
    return True  # Returns IMMEDIATELY!

# Line 127-138
def send_message(self, message_data: dict):
    self._custom_messages.append(message_data)  # Just QUEUES it!
```

**The message is only queued**, not sent! The actual send happens asynchronously in the relay loop (line 323-333).

Then at line 1984 (OLD code):
```python
raise  # Container exits IMMEDIATELY, killing daemon thread!
```

**The daemon thread is killed before it can send the queued message!**

## Solution

**Two-part fix:**

### 1. Add Sleep After Queueing Message
Give the daemon thread time to actually send the message before container exits:

```python
send_custom_message_to_job(job_id, failure_message)
time.sleep(1.0)  # CRITICAL: Wait for async send to complete
```

### 2. Return Error Response Instead of Raising
On final attempt, return a proper response so Modal doesn't mark it as "Failed":

```python
if not is_final_attempt:
    raise  # Trigger Modal retry
else:
    return {  # Return error response, keep container alive
        "success": False,
        "status": "failed",
        ...
    }
```

## Changes Made

### File: `modal_apps/inference/comfyui_app.py`

**Lines 1877-1901** (WebSocket failure notification):

```python
# Send failure notification via WebSocket
try:
    send_custom_message_to_job(job_id, failure_message)
    print(f"📤 Sent final failure notification via WebSocket for job {job_id}")
    
    # CRITICAL: Give the daemon thread time to actually send the queued message
    # The send is async - just queues it. Need to wait for actual delivery.
    import time
    time.sleep(1.0)  # Allow WebSocket message to flush
    print(f"⏳ Waited for WebSocket message delivery")
except Exception as ws_fail_e:
    print(f"⚠️ Failed to send failure notification via WebSocket: {ws_fail_e}")
```

**Lines 1981-2000** (Conditional exception raising):

```python
# Re-raise the exception ONLY if not final attempt
if not is_final_attempt:
    print(f"🔄 Re-raising exception to allow Modal container retry...")
    raise
else:
    print(f"🚨 Final attempt failed - returning error response instead of raising")
    return {
        "job_id": job_id,
        "success": False,
        "status": "failed",
        ...
    }
```

## Why Both Changes Are Needed

### Without Sleep (WebSocket message queued but not sent)
```
Container 3:
  1. Queue failure message ✅
  2. Raise exception → container killed ❌
  3. Daemon thread killed before sending → message lost ❌
  
Frontend: Keeps showing "Generating..." ❌
```

### With Sleep But Still Raising
```
Container 3:
  1. Queue failure message ✅
  2. Sleep 1s → message sent ✅
  3. Raise exception → Modal marks as "Failed" ❌
  4. HTTP 500 response → frontend confused ❌
  
Frontend: Might show "Generating..." (depends on timing) ❌
```

### With Sleep AND Return Response
```
Container 3:
  1. Queue failure message ✅
  2. Sleep 1s → message sent ✅
  3. Return error response → HTTP 200 ✅
  4. Container exits gracefully ✅
  
Frontend: Shows "Failed" status ✅
```

## Expected Behavior

### Before Fix
```
Final attempt fails:
  - Queues WebSocket message
  - Raises exception → container killed
  - Message never sent (daemon thread killed)
  - Modal returns HTTP 500
  - Frontend: "Generating..." indefinitely ❌
```

### After Fix
```
Final attempt fails:
  - Queues WebSocket message ✅
  - Sleeps 1s → message delivered ✅
  - Database updated via EF ✅
  - Returns error response ✅
  - Modal returns HTTP 200 ✅
  - Frontend: "Generation failed" ✅
```

## Testing

### Test Case: Missing LoRA File
```
Expected logs on attempt 3:
  🚨 FINAL ATTEMPT FAILED - Marking job as failed
  📤 Sent final failure notification via WebSocket
  ⏳ Waited for WebSocket message delivery  ← NEW
  ✅ Marked job as failed via inference-complete EF
  🚨 Final attempt failed - returning error response  ← NEW
```

### Frontend Behavior
```
Before: Indefinite "Generating..."
After:  Shows "Failed: Generation failed after 3 attempts..."
```

## Why the Sleep Duration is 1 Second

The WebSocket relay loop (line 323-333) processes custom messages on every iteration:
- Loop iteration: ~100-500ms
- WebSocket send: ~50-200ms
- Network latency: ~10-100ms

**1 second is sufficient** for:
- At least 2 loop iterations
- Message send + network round-trip
- Safety buffer for slow networks

Could be reduced to 500ms if needed, but 1s is safe and only adds 1s to failure reporting (acceptable).

## Author

Fixed based on user feedback: "Is it because it closes the WS connection on the first failure, go check ffs!"

You were right - the daemon thread was being killed before sending queued messages!

Date: November 8, 2025

**Lines 1978-2000** (updated exception handling):

1. **Added conditional check** - Only re-raise if not final attempt
2. **Return error response on final attempt** - Proper HTTP response with error details
3. **Keep all existing notifications** - WebSocket and Edge Function calls still happen

## How It Works

### Non-Final Attempts (1-2)

```
Attempt 1 fails:
  1. Send retry notification via WebSocket ✅
  2. Keep relay alive ✅
  3. Re-raise exception → Modal spawns new container ✅
```

### Final Attempt (3)

```
Attempt 3 fails:
  1. Send failure notification via WebSocket ✅
  2. Update database via inference-complete EF ✅
  3. Clean up LoRAs and VRAM ✅
  4. Signal WebSocket completion ✅
  5. Return error response (don't raise!) ✅
     
Frontend receives:
  {
    "success": false,
    "status": "failed",
    "message": "Generation failed after 3 attempts...",
    ...
  }
```

## Expected Behavior

### Before Fix
```
Final attempt fails:
  - WebSocket message: "failed" (might not deliver)
  - Modal response: Exception raised → HTTP 500
  - Frontend: Keeps showing "Generating..." ❌
```

### After Fix
```
Final attempt fails:
  - WebSocket message: "failed" ✅
  - Database: Updated to "failed" ✅
  - Modal response: HTTP 200 with error payload ✅
  - Frontend: Shows "Generation failed" ✅
```

## Testing

### Test Case 1: Missing LoRA File (Your Current Issue)
```python
{
  "character_lora": "/data/path/deleted_lora.safetensors"  # File doesn't exist
}

Expected:
  - Attempt 1: Fails ~7s
  - Attempt 2: Fails ~7s  
  - Attempt 3: Fails ~7s, returns error response
  - Frontend: Shows "Failed" status
```

### Test Case 2: ComfyUI Validation Error
```python
{
  "workflow": "invalid_node_parameters.json"
}

Expected:
  - All 3 attempts fail quickly
  - Final attempt returns error response
  - Frontend shows validation error message
```

### Test Case 3: Timeout/Stuck Generation
```python
{
  "steps": 1000,  # Takes too long
}

Expected:
  - Attempts timeout after 90s idle
  - Final attempt returns timeout error
  - Frontend shows timeout message
```

## Monitoring

### Successful Final Failure Response

Look for this log sequence on attempt 3:

```
🚨 FINAL ATTEMPT FAILED - Marking job as failed
✅ Marked job {job_id} as failed via inference-complete EF
📤 Sent final failure notification via WebSocket for job {job_id}
🚨 Final attempt failed - returning error response instead of raising
📤 Sent final completion message for job {job_id}
```

### Modal Response

Modal should now return **HTTP 200** with error payload instead of **HTTP 500**:

```json
{
  "job_id": "abc-123",
  "success": false,
  "status": "failed",
  "error": "Failed to submit prompt: HTTP 400...",
  "message": "Generation failed after 3 attempts: Check logs for detailed error information",
  "total_attempts": 3
}
```

## Related Fixes

This fix complements:
1. **Idle Timeout Fix** (`IDLE_TIMEOUT_FIX.md`) - Prevents false timeout positives
2. **Character LoRA Bypass Fix** (`CHARACTER_LORA_BYPASS_FIX.md`) - Prevents validation errors when LoRA is missing

## Impact

- ✅ Frontend properly shows "Failed" status on final failure
- ✅ No more indefinite "Generating..." after all retries exhausted
- ✅ Proper error messages displayed to user
- ✅ Modal function returns successfully (HTTP 200) even on job failure
- ✅ Database and WebSocket notifications still work
- ✅ No breaking changes to retry logic

## Rollback

If this causes issues:

```bash
cd /Users/ledave/Documents/Primeshot/App/modal_apps/inference
git diff HEAD~1 comfyui_app.py
git checkout HEAD~1 -- comfyui_app.py
```

## Author

Fixed based on user report: "On the third and final attempt if failing, it should return failed to the frontend"

Date: November 8, 2025

