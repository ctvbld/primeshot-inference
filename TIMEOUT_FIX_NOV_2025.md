# Inference Timeout Fix - November 26, 2025

## Problem Summary

Analysis of inference logs revealed a race condition where:
1. Generation completed successfully around 143s
2. Timeout logic triggered at 120s and declared failure at 142.7s
3. Success message (`EXECUTION_SUCCESS`) arrived milliseconds after failure was declared
4. System wasted resources retrying an already-successful generation

### Timeline of Failed Job
```
120s   → Absolute timeout triggers
130s   → Grace period ends (10s)
142.7s → Directory scanning completes, job marked as FAILED
143s   → ❌ EXECUTION_SUCCESS arrives (TOO LATE)
```

## Root Causes

1. **Timeout too aggressive**: 120s insufficient for complex generations
2. **Grace period too short**: 10s not enough for ComfyUI to finish processing
3. **No late success recovery**: Once timeout path started, couldn't recover from race condition
4. **Directory scanning adds delay**: 6+ seconds of scanning but still missed actual completion

## Solutions Implemented

### 1. Increased Timeout: 120s → 180s

**Changed:**
- Function parameter default: `timeout: int = 180`
- Function call: `get_image_outputs(..., timeout=180)`
- Updated docstring to reflect new default

**Rationale:**
- Analysis of Modal dashboard shows:
  - 95% of jobs complete in <60s
  - Outliers take 120-150s
  - 180s provides safe buffer without being excessive
  - Still fails "fast enough" for truly stuck jobs

**Files Modified:**
- `comfyui_app.py` line 470: Function signature
- `comfyui_app.py` line 478: Docstring
- `comfyui_app.py` line 1466: Function call

### 2. Increased Grace Period: 10s → 30s

**Changed:**
- Grace period constant: `grace_period = 30`

**Rationale:**
- Original 10s wasn't sufficient for:
  - ComfyUI to finish final processing steps
  - WebSocket messages to arrive
  - File system operations to complete
- 30s provides adequate buffer for late completions
- Total timeout now: 180s + 30s = 210s max

**Files Modified:**
- `comfyui_app.py` line 507: Grace period value

### 3. Added Late Success Recovery

**New Logic:**
```python
# After all methods fail, check one final time
print(f"🔍 Performing final check for late completion data...")
completion_data = get_prompt_completion_data(job_id, prompt_id)
if completion_data and completion_data.get("outputs"):
    print(f"🎉 Late success detected! Completion data arrived during directory scanning.")
    return completion_data["outputs"], "websocket_late"
```

**Rationale:**
- Catches race condition where success arrives during directory scanning
- Prevents unnecessary retries of successful generations
- Saves compute resources and reduces user wait time
- Returns special source tag `"websocket_late"` for monitoring

**Files Modified:**
- `comfyui_app.py` lines 559-564: Added final check before raising error

## Expected Impact

### Before Fix
```
Timeout at 120s → Grace 10s → Directory scan 12s → FAIL at 142s
→ Success arrives at 143s (MISSED)
→ Container retry #1
→ Wasted resources + confused user experience
```

### After Fix
```
Timeout at 180s → Grace 30s → Directory scan 12s
→ Final check catches late success at ~222s
→ Returns successful result
→ No retry needed ✅
```

### Alternative Timeline (Truly Stuck)
```
Timeout at 180s → Grace 30s → Directory scan 12s
→ Final check finds nothing
→ FAIL at ~222s
→ Container retry (legitimately needed)
```

## Monitoring & Validation

### Success Indicators
- Look for log messages: `🎉 Late success detected!`
- Check for `source: "websocket_late"` in successful completions
- Monitor reduction in unnecessary container retries
- Track generation completion time distribution

### Metrics to Track
1. **Timeout rate**: Should decrease significantly
2. **Late success rate**: Track how often late recovery saves jobs
3. **Average completion time**: May increase slightly but should stabilize
4. **Retry rate**: Should decrease for race-condition scenarios

### Log Messages to Monitor
```
✅ Found completion data during grace period!  # Grace period working
🎉 Late success detected!                      # Late recovery working
⏳ WebSocket timeout - adding 30s grace...     # New timeout value
```

## Environment Variables

The timeout can still be overridden via environment variable:
```bash
COMFY_GENERATION_TIMEOUT=180  # New default (up from 120)
```

If you need longer timeouts for specific scenarios (e.g., batch jobs):
```bash
COMFY_GENERATION_TIMEOUT=300  # 5 minutes for batch processing
```

## Rollback Plan

If these changes cause issues:

1. **Quick rollback**: Revert the three changes
   ```bash
   git revert <commit-hash>
   ```

2. **Partial rollback**: Keep timeout increase, remove late recovery
   - Change `grace_period = 30` back to `grace_period = 10`
   - Remove the final check before raising error
   - Keep `timeout=180`

## Testing Recommendations

1. **Monitor Dashboard**: Watch Modal dashboard for success rate changes
2. **Check Logs**: Look for the new log messages indicating late success
3. **Performance**: Verify overall completion times remain reasonable
4. **Error Rate**: Ensure truly stuck jobs still fail appropriately

## Related Documentation

- `TIMEOUT_FIX_SUMMARY.md` - Previous timeout fixes (120s implementation)
- `IDLE_TIMEOUT_FIX.md` - Idle timeout mechanism
- `VRAM_CLEANUP_CHANGES.md` - Related cleanup logic

## Summary

This fix addresses a critical race condition where successful generations were being marked as failures due to tight timeouts and lack of late success recovery. The changes are:

- ✅ **Conservative**: Increased timeouts based on real data
- ✅ **Safe**: Added recovery mechanism for race conditions
- ✅ **Reversible**: Easy to rollback if needed
- ✅ **Monitored**: Clear log messages for tracking effectiveness

Expected result: **Fewer false failures, better resource utilization, improved user experience.**


