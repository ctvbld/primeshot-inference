# Error Message Improvements & Credit Refund Communication

## Overview
Enhanced error messaging system to provide users with clear, actionable feedback when generation fails, with explicit confirmation that credits have been refunded.

## Changes Made

### 1. Backend: Modal Inference App (`modal_apps/inference/comfyui_app.py`)

#### A. Added `credits_refunded` flag to error response
```python
return {
    "job_id": job_id,
    "success": False,
    "status": "failed",
    "error": str(e),
    "error_type": error_details.get("error_type", "Unknown"),
    "message": f"Generation failed after {max_retries} attempts: {error_details.get('suggestion', str(e))}",
    "credits_refunded": True,  # ← NEW: Explicit credit refund confirmation
    "total_attempts": max_retries,
    "comfy_error_details": error_details.get("comfy_error_details")
}
```

#### B. Improved error categorization with specific user messages

**Before:**
```python
# Generic messages like "Check logs for detailed error information"
```

**After:**
```python
# HTTP 400 - Validation errors
if "http 400" in error_str or "prompt_outputs_failed_validation" in error_str:
    error_details["category"] = "validation_error"
    if "value_not_in_list" in error_str:
        if "lora" in error_str or "node" in error_str and ("35" in error_str or "45" in error_str):
            error_details["suggestion"] = "The requested style or character is no longer available. Please try again or select a different style."
        else:
            error_details["suggestion"] = "Invalid parameter value. Please try again with different settings."
```

**Error Categories:**
- `validation_error` - Invalid settings or missing resources
- `connection_error` - Service temporarily unavailable
- `timeout_error` - Request took too long (with idle vs general timeout distinction)
- `endpoint_error` - Service configuration error
- `resource_error` - Insufficient resources (memory/GPU)
- `missing_resource` - Required file not found
- `generation_error` - General generation failure

### 2. Frontend: Translation Files (All 10 Languages)

Added new error message keys to `common/locales/*/inference.json`:

```json
"error": {
  // Existing messages...
  "missingStyle": "The requested style or character is no longer available. Credits refunded.",
  "validation": "Invalid settings provided. Credits refunded.",
  "resourceError": "Insufficient resources. Credits refunded.",
  "creditsRefunded": "Credits refunded"
}
```

#### Languages Updated:
- ✅ **US English** (`us`) - "Credits refunded"
- ✅ **UK English** (`gb`) - "Credits refunded"
- ✅ **Spanish** (`es`) - "Créditos reembolsados"
- ✅ **French** (`fr`) - "Crédits remboursés"
- ✅ **Italian** (`it`) - "Crediti rimborsati"
- ✅ **Portuguese** (`pt`) - "Créditos reembolsados"
- ✅ **German** (`de`) - "Credits erstattet"
- ✅ **Dutch** (`nl`) - "Credits terugbetaald"
- ✅ **Chinese** (`cn`) - "积分已退还"
- ✅ **Japanese** (`jp`) - "クレジットを返金いたしました"

## User-Facing Message Examples

### Missing LoRA (Your Specific Error)
**Before:** "Check logs for detailed error information"
**After:** "The requested style or character is no longer available. Credits refunded."

### Timeout
**Before:** "Request timed out - server may be overloaded"
**After:** 
- Idle: "Generation took too long without progress. This may be due to high server load. Please try again."
- General: "Request timed out. Please try again or use lower quality settings."

### Out of Memory
**Before:** N/A
**After:** "Insufficient resources. Please try again with lower resolution or fewer images."

### Connection Error
**Before:** "ComfyUI server may not be running or ready"
**After:** "Service temporarily unavailable. Please try again in a moment."

## Translation Guidelines Followed

Following `.cursor/rules/translations.mdc`:

1. **Brand Voice**: Professional yet friendly, simple yet premium
2. **Clear & Concise**: Short, direct language
3. **Reassuring**: Builds trust by explicitly confirming credit refund
4. **Natural Language**: Adapted to each language's cultural norms
5. **Accuracy**: Preserved meaning while respecting language-specific expressions

### Language-Specific Considerations

- **French**: Used "crédits remboursés" (not "retours")
- **Spanish**: Maintained consistent "tú" informality
- **Italian**: Avoided anglicisms
- **Portuguese**: European Portuguese tone
- **Chinese**: Natural phrasing ("积分已退还")
- **Japanese**: Polite business tone ("クレジットを返金いたしました")

## Technical Implementation

### Backend Flow
```
1. Error occurs in generation
2. Error categorized by type
3. User-friendly suggestion generated
4. credits_refunded flag set to true
5. Response returned with error details
```

### Frontend Usage
```typescript
// Use translation key with error category
t('inference:group.error.missingStyle')
// Output: "The requested style or character is no longer available. Credits refunded."

// Generic refund confirmation
t('inference:group.error.creditsRefunded')
// Output: "Credits refunded" (in selected language)
```

## Testing

To test the new error messages:

1. **Missing LoRA**: Trigger generation without character/style LoRA
2. **Validation Error**: Send invalid parameter values
3. **Timeout**: Trigger idle timeout scenario
4. **Resource Error**: Test with excessive resource requirements

Expected: User sees clear message in their selected language with explicit credit refund confirmation.

## Related Files
- Backend: `modal_apps/inference/comfyui_app.py` (lines 1832-1890, 2040-2050)
- Translations: `common/locales/*/inference.json` (all 10 languages)
- Previous fixes:
  - `IDLE_TIMEOUT_FIX.md`
  - `CHARACTER_LORA_BYPASS_FIX.md`
  - `FINAL_ATTEMPT_ERROR_RESPONSE_FIX.md`

