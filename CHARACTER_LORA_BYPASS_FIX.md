# Character LoRA Bypass Fix

## Problem

ComfyUI was rejecting workflows with HTTP 400 validation error when `character_lora` was `None`:

```
❌ HTTP 400 from ComfyUI /prompt: {
  "error": {
    "type": "prompt_outputs_failed_validation",
    "message": "Prompt outputs failed validation"
  },
  "node_errors": {
    "35": {
      "errors": [{"type": "value_not_in_list", ...}]
    }
  }
}
```

## Root Cause

**Node 35** = `CharacterLora` LoRA loader node

When `character_lora=None` (style-only generation), the code:
1. ✅ Didn't add it to the `desired_loras` list
2. ❌ Left the node with its **hardcoded default LoRA filename** from the workflow
3. ❌ That hardcoded filename doesn't exist → **validation fails**

### The Bug in `workflow_patcher.py`

```python
# Lines 219-252 (BEFORE FIX)
def _patch_loras(self, character_lora, style_lora, lora_filename):
    desired_loras = []
    if character_lora:
        desired_loras.append(character_lora)  # ← character_lora=None, not added
    if style_lora:
        desired_loras.append(style_lora)
    
    # ... patch LoRA nodes with desired_loras ...
    
    # ❌ BUG: Still tries to set by title even when character_lora=None
    if character_lora:
        self.manager.set_node_input_by_title("CharacterLora", "lora_name", character_lora)
    # ← When character_lora=None, this doesn't run, leaving hardcoded value!
```

The hardcoded value in `V1.3.json` node 35:
```json
{
  "35": {
    "inputs": {
      "lora_name": "9f3cd0e0-1c45-4c07-baef-cca36171d22a_training_ff8ccfc9...safetensors",
      // ↑ This file doesn't exist! ComfyUI validation fails!
    },
    "class_type": "LoraLoader",
    "_meta": {"title": "CharacterLora"}
  }
}
```

## Solution

When `character_lora=None`, **bypass** the CharacterLora node instead of leaving it with an invalid filename.

### Changes Made

#### 1. Updated `_patch_loras()` 

Added logging to indicate when CharacterLora needs bypassing:

```python
# Line 248-264 (AFTER FIX)
if character_lora:
    for alias in ["CharacterLora", "CharacterLoRa"]:
        self.manager.set_node_input_by_title(alias, "lora_name", character_lora)
else:
    # ✅ NEW: Log that we need to bypass
    logger.info("⚠️ No character LoRA - will bypass CharacterLora node")

if style_lora:
    for alias in ["StyleLora", "StyleLoRa"]:
        self.manager.set_node_input_by_title(alias, "lora_name", style_lora)
else:
    # ✅ NEW: Log that we need to bypass
    logger.info("⚠️ No style LoRA - will bypass StyleLora node")
```

#### 2. Updated `_auto_bypass_nodes()`

Added `character_lora` parameter and bypass logic:

```python
# Line 346-382 (AFTER FIX)
def _auto_bypass_nodes(self, character_lora, style_lora, settings_override):
    # ✅ NEW: Auto-bypass CharacterLora if no character_lora provided
    if not character_lora:
        logger.info("🔄 No character_lora provided, auto-bypassing CharacterLora node...")
        
        # Bypass both outputs of LoRA loader (model and clip)
        for output_idx, input_key in [(0, "model"), (1, "clip")]:
            try:
                self.workflow = bypass_node(
                    self.workflow,
                    target_ui_name="CharacterLora",
                    passthrough_input_key=input_key,
                    output_index=output_idx,
                    remove=(output_idx == 1)  # Remove on last operation
                )
                logger.info(f"✅ Bypassed CharacterLora output {output_idx}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to bypass CharacterLora: {e}")
    
    # (Existing StyleLora bypass logic remains the same)
    if not style_lora:
        # ... same bypass logic for StyleLora
```

#### 3. Updated function signature

```python
# Line 108 (AFTER FIX)
self._auto_bypass_nodes(character_lora, style_lora, settings_override)
```

## How Bypass Works

The `bypass_node()` function rewires the workflow to skip the LoRA loader:

**Before Bypass:**
```
Model → [CharacterLora node with invalid filename] → KSampler
CLIP  → [CharacterLora node with invalid filename] → CLIPTextEncode
```

**After Bypass:**
```
Model → KSampler (directly)
CLIP  → CLIPTextEncode (directly)
```

The CharacterLora node is removed from the workflow entirely.

## Expected Behavior

### Before Fix
```
Generation with character_lora=None:
❌ HTTP 400: value_not_in_list for node 35
❌ Container retries 3 times
❌ Job fails
```

### After Fix
```
Generation with character_lora=None:
🔄 No character_lora provided, auto-bypassing CharacterLora node...
✅ Bypassed CharacterLora output 0
✅ Bypassed CharacterLora output 1
✅ Workflow validates successfully
✅ Generation proceeds normally
```

## Test Cases

### Test Case 1: Style-Only Generation (No Character LoRA)
```python
{
  "character_lora": None,
  "style_lora": "job_id__lora_xyz.safetensors"
}
```
- ✅ CharacterLora node bypassed
- ✅ StyleLora node used
- ✅ Generation succeeds

### Test Case 2: Character-Only Generation (No Style LoRA)
```python
{
  "character_lora": "job_id__lora_abc.safetensors",
  "style_lora": None
}
```
- ✅ CharacterLora node used
- ✅ StyleLora node bypassed  
- ✅ Generation succeeds

### Test Case 3: Both LoRAs Provided
```python
{
  "character_lora": "job_id__lora_abc.safetensors",
  "style_lora": "job_id__lora_xyz.safetensors"
}
```
- ✅ CharacterLora node used
- ✅ StyleLora node used
- ✅ No bypassing occurs
- ✅ Generation succeeds

### Test Case 4: No LoRAs (Edge Case)
```python
{
  "character_lora": None,
  "style_lora": None
}
```
- ✅ Both nodes bypassed
- ✅ Base model only
- ✅ Generation succeeds

## Monitoring

Look for these log messages:

### Successful Bypass
```
⚠️ No character LoRA - will bypass CharacterLora node
🔄 No character_lora provided, auto-bypassing CharacterLora node...
✅ Bypassed CharacterLora output 0
✅ Bypassed CharacterLora output 1
```

### Bypass Failure (Shouldn't Happen)
```
⚠️ Failed to bypass CharacterLora: [error details]
```

If you see bypass failures, the CharacterLora node might not exist in the workflow or has a different title.

## Related Files

- `lib/workflow_patcher.py` - Main fix location
- `workflows/V1.3.json` - Workflow with node 35 (CharacterLora)
- `workflows/V1.2.json` - Older workflow (also has this issue)
- `lib/workflow_patcher_utils.py` - Contains `bypass_node()` function

## Impact

- **Style-only generations now work** ✅
- **Character-only generations still work** ✅
- **Both LoRAs work** ✅
- **No breaking changes** ✅

## Rollback

If this causes issues, revert the commit:

```bash
cd /Users/ledave/Documents/Primeshot/App/modal_apps/inference
git diff HEAD~1 lib/workflow_patcher.py
git checkout HEAD~1 -- lib/workflow_patcher.py
```

Or temporarily disable auto-bypass by setting both LoRAs explicitly (even if they don't exist).

## Author

Fixed based on production error: Node 35 (CharacterLora) validation failure when character_lora=None

Date: November 8, 2025

