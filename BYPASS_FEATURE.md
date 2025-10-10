# Node Bypass Feature via Settings Column

## Overview

This feature allows you to bypass ComfyUI nodes directly through the `settings` column in your database, eliminating the need for a separate `bypass_nodes` parameter or column.

## Quick Start

### Example 1: Simple Bypass

Bypass the VibSat node entirely:

```json
{
  "VibSat": {
    "__bypass__": true
  }
}
```

### Example 2: Bypass with Custom Configuration

Bypass a node with custom passthrough settings:

```json
{
  "LightLeaks": {
    "__bypass__": true,
    "__passthrough_key__": "image",
    "__output_index__": 0
  }
}
```

### Example 3: Mixed Configuration

Combine parameter overrides and bypasses:

```json
{
  "FilmGrain": {
    "grain_intensity": 0.15,
    "grain_size": 1.2
  },
  "VibSat": {
    "__bypass__": true
  },
  "ChannelMixer": {
    "red_adjust": 1.1
  }
}
```

## Control Keys Reference

### `__bypass__` (boolean, required for bypass)
- Set to `true` to bypass the node
- When true, the node is removed from the workflow and its inputs are passed through

### `__passthrough_key__` (string, optional)
- Default: `"image"`
- Specifies which input connection should be forwarded to downstream nodes
- Common values: `"image"`, `"model"`, `"clip"`, `"latent"`

### `__output_index__` (integer, optional)
- Default: `0`
- For nodes with multiple outputs, specifies which output slot to replace
- Example: LoRA loaders have output 0 (model) and output 1 (clip)

## Database Integration

### Where to Store

Store bypass configurations in:

1. **Style table** - `styles.settings` column (applies to all jobs using that style)
2. **Inference job table** - `inference_jobs.settings_override` column (job-specific overrides)

Both are merged, with job-specific settings taking precedence.

### SQL Examples

#### Add bypass to a style:

```sql
UPDATE styles 
SET settings = jsonb_set(
  COALESCE(settings, '{}'::jsonb),
  '{VibSat}',
  '{"__bypass__": true}'::jsonb
)
WHERE id = 'your-style-id';
```

#### Add bypass to a specific job:

```sql
UPDATE inference_jobs
SET settings_override = jsonb_set(
  COALESCE(settings_override, '{}'::jsonb),
  '{LightLeaks}',
  '{"__bypass__": true}'::jsonb
)
WHERE id = 'your-job-id';
```

#### Remove a bypass:

```sql
UPDATE styles
SET settings = settings - 'VibSat'
WHERE id = 'your-style-id';
```

## Implementation Details

### Processing Order

1. **Settings override processing** - Regular parameter overrides are applied first
2. **Bypass detection** - Nodes with `__bypass__: true` are collected
3. **Bypass application** - Collected bypasses are applied using the existing `bypass_node` function
4. **Auto-bypass logic** - Remaining auto-bypass logic runs (for nodes not in settings_override)

### Node Title Resolution

The system uses title aliases for common nodes:
- `"VibSat"` → matches nodes with title "VibSat"
- `"FilmGrain"` → matches nodes with title "FilmGrain"
- `"CharacterLora"` → matches nodes with title "CharacterLora"
- Case-insensitive matching

### Multi-Output Nodes

For nodes with multiple outputs (like LoRA loaders), you may need to bypass each output separately:

```json
{
  "StyleLora": {
    "__bypass__": true,
    "__passthrough_key__": "model",
    "__output_index__": 0
  }
}
```

**Note:** Currently, the implementation applies bypass once per node. For complex multi-output scenarios, you may need to use the manual `bypass_nodes` parameter.

## Code Changes

### Files Modified

1. **`modal_apps/inference/lib/workflow_patcher.py`**
   - Modified `_apply_settings_override()` method to detect and handle bypass flags
   - Added bypass spec collection and application
   - Updated docstring with bypass usage examples

2. **`modal_apps/inference/README.md`**
   - Added "Node Bypass and Settings Override" section
   - Included usage examples and control key documentation

3. **`modal_apps/inference/BYPASS_FEATURE.md`** (this file)
   - Comprehensive feature documentation
   - SQL examples for database integration

### Key Functions

- `WorkflowPatcher._apply_settings_override()` - Detects `__bypass__` flags and collects bypass specs
- `WorkflowPatcher._apply_bypass_nodes()` - Applies bypass specifications
- `bypass_node()` - Core bypass logic (unchanged, reused)

## Testing

### Manual Testing

1. Create a test style with bypass configuration:
```sql
INSERT INTO styles (id, name, settings) VALUES (
  'test-bypass-style',
  'Test Bypass Style',
  '{"VibSat": {"__bypass__": true}}'::jsonb
);
```

2. Create an inference job using that style

3. Check logs for bypass confirmation:
```
🔄 Will bypass VibSat (passthrough: image, output: 0)
🔄 Applying 1 bypass specifications from settings
✅ Auto-bypassed VibSat
```

### Verification

Monitor the workflow patching logs to confirm:
- Bypass flags are detected
- Bypass specs are collected
- Bypass operations complete successfully
- Downstream nodes are properly rewired

## Troubleshooting

### Bypass Not Applied

**Problem:** Node is still processing despite `__bypass__: true`

**Solutions:**
1. Check node title matches exactly (case-insensitive)
2. Verify JSON syntax in database
3. Check logs for "Will bypass" messages
4. Ensure `__bypass__` is boolean `true`, not string `"true"`

### Wrong Input Forwarded

**Problem:** Downstream nodes receive wrong input

**Solutions:**
1. Check `__passthrough_key__` matches the input name in your workflow
2. Common keys: `"image"`, `"model"`, `"clip"`, `"latent"`
3. Inspect workflow JSON to confirm input key names

### Node Not Found

**Problem:** Warning "No node with title 'X' found"

**Solutions:**
1. Open workflow JSON and check `_meta.title` or `_meta._ui_name`
2. Use exact title from workflow
3. Check title aliases in `get_title_aliases()`

## Future Enhancements

Potential improvements:
- Support for bypassing multiple outputs in a single spec
- Wildcard node matching (e.g., `"*Lora"`)
- Conditional bypasses based on other settings
- Visual bypass configuration in admin UI

