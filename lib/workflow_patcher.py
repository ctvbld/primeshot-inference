from __future__ import annotations

from typing import Any, Dict, Tuple


def compute_dimensions(quality: str, aspect_ratio: str) -> Tuple[int, int]:
    base = {"1K": 1024, "2K": 2048, "4K": 4096}.get(quality, 1024)
    if aspect_ratio == "1:1":
        return base, base
    if aspect_ratio == "3:2":
        return base, int(round(base * 2 / 3))
    if aspect_ratio == "2:3":
        return int(round(base * 2 / 3)), base
    return base, base


def patch_workflow(
    workflow: Dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int | None,
    images_count: int,
    lora_filename: str | None = None,
    character_lora: str | None = None,
    style_lora: str | None = None,
    bypass_nodes: list[dict] | None = None,
) -> Dict[str, Any]:
    """Apply minimal patches to a ComfyUI workflow JSON.
    This assumes nodes are identified by common labels; you may adjust mapping.
    """
    wf = dict(workflow)

    def set_node_input(node_label: str, key: str, value: Any) -> None:
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("_meta", {}).get("_ui_name") == node_label:
                node.setdefault("inputs", {})[key] = value

    def set_node_input_by_title(node_title: str, key: str, value: Any) -> None:
        updated = False
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            meta = node.get("_meta", {})
            if meta.get("title") == node_title:
                node.setdefault("inputs", {})[key] = value
                updated = True
                print(f"✅ Updated {node_title} (node {node_id}) {key} = {value}")
        if not updated:
            print(f"⚠️ Node with title '{node_title}' not found for {key} = {value}")

    # Text encoders
    set_node_input("CLIPTextEncode", "text", prompt)
    set_node_input("CLIPTextEncodeNeg", "text", negative_prompt)

    # Seed, steps, cfg are workflow defaults; set seed if provided
    if seed is not None:
        set_node_input("KSampler", "seed", int(seed))

    # Resolution (EmptyLatentImage path)
    set_node_input("EmptyLatentImage", "width", int(width))
    set_node_input("EmptyLatentImage", "height", int(height))

    # LoRA injection
    # Priority: explicit character/style targets by title; fallback to generic LoraLoader by _ui_name
    if character_lora:
        # Current workflow uses title "CharacterLoRA" for character loader
        set_node_input_by_title("CharacterLoRA", "lora_name", character_lora)
    if style_lora:
        # Current workflow uses title "StyleLoRA" for optional style loader
        set_node_input_by_title("StyleLoRA", "lora_name", style_lora)
    if lora_filename and not character_lora and not style_lora:
        # Back-compat single lora case
        set_node_input("LoraLoader", "lora_name", lora_filename)

    # NB takes: prefer batch_size if present; otherwise caller will loop
    set_node_input("KSampler", "batch_size", int(images_count))

    # Auto-bypass StyleLoRA if no style_lora is provided
    if not style_lora:
        print("🔄 No style_lora provided, auto-bypassing StyleLoRA node...")
        
        # Bypass the StyleLoRA node by forwarding its model and clip inputs
        # This requires two operations since LoRA loaders have two outputs
        auto_bypass_specs = [
            {
                "ui_name": "StyleLoRA",
                "passthrough_input_key": "model",
                "output_index": 0,  # model output
                "remove": False
            },
            {
                "ui_name": "StyleLoRA", 
                "passthrough_input_key": "clip",
                "output_index": 1,  # clip output
                "remove": True  # Remove node after both outputs are rewired
            }
        ]
        
        bypass_success = False
        for spec in auto_bypass_specs:
            try:
                wf = bypass_node(
                    wf,
                    target_ui_name=spec["ui_name"],
                    passthrough_input_key=spec["passthrough_input_key"],
                    output_index=spec["output_index"],
                    remove=spec["remove"],
                )
                print(f"✅ Bypassed StyleLoRA output {spec['output_index']} -> {spec['passthrough_input_key']}")
                bypass_success = True
            except Exception as e:
                print(f"⚠️ Failed to bypass StyleLoRA: {e}")
        
        # If bypass failed, at least clear the problematic lora_name
        if not bypass_success:
            print("🔧 Bypass failed, attempting to clear StyleLoRA lora_name...")
            set_node_input_by_title("StyleLoRA", "lora_name", "None")

    # Optional bypass rewiring for nodes specified by UI name
    if bypass_nodes:
        for spec in bypass_nodes:
            try:
                ui_name = spec.get("ui_name")
                passthrough_key = spec.get("passthrough_input_key")
                if not ui_name or not passthrough_key:
                    continue
                output_index = spec.get("output_index", 0)
                remove = spec.get("remove", False)
                wf = bypass_node(
                    wf,
                    target_ui_name=ui_name,
                    passthrough_input_key=passthrough_key,
                    output_index=output_index,
                    remove=remove,
                )
            except Exception:
                # Non-fatal: continue applying remaining patches
                pass

    return wf


def _find_node_ids_by_ui_name(workflow: Dict[str, Any], ui_name: str) -> list[str]:
    """Find node IDs that match the given UI name or title."""
    ids: list[str] = []
    for node_id, node in workflow.items():
        if isinstance(node, dict):
            meta = node.get("_meta", {})
            # Check both _ui_name and title fields
            name = meta.get("_ui_name") or meta.get("title")
            if name == ui_name:
                ids.append(node_id)
    return ids


def bypass_node(
    workflow: Dict[str, Any],
    target_ui_name: str,
    passthrough_input_key: str,
    output_index: int = 0,
    remove: bool = False,
) -> Dict[str, Any]:
    """Bypass a node by UI name by rewiring all consumers to the node's passthrough input.

    This function allows you to "skip" a node in the workflow by redirecting all connections
    that point to the target node to instead point to one of the target node's inputs.

    Args:
        workflow: The ComfyUI workflow JSON dictionary
        target_ui_name: UI name shown in ComfyUI (stored in _meta._ui_name)
        passthrough_input_key: which input of the target node should be forwarded
        output_index: which output slot of the target node to replace (default 0)
        remove: if True, delete the target node after rewiring

    HOW TO GET NODE UI NAMES FROM COMFYUI:
    1. Open your workflow in ComfyUI
    2. Go to Settings (gear icon) → Enable "Developer mode" 
    3. Right-click in the workflow area → "Save (API Format)"
    4. Save as JSON file
    5. Open the JSON file and look for "_meta" sections:
       {
         "1": {
           "_meta": {
             "_ui_name": "CharacterLoRA",     ← This is what you use for target_ui_name
             "title": "Load LoRA"
           },
           "inputs": {
             "model": ["2", 0],              ← This shows input connections
             "clip": ["2", 1],
             "lora_name": "character.safetensors"
           }
         }
       }

    HANDLING MULTI-OUTPUT NODES (e.g., LoRA Loaders):
    LoRA loaders typically have 2 outputs: [0]=model, [1]=clip
    To completely bypass a LoRA loader, you need TWO bypass operations:

    Example bypass_nodes array for skipping "StyleLoRA":
    [
        {
            "ui_name": "StyleLoRA",
            "passthrough_input_key": "model",    ← Forward the model input
            "output_index": 0,                   ← For output slot 0 (model)
            "remove": false                      ← Don't delete yet
        },
        {
            "ui_name": "StyleLoRA", 
            "passthrough_input_key": "clip",     ← Forward the clip input
            "output_index": 1,                   ← For output slot 1 (clip)
            "remove": true                       ← Delete node after this operation
        }
    ]

    COMMON USE CASES:
    - Skip optional LoRA loaders when no LoRA is specified
    - Bypass style nodes for certain generation modes
    - Remove processing nodes for faster inference
    - Disable certain effects conditionally

    NOTE: The order of bypass operations matters for multi-output nodes.
    Always set remove=true only on the LAST operation for the same node.
    """
    ids = _find_node_ids_by_ui_name(workflow, target_ui_name)
    if not ids:
        print(f"🔍 No nodes found with ui_name/title: {target_ui_name}")
        return workflow

    target_id = ids[0]
    target = workflow.get(target_id)
    if not isinstance(target, dict):
        print(f"🔍 Target node {target_id} is not a dict")
        return workflow

    passthrough_value = target.get("inputs", {}).get(passthrough_input_key)
    if passthrough_value is None:
        print(f"🔍 No passthrough value found for {passthrough_input_key} in node {target_id}")
        return workflow

    print(f"🔧 Bypassing {target_ui_name} (node {target_id}): {passthrough_input_key} = {passthrough_value}")

    # Count rewired connections
    rewired_count = 0
    # Rewire all inputs pointing to [target_id, output_index]
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for k, v in list(inputs.items()):
            if isinstance(v, list) and len(v) == 2 and v[0] == target_id and (output_index is None or v[1] == output_index):
                inputs[k] = passthrough_value
                rewired_count += 1
                print(f"🔄 Rewired {node_id}.{k}: [{target_id}, {output_index}] -> {passthrough_value}")

    print(f"🔧 Rewired {rewired_count} connections for {target_ui_name}")

    if remove:
        try:
            del workflow[target_id]
            print(f"🗑️ Removed node {target_id} ({target_ui_name})")
        except Exception as e:
            print(f"⚠️ Failed to remove node {target_id}: {e}")

    return workflow


