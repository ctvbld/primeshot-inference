from __future__ import annotations

from typing import Any, Dict, Tuple


def compute_dimensions(quality: str, aspect_ratio: str) -> Tuple[int, int]:
    # Always use 1K base for EmptyHunyuanLatentVideo generation
    # Upscaling to 2K/4K will be handled by conditional upscale nodes
    base_1k = 1024
    
    # Handle supported aspect ratios (1:1, 2:3, 3:2)
    if aspect_ratio == "1:1":
        return base_1k, base_1k  # 1024x1024
    elif aspect_ratio == "3:2":  # Landscape
        return base_1k, int(round(base_1k * 2 / 3))  # 1024x683
    elif aspect_ratio == "2:3":  # Portrait
        return int(round(base_1k * 2 / 3)), base_1k  # 683x1024
    else:
        # Default to square for unknown ratios
        return base_1k, base_1k


def patch_workflow(
    workflow: Dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int | None,
    images_count: int,
    quality: str = "1K",  # Add quality parameter for upscale logic
    lora_filename: str | None = None,
    character_lora: str | None = None,
    style_lora: str | None = None,
    bypass_nodes: list[dict] | None = None,
    enable_previews: bool = True,  # Enable preview generation
    job_id: str | None = None,  # Job ID for output path isolation
    user_id: str | None = None,  # User ID for output path isolation
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

    # Text encoders - try by title first (more specific), then by class type
    set_node_input_by_title("PositivePrompt", "text", prompt)
    set_node_input_by_title("NegativePrompt", "text", negative_prompt)
    
    # Fallback for older workflows without specific titles
    set_node_input("CLIPTextEncode", "text", prompt)
    set_node_input("CLIPTextEncodeNeg", "text", negative_prompt)

    # Seed - find KSampler nodes and update widgets_values[0]
    if seed is not None:
        updated_seed = False
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == "KSampler":
                # KSampler seed is stored in widgets_values[0]
                if "widgets_values" in node and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = int(seed)
                    print(f"✅ Updated KSampler (node {node_id}) seed = {seed}")
                    updated_seed = True
                # Also try inputs format in case the workflow uses that
                elif "inputs" in node:
                    node["inputs"]["seed"] = int(seed)
                    print(f"✅ Updated KSampler (node {node_id}) seed = {seed} (via inputs)")
                    updated_seed = True
        if not updated_seed:
            print(f"⚠️ No KSampler nodes found to update seed = {seed}")

    # Resolution - handle different latent node types
    updated_resolution = False
    for node_type in ["EmptyHunyuanLatentVideo", "EmptyLatentImage", "EmptySD3LatentImage", "EmptyLTXVLatentVideo"]:
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == node_type:
                # Try inputs first (ComfyUI format), then widgets_values (UI format)
                if "inputs" in node:
                    node["inputs"]["width"] = int(width)
                    node["inputs"]["height"] = int(height)
                    print(f"✅ Updated {node_type} (node {node_id}) dimensions = {width}x{height} (via inputs)")
                    updated_resolution = True
                    break
                elif "widgets_values" in node and len(node["widgets_values"]) >= 2:
                    node["widgets_values"][0] = int(width)   # width is usually first
                    node["widgets_values"][1] = int(height)  # height is usually second
                    print(f"✅ Updated {node_type} (node {node_id}) dimensions = {width}x{height} (via widgets_values)")
                    updated_resolution = True
                    break
        if updated_resolution:
            break
    
    # Fallback for older workflows
    if not updated_resolution:
        set_node_input("EmptyLatentImage", "width", int(width))
        set_node_input("EmptyLatentImage", "height", int(height))

    # LoRA injection
    # Priority: explicit character/style targets by title; fallback to generic LoraLoader by _ui_name
    if character_lora:
        # Current workflow uses title "CharacterLoRA" for character loader
        set_node_input_by_title("CharacterLoRA", "lora_name", character_lora)
    if style_lora:
        # Current workflow uses title "StyleLoRA" for optional style loader
        print(f"🎨 Attempting to apply style LoRA: {style_lora}")
        updated_style = False
        
        # Try to find StyleLoRA node by title
        for node_id, node in wf.items():
            if isinstance(node, dict):
                title = node.get("_meta", {}).get("title")
                if title == "StyleLoRA":
                    node.setdefault("inputs", {})["lora_name"] = style_lora
                    print(f"✅ Updated StyleLoRA (node {node_id}) lora_name = {style_lora}")
                    updated_style = True
                    break
        
        if not updated_style:
            print(f"⚠️ No StyleLoRA node found in workflow - style LoRA cannot be applied: {style_lora}")
            print(f"📝 Available node titles: {[node.get('_meta', {}).get('title') for node_id, node in wf.items() if isinstance(node, dict) and node.get('_meta', {}).get('title')]}")
    if lora_filename and not character_lora and not style_lora:
        # Back-compat single lora case
        set_node_input("LoraLoader", "lora_name", lora_filename)

    # NB takes: set batch_size on the appropriate node for multiple images
    # Try common latent generation nodes that support batch_size
    updated_batch = False
    for node_type in ["EmptyHunyuanLatentVideo", "EmptyLatentImage", "EmptySD3LatentImage", "EmptyLTXVLatentVideo"]:
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == node_type:
                # Try inputs first (ComfyUI format), then widgets_values (UI format)
                if "inputs" in node and "batch_size" in node["inputs"]:
                    node["inputs"]["batch_size"] = int(images_count)
                    print(f"✅ Updated {node_type} (node {node_id}) batch_size = {images_count} (via inputs)")
                    updated_batch = True
                    break
                elif "widgets_values" in node and len(node["widgets_values"]) >= 3:
                    node["widgets_values"][-1] = int(images_count)  # batch_size is usually last
                    print(f"✅ Updated {node_type} (node {node_id}) batch_size = {images_count} (via widgets_values)")
                    updated_batch = True
                    break
        if updated_batch:
            break
    
    # Fallback: try setting on KSampler for older workflows
    if not updated_batch:
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

    # TODO: Conditional upscaling based on quality setting
    # For now, EmptyHunyuanLatentVideo always generates at 1K base resolution
    # When upscale nodes are implemented in the workflow, add/enable them based on quality:
    
    if quality == "2K":
        # TODO: Enable 2K upscale nodes when available in workflow
        # Expected nodes: "Upscale2K", "UltimateSDUpscale", or similar
        print(f"🔄 TODO: Enable 2K upscaling nodes (quality={quality})")
        # set_node_input_by_title("Upscale2K", "enabled", True)
        # set_node_input_by_title("Upscale2K", "scale_factor", 2.0)
        
    elif quality == "4K":
        # TODO: Enable 4K upscale nodes when available in workflow  
        # Expected nodes: "Upscale4K", "UltimateSDUpscale", or similar
        print(f"🔄 TODO: Enable 4K upscaling nodes (quality={quality})")
        # set_node_input_by_title("Upscale4K", "enabled", True)
        # set_node_input_by_title("Upscale4K", "scale_factor", 4.0)
        
    else:  # quality == "1K"
        # TODO: Ensure upscale nodes are disabled/bypassed for 1K generation
        print(f"✅ Using base 1K generation (quality={quality}) - no upscaling needed")
        # bypass_upscale_nodes(wf)

    # Set job-specific output path for Save Image node to prevent concurrent job interference
    if job_id and user_id:
        # Use relative path from ComfyUI's default output directory
        # ComfyUI will save to: {output_dir}/{filename_prefix}{counter}_{timestamp}.png
        output_prefix = f"{job_id}/IMG-"
        
        # Update Save Image node to use job-specific path
        updated_save_path = False
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                node.setdefault("inputs", {})["filename_prefix"] = output_prefix
                print(f"✅ Updated SaveImage (node {node_id}) filename_prefix = {output_prefix}")
                updated_save_path = True
        
        if not updated_save_path:
            print(f"⚠️ No SaveImage nodes found to update output path")
    else:
        print(f"⚠️ Missing job_id or user_id - cannot set job-specific output path")

    return wf


def _add_preview_nodes(workflow: Dict[str, Any]) -> None:
    """Enable ComfyUI's built-in real-time preview system.
    
    Instead of adding custom preview nodes, this function ensures that ComfyUI's
    built-in preview system is enabled for real-time previews during sampling.
    """
    try:
        # Find KSampler nodes (main generation nodes)
        sampler_nodes = []
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "KSampler":
                sampler_nodes.append((node_id, node))
        
        if not sampler_nodes:
            print("⚠️ No KSampler nodes found, trying alternative sampling nodes")
            # Look for other sampling node types
            for node_id, node in workflow.items():
                if isinstance(node, dict):
                    class_type = node.get("class_type", "")
                    if "sampler" in class_type.lower() or "sample" in class_type.lower():
                        sampler_nodes.append((node_id, node))
        
        if not sampler_nodes:
            print("⚠️ No sampling nodes found, skipping preview setup")
            return
        
        print(f"🎯 Found {len(sampler_nodes)} sampling nodes for real-time preview setup")
        
        # Enable built-in previews for each sampler by ensuring proper configuration
        for sampler_id, sampler_node in sampler_nodes:
            # Ensure the sampler has proper metadata for preview generation
            if "_meta" not in sampler_node:
                sampler_node["_meta"] = {}
            
            # Enable preview generation during sampling
            sampler_node["_meta"]["preview_enabled"] = True
            sampler_node["_meta"]["preview_method"] = "auto"  # Use ComfyUI's automatic preview
            
            # Also ensure the sampler inputs are configured for preview generation
            # ComfyUI samplers can generate previews automatically during sampling
            if "inputs" not in sampler_node:
                sampler_node["inputs"] = {}
            
            # Some ComfyUI versions use these settings for preview control
            # Note: These might not be standard, but we're ensuring compatibility
            print(f"✅ Enabled real-time previews for sampler {sampler_id} (class: {sampler_node.get('class_type', 'unknown')})")
        
        # Also add a single preview node at the final VAE decode for completed images
        vae_decode_nodes = []
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "VAEDecode":
                vae_decode_nodes.append((node_id, node))
        
        if vae_decode_nodes:
            # Add one preview node for the final output (completed images)
            vae_id, vae_node = vae_decode_nodes[0]  # Use the first VAE decode node
            preview_id = "9999"  # Use a high ID to avoid conflicts
            
            workflow[preview_id] = {
                "class_type": "PreviewImage", 
                "inputs": {
                    "images": [vae_id, 0]  # Connect to VAE decode output
                },
                "_meta": {
                    "title": "FinalPreview",
                    "_ui_name": "FinalPreview",
                    "preview_enabled": True
                }
            }
            
            print(f"✅ Added final preview node {preview_id} for completed images")
        
        print(f"🎨 Successfully configured real-time preview system")
        
    except Exception as e:
        print(f"⚠️ Failed to configure preview system: {e}")
        # Don't fail the entire workflow if preview setup fails


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


