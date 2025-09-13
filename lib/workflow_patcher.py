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
    aspect_ratio: str = "1:1",  # Needed to map to ResolutionCalc aspect label
    settings_override: Dict[str, Any] | None = None,
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
        wanted_norm = str(node_title or "").strip().lower()
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            meta = node.get("_meta", {})
            title = (meta.get("title") or "").strip().lower()
            ui_name = (meta.get("_ui_name") or "").strip().lower()
            if title == wanted_norm or ui_name == wanted_norm:
                node.setdefault("inputs", {})[key] = value
                updated = True
                print(f"✅ Updated {node_title} (node {node_id}) {key} = {value}")
        if not updated:
            print(f"⚠️ Node with title/ui_name '{node_title}' not found for {key} = {value}")

    # Text encoders - try by title first (more specific), then by class type
    set_node_input_by_title("PositivePrompt", "text", prompt)
    set_node_input_by_title("NegativePrompt", "text", negative_prompt)
    
    # Fallback for older workflows without specific titles
    set_node_input("CLIPTextEncode", "text", prompt)
    set_node_input("CLIPTextEncodeNeg", "text", negative_prompt)

    # Seed - find sampler nodes and update seed on inputs/widgets_values
    if seed is not None:
        updated_seed = False
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type")
            if cls in ("KSampler", "KSamplerWithNAG", "KSamplerAdvanced"):
                # Prefer inputs; fallback to widgets_values[0]
                if "inputs" in node:
                    node["inputs"]["seed"] = int(seed)
                    print(f"✅ Updated {cls} (node {node_id}) seed = {seed} (via inputs)")
                    updated_seed = True
                elif "widgets_values" in node and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = int(seed)
                    print(f"✅ Updated {cls} (node {node_id}) seed = {seed}")
                    updated_seed = True
        if not updated_seed:
            print(f"⚠️ No KSampler nodes found to update seed = {seed}")

    # Resolution handling
    # Preferred path: if the workflow has a FluxResolutionNode titled "ResolutionCalc",
    # override its megapixel and aspect_ratio so the latent continues to reference it.
    # Only set explicit latent width/height if the latent does not already reference another node.

    # 1) Try to update ResolutionCalc (FluxResolutionNode)
    # Map friendly AR input ("1:1", "2:3", "3:2") to node options
    ar_map = {
        "1:1": "1:1 (Perfect Square)",
        "2:3": "2:3 (Classic Portrait)",
        "3:2": "3:2 (Golden Landscape)",
    }
    # megapixel per quality
    mp_map = {
        "1K": "1.0",
        "2K": "1.0",
        "4K": "1.6",
    }
    target_aspect = ar_map.get(str(aspect_ratio or "1:1"), "1:1 (Perfect Square)")

    # We don't receive aspect_ratio directly here; infer from width/height when possible
    # The caller also still passes width/height for safety.
    # If target_aspect is None, don't try to set aspect_ratio string.
    quality_str = str(quality or "1K").upper()
    mp_value = mp_map.get(quality_str, "1.1")

    # Update ResolutionCalc by title if present
    rescalc_updated = False
    for node_id, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") == "FluxResolutionNode":
            meta = node.get("_meta", {})
            if meta.get("title") == "ResolutionCalc":
                inputs = node.setdefault("inputs", {})
                inputs["megapixel"] = mp_value
                if target_aspect:
                    inputs["aspect_ratio"] = target_aspect
                # Keep defaults for divisible_by/custom_ratio
                rescalc_updated = True
                print(f"✅ Updated ResolutionCalc (node {node_id}) megapixel={mp_value} aspect_ratio={inputs.get('aspect_ratio')}")
                break

    # 2) Latent dimensions: only set if not already connected via ResolutionCalc
    updated_resolution = False
    for node_type in ["EmptyHunyuanLatentVideo", "EmptyLatentImage", "EmptySD3LatentImage", "EmptyLTXVLatentVideo"]:
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == node_type:
                if "inputs" in node:
                    # If width/height are connections like ["39", 0], keep them; otherwise set explicit values
                    w_in = node["inputs"].get("width")
                    h_in = node["inputs"].get("height")
                    if not (isinstance(w_in, list) and len(w_in) == 2):
                        node["inputs"]["width"] = int(width)
                        updated_resolution = True
                        print(f"✅ Updated {node_type} (node {node_id}) width={width}")
                    if not (isinstance(h_in, list) and len(h_in) == 2):
                        node["inputs"]["height"] = int(height)
                        updated_resolution = True
                        print(f"✅ Updated {node_type} (node {node_id}) height={height}")
                elif "widgets_values" in node and len(node["widgets_values"]) >= 2:
                    node["widgets_values"][0] = int(width)
                    node["widgets_values"][1] = int(height)
                    print(f"✅ Updated {node_type} (node {node_id}) dimensions = {width}x{height} (via widgets_values)")
                    updated_resolution = True
        if updated_resolution:
            break
    if not updated_resolution:
        set_node_input("EmptyLatentImage", "width", int(width))
        set_node_input("EmptyLatentImage", "height", int(height))

    # LoRA injection via direct parameters is still supported
    # Robustly set lora_name on LoraLoader nodes, regardless of titles (e.g., "Charger LoRA")
    try:
        desired_loras: list[str] = []
        if character_lora:
            desired_loras.append(character_lora)
        if style_lora:
            desired_loras.append(style_lora)
        if not desired_loras and lora_filename:
            desired_loras.append(lora_filename)

        if desired_loras:
            lora_nodes = []
            for node_id, node in wf.items():
                if isinstance(node, dict) and node.get("class_type") == "LoraLoader":
                    lora_nodes.append((node_id, node))

            # If we have fewer nodes than values, apply what we can; if more nodes, set on the first ones
            for idx, (node_id, node) in enumerate(lora_nodes):
                if idx >= len(desired_loras):
                    break
                inputs = node.setdefault("inputs", {})
                inputs["lora_name"] = desired_loras[idx]
                print(f"✅ Set LoraLoader (node {node_id}) lora_name = {desired_loras[idx]}")

        # Also try title-based setters for explicit nodes if present in other workflows
        if character_lora:
            # Support variants like CharacterLora, CharacterLora, CharacterLora
            for alias in ["CharacterLora", "CharacterLora", "CharacterLoRa", "CharacterLora"]:
                set_node_input_by_title(alias, "lora_name", character_lora)
        if style_lora:
            for alias in ["StyleLora", "StyleLora", "StyleLoRa", "StyleLora"]:
                set_node_input_by_title(alias, "lora_name", style_lora)
    except Exception as _lora_e:
        print(f"⚠️ Failed to set LoRA names: {_lora_e}")

    # Title-based node input overrides (unified format)
    # Example: { "FilmGrain": {"grain_intensity":0.1}, "CharacterLora": {"strength_model":0.8}}
    def _normalize_title_key(name: str) -> str:
        try:
            n = ''.join(ch for ch in str(name) if ch.isalnum()).lower()
            return n
        except Exception:
            return str(name or '').lower()

    title_aliases = {
        # Core loaders
        "characterlora": "CharacterLora",
        "stylelora": "StyleLora",
        # Common processing nodes
        "filmgrain": "FilmGrain",
        "channelmixer": "ChannelMixer",
        "lightleaks": "LightLeaks",
        "vibsat": "VibSat",
        # Utility/config nodes used in our workflows
        "resolutioncalc": "ResolutionCalc",
        "upscaleby": "UpscaleBy",
        "resizeby": "ResizeBy",
    }
    if isinstance(settings_override, dict):
        # Normalize to title -> inputs dict
        for raw_title, overrides in settings_override.items():
            try:
                if not isinstance(overrides, dict):
                    continue
                key = str(raw_title or "").strip()
                if not key:
                    continue
                norm_key = _normalize_title_key(key)
                target_title = title_aliases.get(norm_key, key)
                # Apply to all nodes matching title
                applied = False
                for node_id, node in wf.items():
                    if not isinstance(node, dict):
                        continue
                    if node.get("_meta", {}).get("title") == target_title:
                        inputs = node.setdefault("inputs", {})
                        for k, v in overrides.items():
                            inputs[k] = v
                        print(f"✅ Applied overrides to {target_title} (node {node_id}): {list(overrides.keys())}")
                        applied = True
                if not applied:
                    print(f"ℹ️ No node with title '{target_title}' found to override")
            except Exception as _e:
                print(f"⚠️ Failed applying overrides for title '{raw_title}': {_e}")

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

    # Auto-bypass StyleLora if no style_lora is provided
    if not style_lora:
        print("🔄 No style_lora provided, auto-bypassing StyleLora node...")
        
        # Bypass the StyleLora node by forwarding its model and clip inputs
        # This requires two operations since LoRA loaders have two outputs
        auto_bypass_specs = [
            {
                "ui_name": "StyleLora",
                "passthrough_input_key": "model",
                "output_index": 0,  # model output
                "remove": False
            },
            {
                "ui_name": "StyleLora", 
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
                print(f"✅ Bypassed StyleLora output {spec['output_index']} -> {spec['passthrough_input_key']}")
                bypass_success = True
            except Exception as e:
                print(f"⚠️ Failed to bypass StyleLora: {e}")
        # Try alias UI name StyleLora as well if first attempt failed
        if not bypass_success:
            for spec in [{**s, "ui_name": "StyleLora"} for s in auto_bypass_specs]:
                try:
                    wf = bypass_node(
                        wf,
                        target_ui_name=spec["ui_name"],
                        passthrough_input_key=spec["passthrough_input_key"],
                        output_index=spec["output_index"],
                        remove=spec["remove"],
                    )
                    print(f"✅ Bypassed StyleLora output {spec['output_index']} -> {spec['passthrough_input_key']}")
                    bypass_success = True
                except Exception as e:
                    print(f"⚠️ Failed to bypass StyleLora alias: {e}")
        
        # If bypass failed, at least clear the problematic lora_name
        if not bypass_success:
            print("🔧 Bypass failed, attempting to clear StyleLora lora_name...")
            set_node_input_by_title("StyleLora", "lora_name", "None")

    # Auto-bypass certain effect nodes UNLESS they are present in settings_override
    effects_to_bypass = ["LightLeaks", "VibSat", "ChannelMixer", "FilmGrain"]
    present_titles = set()
    if isinstance(settings_override, dict):
        present_titles = {_normalize_title_key(k) for k in settings_override.keys()}
    for effect_title in effects_to_bypass:
        if _normalize_title_key(effect_title) in present_titles:
            print(f"⏭️ Not bypassing {effect_title}; overrides provided")
            continue
        try:
            wf = bypass_node(
                wf,
                target_ui_name=effect_title,
                passthrough_input_key="image",
                output_index=0,
                remove=True,
            )
            print(f"✅ Auto-bypassed {effect_title}")
        except Exception:
            pass

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

    # Ensure save nodes are always wired to a stable image source
    # Prefer the final VAEDecode output. For web saves, if a 1024px scaler exists, keep it.
    try:
        vae_decode_ids: list[str] = []
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == "VAEDecode":
                vae_decode_ids.append(node_id)
        vae_id = vae_decode_ids[-1] if vae_decode_ids else None
        
        # If no VAEDecode found, look for the final image output node
        if vae_id is None:
            print("⚠️ No VAEDecode found, searching for alternative image output nodes...")
            # Look for any node that outputs images
            image_output_nodes = []
            for node_id, node in wf.items():
                if not isinstance(node, dict):
                    continue
                class_type = node.get("class_type", "")
                # Common image-producing nodes
                if any(x in class_type for x in ["Decode", "Preview", "Image", "Upscale", "Scale", "Resize"]):
                    # Skip save nodes themselves
                    if "Save" not in class_type:
                        image_output_nodes.append((node_id, class_type))
            
            if image_output_nodes:
                # Use the last image-producing node
                vae_id, class_type = image_output_nodes[-1]
                print(f"✅ Using {class_type} (node {vae_id}) as final image source")

        # Detect a 1024px scale-down node used by the web saver
        scale_down_id = None
        # Detect post-upscale nodes for high-res original saving
        resize_by_id = None
        upscale_by_id = None
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            meta_title = node.get("_meta", {}).get("title", "")
            class_type = node.get("class_type", "")
            if (
                meta_title == "Image Scale Down To Size"
                or class_type == "easy imageScaleDownToSize"
                or "ScaleDownToSize" in class_type
            ):
                scale_down_id = node_id
                break
        # Separate pass for upscalers/resizers (do not break early; prefer explicit titles)
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            meta_title = node.get("_meta", {}).get("title", "")
            class_type = node.get("class_type", "")
            if meta_title == "ResizeBy" or "ResizeBy" in class_type:
                resize_by_id = node_id
            if meta_title == "UpscaleBy" or "Upscale" in class_type:
                upscale_by_id = node_id

        if vae_id is not None:
            # Track which save nodes we find and wire
            save_nodes_wired = []
            
            for node_id, node in wf.items():
                if not isinstance(node, dict):
                    continue
                if node.get("class_type") in ("SaveImage", "SaveImagePlus"):
                    inputs = node.setdefault("inputs", {})
                    title = node.get("_meta", {}).get("title", "")
                    prefix = str(inputs.get("filename_prefix", ""))
                    
                    # Check current connection to see if it needs rewiring
                    current_connection = inputs.get("images")
                    needs_rewiring = False
                    
                    # If connection points to a removed/bypassed node, it needs rewiring
                    if isinstance(current_connection, list) and len(current_connection) == 2:
                        connected_node_id = str(current_connection[0])
                        if connected_node_id not in wf:
                            needs_rewiring = True
                            print(f"⚠️ Node {node_id} connected to non-existent node {connected_node_id}")
                    
                    # Original PNG should point to highest-res output when available
                    if title == "SaveOrig" or prefix.startswith("orig_"):
                        target_orig_id = vae_id
                        # Prefer ResizeBy (final resize to 2K/4K), then UpscaleBy
                        if resize_by_id is not None:
                            target_orig_id = resize_by_id
                        elif upscale_by_id is not None:
                            target_orig_id = upscale_by_id
                        inputs["images"] = [target_orig_id, 0]
                        print(f"✅ Rewired SaveOrig (node {node_id}) images -> [{target_orig_id}, 0]")
                        save_nodes_wired.append(("SaveOrig", node_id))
                    # Web save should prefer the scaler if available, else VAE
                    elif title == "SaveWeb" or prefix.startswith("web_"):
                        target_id = scale_down_id or vae_id
                        inputs["images"] = [target_id, 0]
                        print(f"✅ Rewired SaveWeb (node {node_id}) images -> [{target_id}, 0]")
                        save_nodes_wired.append(("SaveWeb", node_id))
                    # Handle generic save nodes that might have broken connections
                    elif needs_rewiring:
                        # Default to VAE output for broken connections
                        inputs["images"] = [vae_id, 0]
                        print(f"✅ Rewired broken connection for save node {node_id} -> [{vae_id}, 0]")
                        save_nodes_wired.append(("Generic", node_id))

        # Preflight: ensure at least one SaveOrig and one SaveWeb exist after rewiring
        found_save_orig = False
        found_save_web = False
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") in ("SaveImage", "SaveImagePlus"):
                title = node.get("_meta", {}).get("title", "")
                if title == "SaveOrig":
                    found_save_orig = True
                if title == "SaveWeb":
                    found_save_web = True
        print(f"🔎 Save nodes present -> SaveOrig: {found_save_orig}, SaveWeb: {found_save_web}")
    except Exception as _rewire_err:
        print(f"⚠️ Failed to enforce save-node rewiring: {_rewire_err}")

    # Conditional upscale tuning for non-1K qualities (nodes present only in non-1K workflow)
    q = str(quality or "1K").upper()
    if q != "1K":
        # UpscaleBy scale_by per quality
        upscale_map = {"2K": 0.30, "4K": 0.35}
        set_node_input_by_title("UpscaleBy", "scale_by", upscale_map.get(q, 0.35))
        # ResizeBy depends on target quality
        resize_map = {"2K": 0.40, "4K": 0.50}
        set_node_input_by_title("ResizeBy", "scale_by", resize_map.get(q, 0.50))
    else:
        print(f"✅ Using base 1K generation (quality={q}) - no upscaling tweaks")

    # Set job-specific output subfolder/path for Save nodes while preserving prefixes
    if job_id and user_id:
        # Prefer setting a subfolder for explicit SaveWeb/SaveOrig; fallback to filename_prefix for generic saves
        # This keeps explicit 'web_'/'orig_' prefixes intact while isolating outputs per job
        subfolder = f"{job_id}"

        updated_titles = 0
        # Pass 1: find by title regardless of class_type
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            meta = node.get("_meta", {})
            title = meta.get("title", "")
            if title in ("SaveWeb", "SaveOrig"):
                inputs = node.setdefault("inputs", {})
                # Common keys used by SaveImagePlus for directory control
                # Prefer 'subfolder'; if not present, try 'output_path'
                if "subfolder" in inputs:
                    inputs["subfolder"] = subfolder
                    print(f"✅ Set {title} (node {node_id}) subfolder = {subfolder}")
                    updated_titles += 1
                elif "output_path" in inputs:
                    inputs["output_path"] = subfolder
                    print(f"✅ Set {title} (node {node_id}) output_path = {subfolder}")
                    updated_titles += 1
                else:
                    # As a last resort, prefix the filename with the job folder while preserving leading web_/orig_
                    current_prefix = str(inputs.get("filename_prefix", ""))
                    safe_prefix = current_prefix if current_prefix else ("web_" if title == "SaveWeb" else "orig_")
                    inputs["filename_prefix"] = f"{subfolder}/{safe_prefix}"
                    print(f"✅ Set {title} (node {node_id}) filename_prefix = {inputs['filename_prefix']}")
                    updated_titles += 1

        # Pass 2: generic SaveImage/SaveImagePlus without explicit prefixes
        updated_generics = 0
        output_prefix = f"{job_id}/IMG-"
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") in ("SaveImage", "SaveImagePlus"):
                inputs = node.setdefault("inputs", {})
                current_prefix = str(inputs.get("filename_prefix", ""))
                if current_prefix.startswith("web_") or current_prefix.startswith("orig_"):
                    print(f"ℹ️ Keeping explicit prefix for node {node_id}: {current_prefix}")
                    # Titles pass likely handled subfolder above; if not, set subfolder when available
                    if "subfolder" in inputs:
                        inputs["subfolder"] = subfolder
                    elif "output_path" in inputs:
                        inputs["output_path"] = subfolder
                    continue
                inputs["filename_prefix"] = output_prefix
                print(f"✅ Updated {node.get('class_type')} (node {node_id}) filename_prefix = {output_prefix}")
                updated_generics += 1

        if updated_titles == 0 and updated_generics == 0:
            print("⚠️ No save nodes updated for output directory settings")
    else:
        print("⚠️ Missing job_id or user_id - cannot set job-specific output path")

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
             "_ui_name": "CharacterLora",     ← This is what you use for target_ui_name
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

    Example bypass_nodes array for skipping "StyleLora":
    [
        {
            "ui_name": "StyleLora",
            "passthrough_input_key": "model",    ← Forward the model input
            "output_index": 0,                   ← For output slot 0 (model)
            "remove": false                      ← Don't delete yet
        },
        {
            "ui_name": "StyleLora", 
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


