"""Workflow patching module for ComfyUI inference."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
import logging

from .workflow_patcher_utils import (
    WorkflowNodeManager,
    normalize_title_key,
    get_title_aliases,
    get_quality_settings
)
from .constants import (
    LATENT_NODE_TYPES,
    SAMPLER_NODE_TYPES,
    SAVE_NODE_TYPES,
    DEFAULT_SEED_MAX
)

logger = logging.getLogger(__name__)


def compute_dimensions(quality: str, aspect_ratio: str) -> Tuple[int, int]:
    """Compute dimensions based on quality and aspect ratio.
    
    Returns explicit ImageLatent width/height per quality and aspect ratio.
    """
    q = str(quality or "1K").upper()
    ar = str(aspect_ratio or "1:1")
    
    # 1K mappings
    if q == "1K":
        if ar == "1:1":
            return 1280, 1280
        if ar == "2:3":
            return 904, 1408
        if ar == "3:2":
            return 1408, 904
        return 1280, 1280
    
    # 2K mappings
    if q == "2K":
        if ar == "1:1":
            return 1408, 1408
        if ar == "2:3":
            return 960, 1471
        if ar == "3:2":
            return 1471, 960
        return 1408, 1408
    
    # 4K mappings
    if q == "4K":
        if ar == "1:1":
            return 1536, 1536
        if ar == "2:3":
            return 1072, 1688
        if ar == "3:2":
            return 1688, 1072
        return 1536, 1536
    
    # Fallback: default to 1K square
    return 1280, 1280


class WorkflowPatcher:
    """Handles workflow patching operations."""
    
    def __init__(self, workflow: Dict[str, Any]):
        self.workflow = dict(workflow)  # Create a copy
        self.manager = WorkflowNodeManager(self.workflow)
        
    def patch(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int | None,
        images_count: int,
        quality: str = "1K",
        aspect_ratio: str = "1:1",
        settings_override: Dict[str, Any] | None = None,
        lora_filename: str | None = None,
        character_lora: str | None = None,
        style_lora: str | None = None,
        bypass_nodes: list[dict] | None = None,
        enable_previews: bool = True,
        job_id: str | None = None,
        user_id: str | None = None,
    ) -> Dict[str, Any]:
        """Apply patches to workflow and return the modified version."""
        
        # Apply basic patches
        self._patch_prompts(prompt, negative_prompt)
        self._patch_seed(seed)
        self._patch_resolution(width, height, quality, aspect_ratio)
        self._patch_batch_size(images_count)
        
        # Apply LoRA patches
        self._patch_loras(character_lora, style_lora, lora_filename)
        
        # Apply settings overrides
        if settings_override:
            self._apply_settings_override(settings_override)
        
        # Auto-bypass nodes
        self._auto_bypass_nodes(style_lora, settings_override)
        
        # Apply manual bypass specifications
        if bypass_nodes:
            self._apply_bypass_nodes(bypass_nodes)
        
        # Ensure save nodes are properly wired
        self._ensure_save_node_wiring()
        
        # Apply quality-specific settings
        self._apply_quality_settings(quality)
        
        # Set output paths
        if job_id and user_id:
            self._set_output_paths(job_id, user_id)
        
        return self.workflow
    
    def _patch_prompts(self, prompt: str, negative_prompt: str) -> None:
        """Patch positive and negative prompts."""
        # Try by title first (more specific)
        self.manager.set_node_input_by_title("PositivePrompt", "text", prompt)
        self.manager.set_node_input_by_title("NegativePrompt", "text", negative_prompt)
        
        # Fallback for older workflows
        for node_id, node in self.manager.find_nodes_by_class_type("CLIPTextEncode"):
            # Simple heuristic: if it has "neg" in any metadata, it's negative
            meta = node.get("_meta", {})
            is_negative = any("neg" in str(v).lower() for v in meta.values())
            
            if is_negative:
                self.manager.set_node_input(node_id, "text", negative_prompt)
            else:
                self.manager.set_node_input(node_id, "text", prompt)
    
    def _patch_seed(self, seed: int | None) -> None:
        """Patch seed values on sampler nodes."""
        if seed is None:
            return
        
        updated = False
        for class_type in SAMPLER_NODE_TYPES:
            for node_id, node in self.manager.find_nodes_by_class_type(class_type):
                # Prefer inputs over widgets_values
                if "inputs" in node:
                    node["inputs"]["seed"] = int(seed)
                    logger.info(f"✅ Updated {class_type} (node {node_id}) seed = {seed}")
                    updated = True
                elif "widgets_values" in node and len(node["widgets_values"]) > 0:
                    node["widgets_values"][0] = int(seed)
                    logger.info(f"✅ Updated {class_type} (node {node_id}) seed = {seed} (via widgets)")
                    updated = True
        
        if not updated:
            logger.warning(f"⚠️ No sampler nodes found to update seed = {seed}")
    
    def _patch_resolution(self, width: int, height: int, quality: str, aspect_ratio: str) -> None:
        """Patch resolution settings."""
        # Update latent nodes
        updated = False
        for node_type in LATENT_NODE_TYPES:
            for node_id, node in self.manager.find_nodes_by_class_type(node_type):
                if "inputs" in node:
                    # Only set if not connected to another node
                    w_in = node["inputs"].get("width")
                    h_in = node["inputs"].get("height")
                    
                    if not isinstance(w_in, list):
                        node["inputs"]["width"] = int(width)
                        updated = True
                    if not isinstance(h_in, list):
                        node["inputs"]["height"] = int(height)
                        updated = True
                    
                    if updated:
                        logger.info(f"✅ Updated {node_type} (node {node_id}) dimensions")
                        
                elif "widgets_values" in node and len(node["widgets_values"]) >= 2:
                    node["widgets_values"][0] = int(width)
                    node["widgets_values"][1] = int(height)
                    logger.info(f"✅ Updated {node_type} (node {node_id}) dimensions (via widgets)")
                    updated = True
            
            if updated:
                break
    
    def _patch_batch_size(self, images_count: int) -> None:
        """Patch batch size for multiple image generation."""
        updated = False
        
        # Try latent nodes first
        for node_type in LATENT_NODE_TYPES:
            for node_id, node in self.manager.find_nodes_by_class_type(node_type):
                if "inputs" in node and "batch_size" in node["inputs"]:
                    node["inputs"]["batch_size"] = int(images_count)
                    logger.info(f"✅ Updated {node_type} (node {node_id}) batch_size = {images_count}")
                    updated = True
                    break
                elif "widgets_values" in node and len(node["widgets_values"]) >= 3:
                    node["widgets_values"][-1] = int(images_count)
                    logger.info(f"✅ Updated {node_type} (node {node_id}) batch_size = {images_count} (via widgets)")
                    updated = True
                    break
            
            if updated:
                break
        
        # Fallback to sampler nodes
        if not updated:
            self.manager.set_node_input_by_title("KSampler", "batch_size", int(images_count))
    
    def _patch_loras(self, character_lora: str | None, style_lora: str | None, lora_filename: str | None) -> None:
        """Patch LoRA loader nodes."""
        # Collect desired LoRAs
        desired_loras = []
        if character_lora:
            desired_loras.append(character_lora)
        if style_lora:
            desired_loras.append(style_lora)
        if not desired_loras and lora_filename:
            desired_loras.append(lora_filename)
        
        if not desired_loras:
            return
        
        # Find all LoRA loader nodes
        lora_nodes = self.manager.find_nodes_by_class_type("LoraLoader")
        
        # Apply LoRAs to available nodes
        for idx, (node_id, node) in enumerate(lora_nodes):
            if idx >= len(desired_loras):
                break
            
            inputs = node.setdefault("inputs", {})
            inputs["lora_name"] = desired_loras[idx]
            logger.info(f"✅ Set LoraLoader (node {node_id}) lora_name = {desired_loras[idx]}")
        
        # Also try title-based setters
        if character_lora:
            for alias in ["CharacterLora", "CharacterLoRa"]:
                self.manager.set_node_input_by_title(alias, "lora_name", character_lora)
        
        if style_lora:
            for alias in ["StyleLora", "StyleLoRa"]:
                self.manager.set_node_input_by_title(alias, "lora_name", style_lora)
    
    def _apply_settings_override(self, settings_override: Dict[str, Any]) -> None:
        """Apply title-based node input overrides and handle bypass flags."""
        if settings_override:
            print(f"🔧 SETTINGS_OVERRIDE DEBUG: Processing settings_override: {settings_override}")
            logger.info(f"🔧 Processing settings_override: {settings_override}")

        title_aliases = get_title_aliases()
        
        # Collect bypass specifications
        bypass_specs = []

        for raw_title, overrides in settings_override.items():
            if not isinstance(overrides, dict):
                print(f"⚠️ SETTINGS_OVERRIDE DEBUG: Skipping non-dict override for '{raw_title}': {overrides}")
                logger.warning(f"⚠️ Skipping non-dict override for '{raw_title}': {overrides}")
                continue

            key = str(raw_title or "").strip()
            if not key:
                print(f"⚠️ SETTINGS_OVERRIDE DEBUG: Skipping empty title key: {raw_title}")
                logger.warning(f"⚠️ Skipping empty title key: {raw_title}")
                continue

            norm_key = normalize_title_key(key)
            target_title = title_aliases.get(norm_key, key)

            # Check for bypass flag
            should_bypass = overrides.get("__bypass__", False)
            passthrough_key = overrides.get("__passthrough_key__", "image")
            output_index = overrides.get("__output_index__", 0)
            
            if should_bypass:
                # Collect bypass specification
                bypass_specs.append({
                    "ui_name": target_title,
                    "passthrough_input_key": passthrough_key,
                    "output_index": output_index,
                    "remove": True
                })
                logger.info(f"🔄 Will bypass {target_title} (passthrough: {passthrough_key}, output: {output_index})")
                
                # Skip applying other settings to this node since it will be bypassed
                continue

            print(f"🔧 SETTINGS_OVERRIDE DEBUG: Processing overrides for '{raw_title}' (normalized: '{norm_key}' → target: '{target_title}')")
            logger.info(f"🔧 Processing overrides for '{raw_title}' (normalized: '{norm_key}' → target: '{target_title}')")

            # Filter out bypass control keys before applying
            regular_overrides = {k: v for k, v in overrides.items() 
                                if not k.startswith("__")}
            
            if not regular_overrides:
                continue

            # Apply to all matching nodes
            applied = False
            for node_id, node in self.manager.find_nodes_by_title(target_title):
                inputs = node.setdefault("inputs", {})
                print(f"🔧 SETTINGS_OVERRIDE DEBUG: Applying {len(regular_overrides)} overrides to {target_title} (node {node_id}):")
                logger.info(f"🔧 Applying {len(regular_overrides)} overrides to {target_title} (node {node_id}):")

                for k, v in regular_overrides.items():
                    old_value = inputs.get(k)
                    inputs[k] = v
                    print(f"  📝 SETTINGS_OVERRIDE DEBUG: {k}: {old_value} → {v}")
                    logger.info(f"  📝 {k}: {old_value} → {v}")

                print(f"✅ SETTINGS_OVERRIDE DEBUG: Applied overrides to {target_title} (node {node_id}): {list(regular_overrides.keys())}")
                logger.info(f"✅ Applied overrides to {target_title} (node {node_id}): {list(regular_overrides.keys())}")
                applied = True

            if not applied:
                print(f"⚠️ SETTINGS_OVERRIDE DEBUG: No node with title '{target_title}' found to override")
                logger.warning(f"⚠️ No node with title '{target_title}' found to override")
        
        # Apply collected bypasses
        if bypass_specs:
            logger.info(f"🔄 Applying {len(bypass_specs)} bypass specifications from settings")
            self._apply_bypass_nodes(bypass_specs)
    
    def _auto_bypass_nodes(self, style_lora: str | None, settings_override: Dict[str, Any] | None) -> None:
        """Auto-bypass certain nodes based on configuration."""
        # Auto-bypass StyleLora if no style_lora provided
        if not style_lora:
            logger.info("🔄 No style_lora provided, auto-bypassing StyleLora node...")
            
            # Bypass both outputs of LoRA loader
            for output_idx, input_key in [(0, "model"), (1, "clip")]:
                try:
                    self.workflow = bypass_node(
                        self.workflow,
                        target_ui_name="StyleLora",
                        passthrough_input_key=input_key,
                        output_index=output_idx,
                        remove=(output_idx == 1)  # Remove on last operation
                    )
                    logger.info(f"✅ Bypassed StyleLora output {output_idx}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to bypass StyleLora: {e}")
        
        # Auto-bypass effect nodes unless overridden
        effects_to_bypass = ["LightLeaks", "ChannelMixer", "FilmGrain"]
        
        present_titles = set()
        if isinstance(settings_override, dict):
            present_titles = {normalize_title_key(k) for k in settings_override.keys()}
        
        for effect_title in effects_to_bypass:
            if normalize_title_key(effect_title) in present_titles:
                logger.info(f"⏭️ Not bypassing {effect_title}; overrides provided")
                continue
            
            try:
                self.workflow = bypass_node(
                    self.workflow,
                    target_ui_name=effect_title,
                    passthrough_input_key="image",
                    output_index=0,
                    remove=True,
                )
                logger.info(f"✅ Auto-bypassed {effect_title}")
            except Exception:
                pass
    
    def _apply_bypass_nodes(self, bypass_nodes: List[Dict[str, Any]]) -> None:
        """Apply manual bypass specifications."""
        for spec in bypass_nodes:
            try:
                ui_name = spec.get("ui_name")
                passthrough_key = spec.get("passthrough_input_key")
                
                if not ui_name or not passthrough_key:
                    continue
                
                output_index = spec.get("output_index", 0)
                remove = spec.get("remove", False)
                
                self.workflow = bypass_node(
                    self.workflow,
                    target_ui_name=ui_name,
                    passthrough_input_key=passthrough_key,
                    output_index=output_index,
                    remove=remove,
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to bypass {ui_name}: {e}")
    
    def _ensure_save_node_wiring(self) -> None:
        """Ensure save nodes are properly wired to image sources."""
        # Find final image output
        vae_id = self.manager.find_image_output_node()
        if not vae_id:
            logger.warning("⚠️ No image output node found for save node wiring")
            return
        
        # Find special nodes
        special_nodes = self.manager.find_special_nodes()
        scale_down_id = special_nodes["scale_down"]
        resize_by_id = special_nodes["resize_by"]
        upscale_by_id = special_nodes["upscale_by"]
        
        # Wire save nodes
        save_nodes_wired = []
        
        for node_type in SAVE_NODE_TYPES:
            for node_id, node in self.manager.find_nodes_by_class_type(node_type):
                inputs = node.setdefault("inputs", {})
                title = node.get("_meta", {}).get("title", "")
                prefix = str(inputs.get("filename_prefix", ""))
                
                # Check if rewiring needed
                current = inputs.get("images")
                needs_rewiring = False
                
                if isinstance(current, list) and len(current) == 2:
                    connected_id = str(current[0])
                    if connected_id not in self.workflow:
                        needs_rewiring = True
                        logger.warning(f"⚠️ Node {node_id} connected to non-existent node {connected_id}")
                
                # Wire based on node type
                if title == "SaveOrig" or prefix.startswith("orig_"):
                    # Prefer highest resolution output
                    target = vae_id
                    if resize_by_id:
                        target = resize_by_id
                    elif upscale_by_id:
                        target = upscale_by_id
                    
                    inputs["images"] = [target, 0]
                    logger.info(f"✅ Wired SaveOrig (node {node_id}) -> [{target}, 0]")
                    save_nodes_wired.append(("SaveOrig", node_id))
                    
                elif needs_rewiring:
                    # Generic broken connection
                    inputs["images"] = [vae_id, 0]
                    logger.info(f"✅ Rewired broken save node {node_id} -> [{vae_id}, 0]")
                    save_nodes_wired.append(("Generic", node_id))
        
        # Log summary
        logger.info(f"🔎 Save nodes wired: {save_nodes_wired}")
    
    def _apply_quality_settings(self, quality: str) -> None:
        """Apply quality-specific settings."""
        if quality == "1K":
            logger.info("✅ Using base 1K generation - no upscaling tweaks")
            return
        
        settings = get_quality_settings(quality)
        
        # Update upscale/resize nodes
        self.manager.set_node_input_by_title("UpscaleBy", "scale_by", settings["upscale_scale"])
        self.manager.set_node_input_by_title("ResizeBy", "scale_by", settings["resize_scale"])
    
    def _set_output_paths(self, job_id: str, user_id: str) -> None:
        """Set job-specific output paths for save nodes."""
        subfolder = f"{job_id}"
        output_prefix = f"{job_id}/IMG-"
        
        # Update titled nodes first
        updated_titles = 0
        for title in ["SaveOrig"]:
            for node_id, node in self.manager.find_nodes_by_title(title):
                inputs = node.setdefault("inputs", {})
                
                # Try different path settings
                if "subfolder" in inputs:
                    inputs["subfolder"] = subfolder
                    logger.info(f"✅ Set {title} (node {node_id}) subfolder = {subfolder}")
                    updated_titles += 1
                elif "output_path" in inputs:
                    inputs["output_path"] = subfolder
                    logger.info(f"✅ Set {title} (node {node_id}) output_path = {subfolder}")
                    updated_titles += 1
                else:
                    # Fallback to filename prefix
                    current_prefix = str(inputs.get("filename_prefix", ""))
                    safe_prefix = current_prefix if current_prefix else (
                        "orig_"
                    )
                    inputs["filename_prefix"] = f"{subfolder}/{safe_prefix}"
                    logger.info(f"✅ Set {title} (node {node_id}) filename_prefix = {inputs['filename_prefix']}")
                    updated_titles += 1
        
        # Update generic save nodes
        updated_generics = 0
        for node_type in SAVE_NODE_TYPES:
            for node_id, node in self.manager.find_nodes_by_class_type(node_type):
                inputs = node.setdefault("inputs", {})
                current_prefix = str(inputs.get("filename_prefix", ""))
                
                # Skip nodes with explicit prefixes
                if current_prefix.startswith(("web_", "orig_")):
                    logger.info(f"ℹ️ Keeping explicit prefix for node {node_id}: {current_prefix}")
                    # Set subfolder if available
                    if "subfolder" in inputs:
                        inputs["subfolder"] = subfolder
                    elif "output_path" in inputs:
                        inputs["output_path"] = subfolder
                    continue
                
                inputs["filename_prefix"] = output_prefix
                logger.info(f"✅ Updated {node_type} (node {node_id}) filename_prefix = {output_prefix}")
                updated_generics += 1
        
        if updated_titles == 0 and updated_generics == 0:
            logger.warning("⚠️ No save nodes updated for output directory settings")


def patch_workflow(
    workflow: Dict[str, Any],
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int | None,
    images_count: int,
    quality: str = "1K",
    aspect_ratio: str = "1:1",
    settings_override: Dict[str, Any] | None = None,
    lora_filename: str | None = None,
    character_lora: str | None = None,
    style_lora: str | None = None,
    bypass_nodes: list[dict] | None = None,
    enable_previews: bool = True,
    job_id: str | None = None,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """Apply minimal patches to a ComfyUI workflow JSON.
    
    This is the main entry point for workflow patching.
    
    Settings Override with Bypass Support:
        The settings_override parameter now supports bypassing nodes using special keys:
        
        Example:
        {
            "VibSat": {
                "__bypass__": true,
                "__passthrough_key__": "image",  # Optional, defaults to "image"
                "__output_index__": 0            # Optional, defaults to 0
            },
            "FilmGrain": {
                "grain_intensity": 0.1           # Regular parameter override
            }
        }
        
        Bypass control keys (all optional):
        - "__bypass__": Set to true to bypass this node
        - "__passthrough_key__": Which input to forward through (default: "image")
        - "__output_index__": Which output slot to replace (default: 0)
    """
    patcher = WorkflowPatcher(workflow)
    return patcher.patch(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        images_count=images_count,
        quality=quality,
        aspect_ratio=aspect_ratio,
        settings_override=settings_override,
        lora_filename=lora_filename,
        character_lora=character_lora,
        style_lora=style_lora,
        bypass_nodes=bypass_nodes,
        enable_previews=enable_previews,
        job_id=job_id,
        user_id=user_id,
    )


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
        logger.info(f"🔍 No nodes found with ui_name/title: {target_ui_name}")
        return workflow

    target_id = ids[0]
    target = workflow.get(target_id)
    if not isinstance(target, dict):
        logger.info(f"🔍 Target node {target_id} is not a dict")
        return workflow

    passthrough_value = target.get("inputs", {}).get(passthrough_input_key)
    if passthrough_value is None:
        logger.info(f"🔍 No passthrough value found for {passthrough_input_key} in node {target_id}")
        return workflow

    logger.info(f"🔧 Bypassing {target_ui_name} (node {target_id}): {passthrough_input_key} = {passthrough_value}")

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
                logger.info(f"🔄 Rewired {node_id}.{k}: [{target_id}, {output_index}] -> {passthrough_value}")

    logger.info(f"🔧 Rewired {rewired_count} connections for {target_ui_name}")

    if remove:
        try:
            del workflow[target_id]
            logger.info(f"🗑️ Removed node {target_id} ({target_ui_name})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove node {target_id}: {e}")

    return workflow


# Preview node support (kept for compatibility, but not actively used)
def _add_preview_nodes(workflow: Dict[str, Any]) -> None:
    """Enable ComfyUI's built-in real-time preview system.
    
    This function is kept for compatibility but may not be actively used
    as ComfyUI has its own preview system.
    """
    pass