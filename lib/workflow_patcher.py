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
) -> Dict[str, Any]:
    """Apply minimal patches to a ComfyUI workflow JSON.
    This assumes nodes are identified by common labels; you may adjust mapping.
    """
    wf = dict(workflow)

    def set_node_input(node_label: str, key: str, value: Any) -> None:
        for node_id, node in wf.items():
            if isinstance(node, dict) and node.get("_meta", {}).get("_ui_name") == node_label:
                node.setdefault("inputs", {})[key] = value

    # Text encoders
    set_node_input("CLIPTextEncode", "text", prompt)
    set_node_input("CLIPTextEncodeNeg", "text", negative_prompt)

    # Seed, steps, cfg are workflow defaults; set seed if provided
    if seed is not None:
        set_node_input("KSampler", "seed", int(seed))

    # Resolution (EmptyLatentImage path)
    set_node_input("EmptyLatentImage", "width", int(width))
    set_node_input("EmptyLatentImage", "height", int(height))

    # Optional LoRA file name
    if lora_filename:
        set_node_input("LoraLoader", "lora_name", lora_filename)

    # NB takes: prefer batch_size if present; otherwise caller will loop
    set_node_input("KSampler", "batch_size", int(images_count))

    return wf


def _find_node_ids_by_ui_name(workflow: Dict[str, Any], ui_name: str) -> list[str]:
    ids: list[str] = []
    for node_id, node in workflow.items():
        if isinstance(node, dict):
            name = node.get("_meta", {}).get("_ui_name")
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

    - target_ui_name: UI name shown in ComfyUI (stored in _meta._ui_name)
    - passthrough_input_key: which input of the target node should be forwarded
    - output_index: which output slot of the target node to replace (default 0)
    - remove: if True, delete the target node after rewiring
    """
    ids = _find_node_ids_by_ui_name(workflow, target_ui_name)
    if not ids:
        return workflow

    target_id = ids[0]
    target = workflow.get(target_id)
    if not isinstance(target, dict):
        return workflow

    passthrough_value = target.get("inputs", {}).get(passthrough_input_key)
    if passthrough_value is None:
        return workflow

    # Rewire all inputs pointing to [target_id, output_index]
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for k, v in list(inputs.items()):
            if isinstance(v, list) and len(v) == 2 and v[0] == target_id and (output_index is None or v[1] == output_index):
                inputs[k] = passthrough_value

    if remove:
        try:
            del workflow[target_id]
        except Exception:
            pass

    return workflow


