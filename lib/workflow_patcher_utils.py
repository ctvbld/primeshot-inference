"""Utility functions for workflow patching."""

from typing import Dict, Any, List, Optional, Tuple
import logging

from .constants import (
    LATENT_NODE_TYPES,
    SAMPLER_NODE_TYPES,
    SAVE_NODE_TYPES,
    IMAGE_PRODUCING_NODE_TYPES
)

logger = logging.getLogger(__name__)


class WorkflowNodeManager:
    """Manages workflow node operations."""
    
    def __init__(self, workflow: Dict[str, Any]):
        self.workflow = workflow
    
    def find_nodes_by_class_type(self, class_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Find all nodes matching a class type."""
        nodes = []
        for node_id, node in self.workflow.items():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                nodes.append((node_id, node))
        return nodes
    
    def find_nodes_by_title(self, title: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Find all nodes matching a title."""
        nodes = []
        wanted_norm = str(title or "").strip().lower()
        
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
                
            meta = node.get("_meta", {})
            node_title = (meta.get("title") or "").strip().lower()
            ui_name = (meta.get("_ui_name") or "").strip().lower()
            
            if node_title == wanted_norm or ui_name == wanted_norm:
                nodes.append((node_id, node))
        
        return nodes
    
    def set_node_input(self, node_id: str, key: str, value: Any) -> bool:
        """Set an input value on a specific node."""
        node = self.workflow.get(node_id)
        if isinstance(node, dict):
            node.setdefault("inputs", {})[key] = value
            return True
        return False
    
    def set_node_input_by_title(self, title: str, key: str, value: Any) -> int:
        """Set input on all nodes matching title. Returns count of updated nodes."""
        updated = 0
        for node_id, node in self.find_nodes_by_title(title):
            if self.set_node_input(node_id, key, value):
                logger.info(f"✅ Updated {title} (node {node_id}) {key} = {value}")
                updated += 1
        
        if updated == 0:
            logger.warning(f"⚠️ No node with title '{title}' found for {key} = {value}")
        
        return updated
    
    def find_image_output_node(self) -> Optional[str]:
        """Find the final image output node (usually VAEDecode)."""
        # First try VAEDecode nodes
        vae_nodes = self.find_nodes_by_class_type("VAEDecode")
        if vae_nodes:
            # Return the last one
            return vae_nodes[-1][0]
        
        # If no VAEDecode, look for other image-producing nodes
        logger.warning("⚠️ No VAEDecode found, searching for alternative image output nodes...")
        
        image_nodes = []
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
                
            class_type = node.get("class_type", "")
            # Check for common image-producing nodes
            if any(x in class_type for x in IMAGE_PRODUCING_NODE_TYPES):
                # Skip save nodes
                if "Save" not in class_type:
                    image_nodes.append((node_id, class_type))
        
        if image_nodes:
            # Use the last image-producing node
            node_id, class_type = image_nodes[-1]
            logger.info(f"✅ Using {class_type} (node {node_id}) as final image source")
            return node_id
        
        return None
    
    def find_special_nodes(self) -> Dict[str, Optional[str]]:
        """Find special nodes like scalers and upscalers."""
        special_nodes = {
            "scale_down": None,
            "resize_by": None,
            "upscale_by": None
        }
        
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
                
            meta_title = node.get("_meta", {}).get("title", "")
            class_type = node.get("class_type", "")
            
            # Scale down node for web saves
            if any(x in class_type for x in ["ScaleDownToSize", "imageScaleDownToSize"]):
                special_nodes["scale_down"] = node_id
            elif meta_title == "Image Scale Down To Size":
                special_nodes["scale_down"] = node_id
            
            # Resize/upscale nodes
            if meta_title == "ResizeBy" or "ResizeBy" in class_type:
                special_nodes["resize_by"] = node_id
            elif meta_title == "UpscaleBy" or "Upscale" in class_type:
                special_nodes["upscale_by"] = node_id
        
        return special_nodes
    
    def is_node_connected(self, node_id: str) -> bool:
        """Check if a node has valid connections to its inputs."""
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return False
        
        inputs = node.get("inputs", {})
        for value in inputs.values():
            if isinstance(value, list) and len(value) == 2:
                # Check if the connected node exists
                connected_id = str(value[0])
                if connected_id not in self.workflow:
                    return False
        
        return True


def normalize_title_key(name: str) -> str:
    """Normalize a title key for matching."""
    try:
        # Remove non-alphanumeric characters and lowercase
        n = ''.join(ch for ch in str(name) if ch.isalnum()).lower()
        return n
    except Exception:
        return str(name or '').lower()


def get_title_aliases() -> Dict[str, str]:
    """Get mapping of normalized names to proper titles."""
    return {
        # Core loaders
        "characterlora": "CharacterLora",
        "stylelora": "StyleLora",
        # Common processing nodes
        "filmgrain": "FilmGrain",
        "channelmixer": "ChannelMixer",
        "lightleaks": "LightLeaks",
        "vibsat": "VibSat",
        # Utility/config nodes
        "upscaleby": "UpscaleBy",
        "resizeby": "ResizeBy",
    }


def get_quality_settings(quality: str) -> Dict[str, Any]:
    """Get quality-specific settings."""
    quality = str(quality or "1K").upper()
    
    settings = {}
    
    if quality == "2K":
        settings["upscale_scale"] = 0.30
        settings["resize_scale"] = 0.28
    elif quality == "4K":
        settings["upscale_scale"] = 0.30
        settings["resize_scale"] = 0.50
    
    return settings
