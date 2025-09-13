"""LoRA model management utilities."""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .constants import LORA_MODELS_DIR, S3_MOUNT_PATH, WORKFLOWS_MOUNT_PATH

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manages LoRA model linking and cleanup."""
    
    def __init__(self, lora_dir: str = LORA_MODELS_DIR):
        self.lora_dir = Path(lora_dir)
        self.created_links: List[str] = []
        
    def link_lora(self, source_path: str, job_id: str) -> Optional[str]:
        """Link a LoRA model with job-scoped naming.
        
        Args:
            source_path: Absolute path to the LoRA file
            job_id: Job ID for scoping the filename
            
        Returns:
            The linked filename if successful, None otherwise
        """
        if not source_path:
            return None
            
        try:
            source = Path(source_path)
            if not source.exists():
                logger.warning(f"LoRA source not found: {source_path}")
                return None
            
            # Ensure directory exists
            self.lora_dir.mkdir(parents=True, exist_ok=True)
            
            # Create job-scoped filename
            unique_name = f"{job_id}__{source.name}"
            dest = self.lora_dir / unique_name
            
            # Remove existing link if present
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            
            # Try symlink first, fall back to copy
            try:
                dest.symlink_to(source)
            except Exception:
                shutil.copy2(str(source), str(dest))
            
            self.created_links.append(unique_name)
            logger.info(f"✅ Linked LoRA: {unique_name}")
            return unique_name
            
        except Exception as e:
            logger.error(f"Failed to link LoRA {source_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up all created LoRA links."""
        for filename in self.created_links:
            try:
                filepath = self.lora_dir / filename
                if filepath.exists() or filepath.is_symlink():
                    filepath.unlink()
                    logger.info(f"🧹 Removed LoRA link: {filename}")
            except Exception as e:
                logger.warning(f"Failed to remove LoRA link {filename}: {e}")
        
        self.created_links.clear()
    
    @staticmethod
    def resolve_lora_path(path: str) -> str:
        """Resolve S3/relative paths to absolute paths.
        
        Args:
            path: S3 URL or relative path
            
        Returns:
            Absolute path to the file
        """
        path = (path or "").strip()
        
        if path.startswith("s3://"):
            # Parse S3 URL
            remainder = path[len("s3://"):]
            if "/" in remainder:
                bucket, key = remainder.split("/", 1)
                
                # Map known prefixes to mount points
                if key.startswith("user-images/"):
                    return os.path.join(S3_MOUNT_PATH, key[len("user-images/"):])
                elif key.startswith("workflows/"):
                    return os.path.join(WORKFLOWS_MOUNT_PATH, key[len("workflows/"):])
                else:
                    # Default to data mount
                    return os.path.join(S3_MOUNT_PATH, key)
            return S3_MOUNT_PATH
            
        return path
    
    @staticmethod
    def create_safe_filename(source_path: str) -> str:
        """Create a safe filename from source path."""
        if source_path.startswith(S3_MOUNT_PATH + "/"):
            # Remove mount prefix and replace separators
            rel_path = source_path[len(S3_MOUNT_PATH + "/"):]
            return rel_path.replace("/", "_").replace("\\", "_")
        return os.path.basename(source_path)


def link_style_loras(lora_dir: str = LORA_MODELS_DIR) -> int:
    """Link all style LoRAs from S3 mount.
    
    Returns:
        Number of LoRAs linked
    """
    source_dir = os.path.join(S3_MOUNT_PATH, "style_loras")
    if not os.path.exists(source_dir):
        logger.info(f"Style LoRAs directory not found: {source_dir}")
        return 0
    
    logger.info(f"🔍 Scanning for style LoRAs in: {source_dir}")
    linked_count = 0
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.safetensors'):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                safe_filename = relative_path.replace('/', '_').replace('\\', '_')
                target_path = os.path.join(lora_dir, safe_filename)
                
                try:
                    # Remove existing link
                    if os.path.exists(target_path) or os.path.islink(target_path):
                        os.unlink(target_path)
                    
                    # Create symlink
                    os.symlink(source_path, target_path)
                    linked_count += 1
                    logger.info(f"🔗 Linked style LoRA: {safe_filename}")
                    
                except Exception as e:
                    logger.warning(f"Failed to link style LoRA {file}: {e}")
    
    return linked_count


def link_all_loras(lora_dir: str = LORA_MODELS_DIR) -> int:
    """Link all LoRAs from S3 bucket (development mode).
    
    Returns:
        Number of LoRAs linked
    """
    if not os.path.exists(S3_MOUNT_PATH):
        logger.warning(f"S3 bucket not mounted at {S3_MOUNT_PATH}")
        return 0
    
    logger.info(f"🔍 Scanning S3 bucket for LoRAs: {S3_MOUNT_PATH}")
    linked_count = 0
    
    for root, _, files in os.walk(S3_MOUNT_PATH):
        for file in files:
            if file.endswith('.safetensors'):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, S3_MOUNT_PATH)
                safe_filename = relative_path.replace('/', '_').replace('\\', '_')
                target_path = os.path.join(lora_dir, safe_filename)
                
                try:
                    # Remove existing link
                    if os.path.exists(target_path) or os.path.islink(target_path):
                        os.unlink(target_path)
                    
                    # Create symlink
                    os.symlink(source_path, target_path)
                    linked_count += 1
                    
                    if linked_count % 50 == 0:
                        logger.info(f"🔗 Linked {linked_count} LoRAs...")
                        
                except Exception as e:
                    logger.warning(f"Failed to link {file}: {e}")
    
    logger.info(f"✅ Linked {linked_count} LoRAs total")
    return linked_count
