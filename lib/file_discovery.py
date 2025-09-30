"""File discovery and directory scanning utilities."""

import os
import time
import glob
from typing import List, Dict, Any, Optional
import logging

from .constants import (
    COMFYUI_OUTPUT_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    DIRECTORY_SCAN_MAX_ATTEMPTS,
    DIRECTORY_SCAN_DELAY,
    DIRECTORY_SCAN_BACKOFF,
    DIRECTORY_SCAN_MAX_DELAY
)

logger = logging.getLogger(__name__)


def find_generated_images(
    job_id: str,
    output_dir: str = COMFYUI_OUTPUT_DIR,
    max_attempts: int = DIRECTORY_SCAN_MAX_ATTEMPTS,
    delay_seconds: float = DIRECTORY_SCAN_DELAY
) -> List[Dict[str, Any]]:
    """Find generated images in ComfyUI output directory with retries.
    
    Args:
        job_id: Job ID to search for
        output_dir: Base output directory
        max_attempts: Maximum number of scan attempts
        delay_seconds: Initial delay between attempts
        
    Returns:
        List of image info dicts with filename, subfolder, type fields
    """
    found_images = []
    job_output_dir = os.path.join(output_dir, job_id)
    
    for attempt in range(max_attempts):
        # Check job-specific directory first
        found_images = _scan_job_directory(job_output_dir, job_id)
        if found_images:
            logger.info(f"✅ Found {len(found_images)} files in job directory after {attempt + 1} attempts")
            return found_images
        
        # Check parent directory for job-prefixed files
        found_images = _scan_parent_directory(output_dir, job_id)
        if found_images:
            logger.info(f"✅ Found {len(found_images)} files in parent directory after {attempt + 1} attempts")
            return found_images
        
        # Check for recent files with web_/orig_ prefixes
        found_images = _scan_recent_files(output_dir)
        if found_images:
            logger.info(f"✅ Found {len(found_images)} recent files after {attempt + 1} attempts")
            return found_images
        
        # Wait before next attempt
        if attempt < max_attempts - 1:
            logger.info(f"⏳ Directory scan attempt {attempt + 1}/{max_attempts}, waiting {delay_seconds}s...")
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * DIRECTORY_SCAN_BACKOFF, DIRECTORY_SCAN_MAX_DELAY)
    
    return found_images


def _scan_job_directory(job_dir: str, job_id: str) -> List[Dict[str, Any]]:
    """Scan job-specific directory for images."""
    if not os.path.exists(job_dir):
        return []
    
    found_images = []
    files = [
        f for f in os.listdir(job_dir)
        if f.endswith(SUPPORTED_IMAGE_EXTENSIONS)
    ]
    
    for filename in files:
        found_images.append({
            "filename": filename,
            "subfolder": job_id,
            "type": "output"
        })
    
    return found_images


def _scan_parent_directory(output_dir: str, job_id: str) -> List[Dict[str, Any]]:
    """Scan parent directory for job-prefixed files."""
    if not os.path.exists(output_dir):
        return []
    
    found_images = []
    parent_files = [
        f for f in os.listdir(output_dir)
        if f.startswith(job_id) and f.endswith(SUPPORTED_IMAGE_EXTENSIONS)
    ]
    
    for filename in parent_files:
        found_images.append({
            "filename": filename,
            "subfolder": "",
            "type": "output"
        })
    
    return found_images


def _scan_recent_files(output_dir: str, max_age_seconds: float = 30) -> List[Dict[str, Any]]:
    """Scan for recently created files with specific prefixes."""
    if not os.path.exists(output_dir):
        return []
    
    found_images = []
    current_time = time.time()
    
    # Get all image files
    pattern = os.path.join(output_dir, "*")
    all_files = glob.glob(pattern)
    
    image_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    
    # Sort by modification time (most recent first)
    image_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    
    # Check recent files only
    for filepath in image_files[:20]:  # Check only 20 most recent
        file_age = current_time - os.path.getmtime(filepath)
        
        if file_age < max_age_seconds:
            basename = os.path.basename(filepath)
            
            # Check for specific prefixes
            if basename.startswith(('web_', 'orig_')):
                found_images.append({
                    "filename": basename,
                    "subfolder": "",
                    "type": "output"
                })
    
    return found_images


def get_image_outputs(
    job_id: str,
    prompt_id: str,
    comfyui_base_url: str,
    timeout: int = 600
) -> tuple[dict, str]:
    """Get image outputs using multiple methods with fallback.
    
    Returns:
        Tuple of (outputs_dict, source) where source indicates the method used
    """
    from .ws_preview_relay import wait_for_prompt_completion, get_prompt_completion_data
    from .comfyui_server import ComfyUIServer
    
    start_time = time.time()
    server = ComfyUIServer()
    server.base_url = comfyui_base_url
    
    # Method 1: WebSocket completion data
    logger.info(f"⏳ Waiting for completion via WebSocket...")
    completed = wait_for_prompt_completion(job_id, prompt_id, timeout=timeout)
    
    if completed:
        elapsed = time.time() - start_time
        logger.info(f"✅ WebSocket reported completion (took {elapsed:.1f}s)")
        
        completion_data = get_prompt_completion_data(job_id, prompt_id)
        if completion_data and completion_data.get("outputs"):
            return completion_data["outputs"], "websocket"
    
    # Method 2: ComfyUI History API
    logger.info(f"⚠️ WebSocket data incomplete, trying history API...")
    history_start = time.time()
    
    result = server.poll_history(prompt_id)
    if result and result.get("outputs"):
        elapsed = time.time() - history_start
        logger.info(f"✅ Got outputs from history (took {elapsed:.1f}s)")
        return result["outputs"], "history"
    
    # Method 3: Directory scanning
    logger.info(f"⚠️ History unavailable, scanning output directory...")
    dir_start = time.time()
    
    found_images = find_generated_images(job_id, max_attempts=5)
    if found_images:
        elapsed = time.time() - dir_start
        logger.info(f"✅ Found images via directory scan (took {elapsed:.1f}s)")
        
        # Build outputs structure
        outputs = {
            "directory_scan": {"images": found_images}
        }
        return outputs, "directory"
    
    # All methods failed
    total_time = time.time() - start_time
    raise RuntimeError(f"Failed to get outputs after {total_time:.1f}s - all methods exhausted")


def extract_images_from_outputs(outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract image information from ComfyUI outputs structure."""
    generated_images = []
    
    for node_id, node_output in outputs.items():
        if isinstance(node_output, dict) and "images" in node_output:
            for img in node_output["images"]:
                if isinstance(img, dict):
                    generated_images.append({
                        "filename": img.get("filename"),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output")
                    })
    
    return generated_images
