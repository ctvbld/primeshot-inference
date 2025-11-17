"""Image processing utilities for inference."""

import os
import shutil
import base64
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import logging

from .constants import (
    SUPPORTED_IMAGE_EXTENSIONS,
    MAX_PREVIEW_SIZE,
    PREVIEW_THUMBNAIL_SIZE,
    PREVIEW_THUMBNAIL_QUALITY,
    PREVIEW_FALLBACK_SIZE,
    PREVIEW_FALLBACK_QUALITY,
    S3_UPLOAD_MAX_RETRIES
)

logger = logging.getLogger(__name__)


def categorize_images(generated_images: list[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """Categorize images by type (web/orig).
    
    Returns:
        Tuple of (web_image, orig_image)
    """
    web_img = None
    orig_img = None
    
    for img in generated_images:
        fname = str(img.get("filename", "")).lower()
        if fname.startswith("web_") or fname.endswith(".webp"):
            web_img = img
        elif fname.startswith("orig_") or fname.endswith(".png"):
            orig_img = img
    
    # Use first image as web if none found
    if web_img is None and generated_images:
        web_img = generated_images[0]
    
    return web_img, orig_img


def process_comfyui_image(
    img_info: dict,
    image_index: int,
    job_id: str,
    user_id: str,
    output_base_path: str = "/root/comfy/ComfyUI/output"
) -> Tuple[str, str, str]:
    """Process a ComfyUI generated image and determine its paths.
    
    Returns:
        Tuple of (full_path, variant_type, base_name)
    """
    filename = img_info.get("filename")
    subfolder = img_info.get("subfolder", "")
    
    if not filename:
        raise ValueError(f"Image {image_index + 1}: No filename provided")
    
    # Skip temporary preview files
    if "temp_" in filename.lower() or filename.startswith("ComfyUI_temp"):
        raise ValueError(f"Skipping temporary file: {filename}")
    
    # Construct full path
    if subfolder:
        image_path = os.path.join(output_base_path, subfolder, filename)
    else:
        image_path = os.path.join(output_base_path, filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_index + 1}: File not found at {image_path}")
    
    # Determine variant type
    variant = "unknown"
    lower_name = filename.lower()
    if lower_name.startswith("orig_"):
        variant = "orig"
    elif lower_name.startswith("web_"):
        variant = "web"
    
    # Use deterministic naming
    base_name = f"IMG-{image_index + 1:02d}"
    
    logger.info(f"🖼️ Processing ComfyUI image: {image_path}")
    logger.info(f"🔍 Variant: {variant}, base name: {base_name}")
    
    return image_path, variant, base_name


def save_image_variants(
    source_path: str,
    base_name: str,
    variant: str,
    user_id: str,
    job_id: str,
    data_path: str = "/data"
) -> Optional[str]:
    """Save image variants to S3-mounted paths.
    
    Returns:
        Final image URL if applicable, None otherwise
    """
    # Prepare target directories
    orig_dir = os.path.join(data_path, user_id, "inference", job_id, "orig")
    web_dir = os.path.join(data_path, user_id, "inference", job_id, "web")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(web_dir, exist_ok=True)
    
    original_ext = os.path.splitext(source_path)[1]
    final_image_url = None
    
    if variant == "orig":
        # Just copy original
        orig_filename = f"{base_name}{original_ext}"
        orig_path = os.path.join(orig_dir, orig_filename)
        shutil.copy(source_path, orig_path)
        logger.info(f"✅ Copied original to S3: {orig_path}")
        
    elif variant == "web":
        # Copy 2048px webp and create downscaled versions
        web_2048_filename = f"{base_name}.webp"
        web_2048_path = os.path.join(web_dir, web_2048_filename)
        shutil.copy(source_path, web_2048_path)
        logger.info(f"✅ Saved 2048px WebP: {web_2048_path}")
        
        # Create downscaled variants
        _create_web_variants(web_2048_path, base_name, web_dir)
        
        final_image_url = f"user-images/{user_id}/inference/{job_id}/web/{web_2048_filename}"
        
    else:
        # Legacy path: generate all variants
        # Copy original
        orig_filename = f"{base_name}{original_ext}"
        orig_path = os.path.join(orig_dir, orig_filename)
        shutil.copy(source_path, orig_path)
        logger.info(f"✅ Copied original to S3: {orig_path}")
        
        # Generate web variants
        _generate_all_web_variants(source_path, base_name, web_dir)
        
        final_image_url = f"user-images/{user_id}/inference/{job_id}/web/{base_name}.webp"
    
    return final_image_url


def _create_web_variants(source_webp_path: str, base_name: str, web_dir: str) -> None:
    """Create downscaled web variants from 2048px webp."""
    try:
        with Image.open(source_webp_path) as img2048:
            if img2048.mode in ('RGBA', 'LA', 'P'):
                img2048 = img2048.convert('RGB')
            
            for size in (720, 480):
                web_copy = img2048.copy()
                web_copy.thumbnail((size, size), Image.Resampling.LANCZOS)
                web_path = os.path.join(web_dir, f"{base_name}-w{size}.webp")
                
                _save_webp_with_retry(web_copy, web_path, size)
                
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate downscaled WebP variants: {e}")


def _generate_all_web_variants(source_path: str, base_name: str, web_dir: str) -> None:
    """Generate all web variants from source image."""
    try:
        with Image.open(source_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            for size in (2048, 720, 480):
                web_img = img.copy()
                web_img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                filename = f"{base_name}.webp" if size == 2048 else f"{base_name}-w{size}.webp"
                web_path = os.path.join(web_dir, filename)
                
                _save_webp_with_retry(web_img, web_path, size)
                
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate WebP variants: {e}")


def _save_webp_with_retry(
    img: Image.Image, 
    path: str, 
    size: int,
    max_retries: int = S3_UPLOAD_MAX_RETRIES
) -> bool:
    """Save WebP image with retry logic."""
    import time
    
    for attempt in range(max_retries):
        try:
            quality = max(65, 85 - attempt * 10)
            img.save(path, format='WEBP', quality=quality, method=6, optimize=True)
            
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.info(f"✅ Saved {size}px WebP: {path}")
                return True
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        time.sleep(0.1 * (attempt + 1))
    
    logger.error(f"❌ Failed to save {size}px WebP after {max_retries} attempts")
    return False


def optimize_preview_base64(
    base64_data: str,
    max_size: int = MAX_PREVIEW_SIZE
) -> Optional[str]:
    """Optimize base64 image for preview with aggressive compression."""
    if not base64_data or len(base64_data) < 100:
        return None
    
    try:
        import io
        
        # Handle data URL format
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        
        # Decode
        image_data = base64.b64decode(base64_data)
        if len(image_data) > 2 * 1024 * 1024:  # 2MB limit
            logger.warning(f"Preview image too large ({len(image_data)} bytes)")
            return None
        
        # Process image
        with Image.open(io.BytesIO(image_data)) as img:
            # Convert to RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize
            img.thumbnail(PREVIEW_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            
            # Save with compression
            buffer = io.BytesIO()
            img.save(
                buffer, 
                format='JPEG', 
                quality=PREVIEW_THUMBNAIL_QUALITY, 
                optimize=True
            )
            optimized_data = buffer.getvalue()
            
            # Check size and retry if needed
            if len(optimized_data) > max_size:
                buffer = io.BytesIO()
                img.thumbnail(PREVIEW_FALLBACK_SIZE, Image.Resampling.LANCZOS)
                img.save(
                    buffer,
                    format='JPEG',
                    quality=PREVIEW_FALLBACK_QUALITY,
                    optimize=True
                )
                optimized_data = buffer.getvalue()
            
            # Convert to base64
            optimized_b64 = base64.b64encode(optimized_data).decode('utf-8')
            return f"data:image/jpeg;base64,{optimized_b64}"
            
    except Exception as e:
        logger.error(f"Preview optimization failed: {e}")
        return None
