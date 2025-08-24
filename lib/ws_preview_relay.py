from __future__ import annotations

import os
import time
import json as _json
import asyncio
import threading
import websockets
import gc
from typing import Optional, Dict, Any

class WebSocketRelay:
    """Simple WebSocket relay with efficient memory management."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.last_preview_ts = 0.0
        self.image_index = 0
        self.max_preview_size = 50 * 1024  # 50KB max per preview
        self.current_preview: Optional[str] = None
        
    def cleanup_memory(self):
        """Clean up cached preview data."""
        if self.current_preview:
            del self.current_preview
            self.current_preview = None
        gc.collect()
        
    def optimize_preview_base64(self, base64_data: str) -> Optional[str]:
        """Optimize base64 image with aggressive memory management."""
        if not base64_data or len(base64_data) < 100:
            return None
            
        try:
            import base64
            import io
            from PIL import Image
            
            # Clean up previous preview immediately
            self.cleanup_memory()
            
            # Handle data URL format
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',')[1]
            
            # Decode with memory limit check
            try:
                image_data = base64.b64decode(base64_data)
                if len(image_data) > 2 * 1024 * 1024:  # 2MB limit
                    print(f"⚠️ Preview image too large ({len(image_data)} bytes), skipping")
                    return None
            except Exception as decode_e:
                print(f"⚠️ Base64 decode failed: {decode_e}")
                return None
            
            # Process image with memory management
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    # Convert to RGB (more memory efficient than RGBA)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Aggressive resize for memory efficiency
                    img.thumbnail((320, 320), Image.Resampling.LANCZOS)
                    
                    # Save with high compression
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=60, optimize=True)
                    optimized_data = buffer.getvalue()
                    
                    # Check final size
                    if len(optimized_data) > self.max_preview_size:
                        # Try even more aggressive compression
                        buffer = io.BytesIO()
                        img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                        img.save(buffer, format='JPEG', quality=45, optimize=True)
                        optimized_data = buffer.getvalue()
                    
                    # Convert to base64 and cache
                    optimized_b64 = base64.b64encode(optimized_data).decode('utf-8')
                    result = f"data:image/jpeg;base64,{optimized_b64}"
                    
                    # Store current preview
                    self.current_preview = result
                    
                    print(f"📸 Optimized preview: {len(base64_data)} → {len(result)} chars")
                    return result
                    
            except Exception as img_e:
                print(f"⚠️ Image processing failed: {img_e}")
                return None
                
        except Exception as e:
            print(f"⚠️ Preview optimization failed: {e}")
            return None
        finally:
            # Force cleanup
            if 'image_data' in locals():
                del image_data
            if 'optimized_data' in locals():
                del optimized_data
            gc.collect()


def start_relay(progress_ws_url: str, comfy_ws_url: str, job_id: str, throttle_sec: float = 1.5) -> None:
    """Start a memory-efficient background relay from ComfyUI WS to our progress WS broadcast endpoint."""
    
    # Keep a strong reference to prevent garbage collection during relay operation
    relay_manager = WebSocketRelay(job_id)
    
    # Store relay manager in a global dict to prevent garbage collection
    if not hasattr(start_relay, '_active_relays'):
        start_relay._active_relays = {}
    start_relay._active_relays[job_id] = relay_manager
    
    # Add a completion flag
    if not hasattr(start_relay, '_completion_flags'):
        start_relay._completion_flags = {}
    start_relay._completion_flags[job_id] = False

    async def _run():
        manager = start_relay._active_relays.get(job_id)
        if not manager:
            print(f"⚠️ Relay manager not found for job {job_id}")
            return
            
        print(f"🔌 Starting WebSocket relay for job {job_id}")
        print(f"  📥 ComfyUI WebSocket: {comfy_ws_url}")
        print(f"  📤 Progress broadcast: {progress_ws_url}")
            
        try:
            # Configure WebSocket with memory limits
            max_size = 2 * 1024 * 1024  # 2MB max message size
            
            # Connect to ComfyUI WebSocket (to receive progress)
            print(f"🔌 Connecting to ComfyUI WebSocket: {comfy_ws_url}")
            comfy_ws = await websockets.connect(
                comfy_ws_url, 
                ping_interval=30,  # Keep connection alive
                max_size=max_size,
                compression=None  # Disable compression to save memory
            )
            print(f"✅ Connected to ComfyUI WebSocket for job {job_id}")
            
            # Connect to progress broadcast WebSocket (to send progress) with retry logic
            print(f"🔌 Connecting to progress broadcast: {progress_ws_url}")
            
            # First, try to check if the progress server is healthy (try both endpoints like training app)
            try:
                import urllib.request
                import urllib.error
                base_url = progress_ws_url.replace("wss://", "https://").replace("ws://", "http://").split("/ws/")[0]
                
                # Try /api/health first (training app pattern), then /health as fallback
                for health_path in ["/api/health", "/health"]:
                    health_url = base_url + health_path
                    try:
                        print(f"🏥 Checking progress server health: {health_url}")
                        with urllib.request.urlopen(health_url, timeout=10) as response:
                            if response.status == 200:
                                print(f"✅ Progress server is healthy")
                                break
                            else:
                                print(f"⚠️ Health check {health_path} returned status {response.status}")
                    except Exception as endpoint_e:
                        print(f"⚠️ Health check {health_path} failed: {endpoint_e}")
                        continue
                else:
                    print(f"⚠️ All health check endpoints failed")
                    
            except Exception as health_e:
                print(f"⚠️ Progress server health check setup failed: {health_e}")
                
            print(f"🔄 Proceeding with WebSocket connection attempt...")
            
            broadcast_ws = None
            max_retries = 3
            retry_delay = 2  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Connection attempt {attempt + 1}/{max_retries} to progress broadcast")
                    broadcast_ws = await websockets.connect(
                        progress_ws_url, 
                        ping_interval=30,  # Keep connection alive (same as training)
                        ping_timeout=20,   # Match training app timeout
                        close_timeout=10,  # Match training app timeout
                        max_size=max_size,
                        compression=None,
                        open_timeout=30    # Match training app timeout (30 seconds)
                    )
                    print(f"✅ Connected to progress broadcast for job {job_id}")
                    break
                except Exception as conn_e:
                    print(f"⚠️ Connection attempt {attempt + 1} failed: {conn_e}")
                    if attempt < max_retries - 1:
                        print(f"⏳ Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"❌ Failed to connect to progress broadcast after {max_retries} attempts")
                        print(f"🔄 Continuing without progress broadcast - inference will still work")
                        broadcast_ws = None
            
            # Send initial connection message (only if broadcast_ws is connected)
            if broadcast_ws:
                try:
                    initial_msg = {
                        "job_id": job_id,
                        "job_type": "inference",
                        "status": "initializing",
                        "progress": 0,
                        "message": "WebSocket relay connected",
                        "timestamp": int(time.time() * 1000)
                    }
                    await broadcast_ws.send(_json.dumps(initial_msg))
                    print(f"📤 Sent initial connection message for job {job_id}")
                except Exception as init_e:
                    print(f"⚠️ Failed to send initial message for job {job_id}: {init_e}")
            else:
                print(f"⚠️ No progress broadcast connection - skipping initial message for job {job_id}")
            
            try:
                message_count = 0
                start_time = time.time()
                max_runtime = 600  # 10 minutes maximum runtime
                
                while True:
                    # Check if manager still exists
                    manager = start_relay._active_relays.get(job_id)
                    if not manager:
                        print(f"⚠️ Relay manager cleaned up, stopping relay for job {job_id}")
                        break
                    
                    # Check for timeout to prevent infinite loops
                    if time.time() - start_time > max_runtime:
                        print(f"⏰ WebSocket relay timeout after {max_runtime}s for job {job_id}")
                        # Send timeout completion message
                        if broadcast_ws:
                            try:
                                timeout_msg = {
                                    "job_id": job_id,
                                    "job_type": "inference",
                                    "status": "closed",
                                    "final": True,
                                    "close_connection": True,
                                    "timeout": True
                                }
                                await broadcast_ws.send(_json.dumps(timeout_msg))
                                print(f"📤 Sent timeout close signal for job {job_id}")
                            except Exception as timeout_e:
                                print(f"⚠️ Failed to send timeout message: {timeout_e}")
                        break
                    
                    # Check if job completion was signaled externally
                    if hasattr(start_relay, '_completion_flags') and start_relay._completion_flags.get(job_id, False):
                        print(f"🏁 External completion signal received for job {job_id}")
                        # Send completion message and exit
                        try:
                            completion_msg = {
                                "job_id": job_id,
                                "job_type": "inference",
                                "status": "completed",
                                "progress": 100,
                                "message": "Generation completed",
                                "timestamp": int(time.time() * 1000),
                                "final": True  # Signal that this is the final message
                            }
                            await broadcast_ws.send(_json.dumps(completion_msg))
                            print(f"📤 Sent external completion message for job {job_id}")
                            
                            # Send a close signal after a brief delay
                            await asyncio.sleep(0.1)
                            close_msg = {
                                "job_id": job_id,
                                "job_type": "inference",
                                "status": "closed",
                                "message": "Connection closing",
                                "timestamp": int(time.time() * 1000),
                                "close_connection": True
                            }
                            await broadcast_ws.send(_json.dumps(close_msg))
                            print(f"📤 Sent close signal for job {job_id}")
                            
                        except Exception as ext_complete_e:
                            print(f"⚠️ Failed to send external completion message for job {job_id}: {ext_complete_e}")
                        break
                    
                    try:
                        # Receive message from ComfyUI with timeout to check completion signals
                        try:
                            msg = await asyncio.wait_for(comfy_ws.recv(), timeout=1.0)
                            message_count += 1
                        except asyncio.TimeoutError:
                            # No message received, continue to check completion signal
                            continue
                        
                        # Periodic memory cleanup
                        if message_count % 50 == 0:
                            manager.cleanup_memory()
                            print(f"🧹 Memory cleanup for job {job_id} (message #{message_count})")
                        
                        try:
                            # Handle both text (JSON) and binary (preview image) messages
                            if isinstance(msg, bytes):
                                # Binary message - likely a preview image from ComfyUI
                                print(f"📸 Received binary preview data for job {job_id} (size: {len(msg)} bytes)")
                                
                                # ComfyUI sends binary data with 8-byte header format
                                try:
                                    image_data = msg
                                    
                                    # Check if this is ComfyUI's 8-byte header format
                                    if len(msg) >= 8:
                                        # ComfyUI sends preview images as binary WS messages with 8-byte header
                                        # The header indicates message type + format, followed by actual image data
                                        header = msg[:8]
                                        potential_image_data = msg[8:]
                                        
                                        print(f"🔍 Header: {header.hex()}, remaining data: {len(potential_image_data)} bytes")
                                        
                                        # Check if the data after header looks like a valid image
                                        if len(potential_image_data) > 0:
                                            is_png_after_header = potential_image_data.startswith(b'\x89PNG\r\n\x1a\n')
                                            is_jpeg_after_header = potential_image_data.startswith(b'\xff\xd8\xff')
                                            is_webp_after_header = potential_image_data.startswith(b'RIFF') and b'WEBP' in potential_image_data[:12]
                                            
                                            if is_png_after_header or is_jpeg_after_header or is_webp_after_header:
                                                print(f"✅ Found valid image data after 8-byte header for job {job_id}")
                                                image_data = potential_image_data
                                            else:
                                                # Also check if the full message (without header parsing) is a valid image
                                                is_png_full = msg.startswith(b'\x89PNG\r\n\x1a\n')
                                                is_jpeg_full = msg.startswith(b'\xff\xd8\xff')
                                                is_webp_full = msg.startswith(b'RIFF') and b'WEBP' in msg[:12]
                                                
                                                if is_png_full or is_jpeg_full or is_webp_full:
                                                    print(f"✅ Found valid image data in full message for job {job_id}")
                                                    image_data = msg
                                                else:
                                                    print(f"🔍 No valid image format found. Header: {header.hex()}")
                                                    print(f"🔍 First 16 bytes after header: {potential_image_data[:16].hex() if len(potential_image_data) >= 16 else potential_image_data.hex()}")
                                                    continue
                                    
                                    # Check if we have valid image data to process
                                    is_png = image_data.startswith(b'\x89PNG\r\n\x1a\n')
                                    is_jpeg = image_data.startswith(b'\xff\xd8\xff')
                                    is_webp = image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]
                                    
                                    if is_png or is_jpeg or is_webp:
                                        # Process the valid image data
                                        import base64
                                        preview_base64 = base64.b64encode(image_data).decode('utf-8')
                                        
                                        # Send preview to frontend
                                        if broadcast_ws:
                                            try:
                                                now = time.time()
                                                if now - manager.last_preview_ts > throttle_sec:
                                                    optimized_preview = manager.optimize_preview_base64(preview_base64)
                                                    
                                                    if optimized_preview:
                                                        preview_evt = {
                                                            "job_id": job_id,
                                                            "job_type": "inference",
                                                            "status": "running",
                                                            "preview_images": [optimized_preview],
                                                            "preview_index": manager.image_index,
                                                            "timestamp": int(time.time() * 1000)
                                                        }
                                                        await broadcast_ws.send(_json.dumps(preview_evt))
                                                        manager.last_preview_ts = now
                                                        manager.image_index += 1
                                                        print(f"📸 Sent ComfyUI binary preview for job {job_id} (original: {len(image_data)} bytes, optimized: {len(optimized_preview)} chars)")
                                            except Exception as preview_e:
                                                print(f"⚠️ Failed to send binary preview for job {job_id}: {preview_e}")
                                    else:
                                        # Still couldn't find valid image data
                                        print(f"🔍 Binary data doesn't contain valid image format after header parsing")
                                        print(f"🔍 First 16 bytes of processed data: {image_data[:16].hex() if len(image_data) >= 16 else image_data.hex()}")
                                        
                                except Exception as binary_e:
                                    print(f"⚠️ Failed to process binary preview data: {binary_e}")
                                
                                continue
                            
                            # Text message - parse as JSON
                            data = _json.loads(msg)
                            
                            # Debug: Log message types to understand ComfyUI's format
                            if message_count <= 5 or message_count % 25 == 0:
                                msg_type = data.get("type", "unknown") if isinstance(data, dict) else "non-dict"
                                print(f"🔍 ComfyUI message #{message_count} for job {job_id}: type='{msg_type}', keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                                
                        except Exception as parse_e:
                            print(f"⚠️ Failed to parse ComfyUI message for job {job_id}: {parse_e}")
                            continue
                            
                        # Create progress event in the format expected by frontend
                        evt = {
                            "job_id": job_id,
                            "job_type": "inference",
                            "timestamp": int(time.time() * 1000),  # Milliseconds
                            "message": "Processing..."
                        }
                        
                        if isinstance(data, dict):
                            # Handle ComfyUI progress updates (most accurate)
                            if "progress" in data:
                                progress_value = data.get("progress", 0)
                                evt["progress"] = min(100, max(0, progress_value))
                                
                                # Map progress to status
                                if progress_value >= 100:
                                    evt["status"] = "completed"
                                    evt["message"] = "Generation completed"
                                elif progress_value > 0:
                                    evt["status"] = "running"
                                    evt["message"] = f"Generating... {progress_value:.1f}%"
                                else:
                                    evt["status"] = "initializing"
                                    evt["message"] = "Initializing generation..."
                                
                                # Check for preview data in progress messages (ComfyUI often sends previews here)
                                if "preview" in data:
                                    preview = data.get("preview")
                                    print(f"🔍 Found preview in progress message for job {job_id}")
                                elif "image" in data:
                                    preview = data.get("image")
                                    print(f"🔍 Found image in progress message for job {job_id}")
                            
                            # Handle ComfyUI execution updates with progressive progress
                            elif data.get("type") == "execution_start":
                                evt["status"] = "running"
                                evt["progress"] = 5
                                evt["message"] = "Starting generation..."
                                manager.execution_progress = 5  # Track base progress
                                
                            elif data.get("type") == "executing":
                                node_id = data.get("data", {}).get("node")
                                evt["status"] = "running"
                                
                                # Progressive progress based on execution count
                                if not hasattr(manager, 'execution_progress'):
                                    manager.execution_progress = 5
                                
                                # Increment progress gradually for each executing node
                                manager.execution_progress = min(85, manager.execution_progress + 3)
                                evt["progress"] = manager.execution_progress
                                evt["message"] = f"Processing node {node_id}..."
                                
                            elif data.get("type") == "execution_cached":
                                evt["status"] = "running"
                                if not hasattr(manager, 'execution_progress'):
                                    manager.execution_progress = 50
                                else:
                                    manager.execution_progress = min(85, manager.execution_progress + 5)
                                evt["progress"] = manager.execution_progress
                                evt["message"] = "Using cached results..."
                                
                            elif data.get("type") == "executed":
                                # Handle executed messages (these often contain preview images)
                                evt["status"] = "running"
                                if not hasattr(manager, 'execution_progress'):
                                    manager.execution_progress = 60
                                else:
                                    manager.execution_progress = min(95, manager.execution_progress + 5)
                                evt["progress"] = manager.execution_progress
                                evt["message"] = "Processing completed nodes..."
                                
                            elif data.get("type") == "execution_complete":
                                evt["status"] = "completed"
                                evt["progress"] = 100
                                evt["message"] = "Generation completed"
                                
                                # Mark job as completed and break out of loop
                                print(f"🏁 ComfyUI execution_complete received for job {job_id}")
                                
                                # Send the completion message
                                if broadcast_ws:
                                    try:
                                        completion_message = _json.dumps(evt)
                                        await broadcast_ws.send(completion_message)
                                        print(f"📤 Sent completion message for job {job_id}")
                                        
                                        # Send a final close signal
                                        await asyncio.sleep(0.1)
                                        close_msg = {
                                            "job_id": job_id,
                                            "job_type": "inference", 
                                            "status": "closed",
                                            "final": True,
                                            "close_connection": True
                                        }
                                        await broadcast_ws.send(_json.dumps(close_msg))
                                        print(f"📤 Sent close signal for job {job_id}")
                                        
                                    except Exception as completion_e:
                                        print(f"⚠️ Failed to send completion message: {completion_e}")
                                
                                # Break out of the message loop
                                print(f"🏁 Breaking out of WebSocket loop for completed job {job_id}")
                                break
                            
                            # Handle preview images with memory management
                            preview = None
                            
                            # Check for different preview formats from ComfyUI
                            # ComfyUI sends previews in several different formats:
                            
                            # 1. Direct preview message type
                            if data.get("type") == "preview":
                                preview = data.get("data", {}).get("image")
                                print(f"🔍 Found preview in 'preview' type message for job {job_id}")
                            
                            # 2. Progress messages with preview data (most common for sampling previews)
                            elif data.get("type") == "progress":
                                progress_data = data.get("data", {})
                                
                                # Log progress message structure for debugging
                                if message_count <= 10 or "preview" in str(progress_data).lower():
                                    progress_keys = list(progress_data.keys()) if isinstance(progress_data, dict) else []
                                    print(f"🔍 Progress message keys: {progress_keys}")
                                    
                                    # Log detailed structure if it might contain preview
                                    if any(key in str(progress_data).lower() for key in ["preview", "image", "base64"]):
                                        print(f"🔍 Progress data structure: {type(progress_data)} with keys: {progress_keys}")
                                        for k, v in progress_data.items() if isinstance(progress_data, dict) else []:
                                            if isinstance(v, str) and len(v) > 100:
                                                print(f"🔍   {k}: <string length {len(v)}>")
                                            elif isinstance(v, dict):
                                                print(f"🔍   {k}: dict with keys {list(v.keys())}")
                                            else:
                                                print(f"🔍   {k}: {type(v)} = {v}")
                                
                                # Check for preview in progress data
                                if "preview" in progress_data:
                                    preview_data = progress_data["preview"]
                                    if isinstance(preview_data, str):
                                        preview = preview_data
                                        print(f"🔍 Found preview string in progress message for job {job_id}")
                                    elif isinstance(preview_data, dict):
                                        # Preview might be nested in a dict
                                        if "image" in preview_data:
                                            preview = preview_data["image"]
                                            print(f"🔍 Found preview.image in progress message for job {job_id}")
                                        elif "data" in preview_data:
                                            preview = preview_data["data"]
                                            print(f"🔍 Found preview.data in progress message for job {job_id}")
                                
                                # Check for direct image in progress data
                                elif "image" in progress_data:
                                    preview = progress_data["image"]
                                    print(f"🔍 Found image in progress message data for job {job_id}")
                                
                                # Check for base64 encoded image data
                                elif "base64" in progress_data:
                                    preview = progress_data["base64"]
                                    print(f"🔍 Found base64 in progress message data for job {job_id}")
                            
                            # 3. Progress messages with node-specific data (KSampler, etc.)
                            elif data.get("type") == "progress" and "data" in data:
                                progress_data = data.get("data", {})
                                # Sometimes preview is in node-specific data
                                if "node" in progress_data:
                                    node_data = progress_data.get("node", {})
                                    if isinstance(node_data, dict) and "preview" in node_data:
                                        preview = node_data["preview"]
                                        print(f"🔍 Found preview in progress node data for job {job_id}")
                            
                            # 4. Direct top-level image data
                            elif "image" in data:
                                # Direct image data
                                preview = data.get("image")
                                print(f"🔍 Found preview in 'image' field for job {job_id}")
                            elif "preview" in data:
                                # Preview field
                                preview = data.get("preview")
                                print(f"🔍 Found preview in 'preview' field for job {job_id}")
                            elif data.get("type") == "executing" and "data" in data:
                                # Sometimes previews come with executing messages
                                exec_data = data.get("data", {})
                                if "preview" in exec_data:
                                    preview = exec_data.get("preview")
                                    print(f"🔍 Found preview in executing message for job {job_id}")
                                elif "image" in exec_data:
                                    preview = exec_data.get("image")
                                    print(f"🔍 Found image in executing message for job {job_id}")
                            elif data.get("type") == "executed" and "data" in data:
                                # Check for preview images in executed messages
                                exec_data = data.get("data", {})
                                if "output" in exec_data and isinstance(exec_data["output"], dict):
                                    # Look for images in the output
                                    for output_key, output_value in exec_data["output"].items():
                                        if isinstance(output_value, dict) and "images" in output_value:
                                            images = output_value["images"]
                                            if isinstance(images, list) and len(images) > 0:
                                                # Take the first image as preview
                                                first_image = images[0]
                                                if isinstance(first_image, dict):
                                                    # Check for base64 data or filename
                                                    if "data" in first_image:
                                                        preview = first_image["data"]
                                                        print(f"🔍 Found preview image data in executed message for job {job_id}")
                                                    elif "filename" in first_image:
                                                        print(f"🔍 Found preview image filename in executed message: {first_image['filename']}")
                                                elif isinstance(first_image, str):
                                                    preview = first_image
                                                    print(f"🔍 Found preview image string in executed message for job {job_id}")
                                                break
                            
                            # Basic debug logging for message types
                            msg_type = data.get("type", "unknown")
                            
                            # Log key message types only
                            if msg_type in ["execution_start", "execution_complete", "progress"] or message_count <= 5:
                                print(f"🔍 Message #{message_count}: type='{msg_type}'")

                            
                            if isinstance(preview, str) and len(preview) > 100:
                                now = time.time()
                                if now - manager.last_preview_ts > throttle_sec:
                                    optimized_preview = manager.optimize_preview_base64(preview)
                                    
                                    if optimized_preview:
                                        evt["preview_images"] = [optimized_preview]
                                        evt["preview_index"] = manager.image_index
                                        manager.last_preview_ts = now
                                        
                                        print(f"📸 Sending preview for job {job_id} (size: {len(optimized_preview)} chars)")
                                    
                                    # Clean up original preview data immediately
                                    del preview
                                    if 'data' in locals() and 'image' in data:
                                        del data['image']
                                    if 'data' in locals() and 'preview' in data:
                                        del data['preview']
                        
                        # Send update if we have meaningful data
                        if any(key in evt for key in ["progress", "preview_images", "status"]):
                            try:
                                message_json = _json.dumps(evt)
                                
                                # Check message size before sending
                                if len(message_json) > 1024 * 1024:  # 1MB limit
                                    # Remove preview and send progress only
                                    fallback_evt = {k: v for k, v in evt.items() if k != 'preview_images'}
                                    fallback_evt['preview_fallback'] = True
                                    message_json = _json.dumps(fallback_evt)
                                    print(f"⚠️ Large message fallback for job {job_id}")
                                
                                # Send to broadcast WebSocket (only if connected)
                                if broadcast_ws:
                                    await broadcast_ws.send(message_json)
                                    
                                    # Log different types of updates
                                    if "preview_images" in evt:
                                        print(f"📸 Sent preview update for job {job_id}")
                                    elif "progress" in evt:
                                        print(f"📊 Sent progress update for job {job_id}: {evt.get('status', 'unknown')} - {evt.get('progress', 0)}%")
                                    else:
                                        print(f"📤 Sent status update for job {job_id}: {evt.get('status', 'unknown')}")
                                else:
                                    # Log that we're skipping the broadcast
                                    if "progress" in evt:
                                        print(f"📊 Progress update for job {job_id}: {evt.get('status', 'unknown')} - {evt.get('progress', 0)}% (no broadcast)")
                                
                                # Clean up message data
                                del message_json
                                
                            except websockets.exceptions.ConnectionClosed:
                                print(f"🔌 Broadcast WebSocket connection closed for job {job_id}, attempting reconnect...")
                                # Try to reconnect broadcast WebSocket
                                try:
                                    broadcast_ws = await websockets.connect(
                                        progress_ws_url, 
                                        ping_interval=30,  # Keep connection alive (same as training)
                                        ping_timeout=20,   # Match training app timeout
                                        close_timeout=10,  # Match training app timeout
                                        max_size=max_size,
                                        compression=None,
                                        open_timeout=30    # Match training app timeout (30 seconds)
                                    )
                                    print(f"🔄 Reconnected broadcast WebSocket for job {job_id}")
                                    # Retry sending the message
                                    await broadcast_ws.send(message_json)
                                    print(f"📤 Resent message after reconnect for job {job_id}")
                                except Exception as reconnect_e:
                                    print(f"❌ Failed to reconnect broadcast WebSocket for job {job_id}: {reconnect_e}")
                                    broadcast_ws = None
                                    # Continue without broadcast connection
                            except Exception as send_e:
                                print(f"⚠️ Failed to send WebSocket message for job {job_id}: {send_e}")
                                # For other errors, just log and continue
                                continue
                        
                        # Clean up event data
                        del evt
                        
                    except websockets.exceptions.ConnectionClosed:
                        print(f"🔌 ComfyUI WebSocket connection closed for job {job_id}")
                        # Send final completion message if we haven't already
                        if not start_relay._completion_flags.get(job_id, False):
                            try:
                                final_msg = {
                                    "job_id": job_id,
                                    "job_type": "inference",
                                    "status": "completed",
                                    "progress": 100,
                                    "message": "Generation completed (connection closed)",
                                    "timestamp": int(time.time() * 1000)
                                }
                                await broadcast_ws.send(_json.dumps(final_msg))
                                print(f"📤 Sent final message on connection close for job {job_id}")
                            except Exception as final_close_e:
                                print(f"⚠️ Failed to send final message on close for job {job_id}: {final_close_e}")
                        break
                    except Exception as msg_e:
                        print(f"⚠️ Message processing error for job {job_id}: {msg_e}")
                        continue
                        
            finally:
                # Send final completion message before closing (only if broadcast_ws is connected)
                if broadcast_ws:
                    try:
                        final_msg = {
                            "job_id": job_id,
                            "job_type": "inference",
                            "status": "completed",
                            "progress": 100,
                            "message": "Generation completed",
                            "timestamp": int(time.time() * 1000)
                        }
                        await broadcast_ws.send(_json.dumps(final_msg))
                        print(f"📤 Sent final completion message for job {job_id}")
                    except Exception as final_e:
                        print(f"⚠️ Failed to send final completion message for job {job_id}: {final_e}")
                else:
                    print(f"📤 Job {job_id} completed (no broadcast connection)")
                
                # Close WebSocket connections
                try:
                    await comfy_ws.close()
                    print(f"🔌 Closed ComfyUI WebSocket for job {job_id}")
                except:
                    pass
                    
                if broadcast_ws:
                    try:
                        await broadcast_ws.close()
                        print(f"🔌 Closed broadcast WebSocket for job {job_id}")
                    except:
                        pass
                    
        except Exception as e:
            print(f"⚠️ WS relay ended for job {job_id}: {e}")
            import traceback
            print(f"📊 Full error traceback: {traceback.format_exc()}")
        finally:
            # Final cleanup
            manager = start_relay._active_relays.pop(job_id, None)
            start_relay._completion_flags.pop(job_id, None)
            if manager:
                manager.cleanup_memory()
                print(f"🧹 Final cleanup for job {job_id}")
            else:
                print(f"🧹 No cleanup needed for job {job_id} (already cleaned)")

    def run_with_cleanup():
        """Run the relay with proper cleanup."""
        try:
            asyncio.run(_run())
        except Exception as e:
            print(f"⚠️ Relay thread error for job {job_id}: {e}")
        finally:
            # Ensure cleanup even if thread crashes
            manager = start_relay._active_relays.pop(job_id, None)
            if manager:
                manager.cleanup_memory()
                print(f"🧹 Thread cleanup for job {job_id}")
            else:
                print(f"🧹 No thread cleanup needed for job {job_id}")

    threading.Thread(target=run_with_cleanup, daemon=True).start()
    print(f"🚀 Started memory-managed WebSocket relay for job {job_id}")
    print(f"📊 Active relays: {list(start_relay._active_relays.keys())}")


def signal_job_completion(job_id: str) -> None:
    """Signal that a job has completed externally (from main process)."""
    if hasattr(start_relay, '_completion_flags') and job_id in start_relay._completion_flags:
        start_relay._completion_flags[job_id] = True
        print(f"🏁 Signaled completion for job {job_id}")
        
        # Give the relay a moment to send the completion message and close
        import time
        time.sleep(0.5)
        
        # Force cleanup if relay is still active after a short delay
        if hasattr(start_relay, '_active_relays') and job_id in start_relay._active_relays:
            print(f"🧹 Force cleaning up relay for completed job {job_id}")
            manager = start_relay._active_relays.pop(job_id, None)
            start_relay._completion_flags.pop(job_id, None)
            if manager:
                manager.cleanup_memory()
                print(f"✅ Force cleanup completed for job {job_id}")
    else:
        print(f"⚠️ No active relay found to signal completion for job {job_id}")


def cleanup_all_relays() -> None:
    """Clean up all active relays (useful for container shutdown)."""
    if hasattr(start_relay, '_active_relays'):
        active_jobs = list(start_relay._active_relays.keys())
        print(f"🧹 Cleaning up {len(active_jobs)} active relays: {active_jobs}")
        
        for job_id in active_jobs:
            manager = start_relay._active_relays.pop(job_id, None)
            start_relay._completion_flags.pop(job_id, None) if hasattr(start_relay, '_completion_flags') else None
            if manager:
                manager.cleanup_memory()
        
        print(f"✅ Cleaned up all active relays")


def get_active_relays() -> list:
    """Get list of currently active relay job IDs."""
    if hasattr(start_relay, '_active_relays'):
        return list(start_relay._active_relays.keys())
    return []


def start_direct_comfyui_relay(progress_ws_url: str, comfy_ws_url: str, job_id: str, throttle_sec: float = 1.5) -> None:
    """Alias for start_relay - connects directly to ComfyUI WebSocket."""
    start_relay(progress_ws_url, comfy_ws_url, job_id, throttle_sec)


