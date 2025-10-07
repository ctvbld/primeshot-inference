from __future__ import annotations

import os
import time
import json as _json
import asyncio
import threading
import websockets
import gc
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

class WebSocketRelay:
    """Simple WebSocket relay with efficient memory management and status state machine."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.last_preview_ts = 0.0
        self.image_index = 0
        self.max_preview_size = 50 * 1024  # 50KB max per preview
        self.current_preview: Optional[str] = None
        self.generation_started = False
        # Event-based completion signaling
        import threading
        self.completion_event = threading.Event()
        self.completed_prompt_ids = set()  # Track completed prompt IDs
        self.completion_data = {}  # Store full completion data by prompt_id
        self.last_activity_time = time.time()  # Track last activity for timeout detection
        self.last_progress_state = {}  # Track last seen progress state for fallback completion
        self.last_prompt_id: Optional[str] = None  # Track most recent prompt_id for graceful close
        # Error tracking
        self.last_error: Optional[dict] = None
        self.errors_by_prompt: dict[str, dict] = {}
        # Executing node tracking
        self.last_executing_node: Optional[dict] = None
        
    def update_image_index(self, new_index: int):
        """Update the current image index being processed."""
        self.image_index = new_index

        
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
    
    def send_message(self, message_data: dict):
        """Send a custom message through the WebSocket relay."""
        # Simple rule: if already generating, only allow "generating", "image_completed", "completed", and "failed" statuses
        allowed_statuses = ["generating", "image_completed", "completed", "failed"]
        if self.generation_started and message_data.get('status') not in allowed_statuses:
            print(f"🚫 Ignored status '{message_data.get('status')}' - already generating for job {self.job_id}")
            return
            
        # Queue the message to be sent by the relay loop
        if not hasattr(self, '_custom_messages'):
            self._custom_messages = []
        self._custom_messages.append(message_data)


def start_relay(progress_ws_url: str, comfy_ws_url: str, job_id: str, throttle_sec: float = 1.5, image_index: int = 0) -> None:
    """Start a memory-efficient background relay from ComfyUI WS to our progress WS broadcast endpoint."""
    
    # Keep a strong reference to prevent garbage collection during relay operation
    relay_manager = WebSocketRelay(job_id)
    relay_manager.image_index = image_index  # Set the image index for this relay
    
    # Store relay manager in a global dict to prevent garbage collection
    if not hasattr(start_relay, '_active_relays'):
        start_relay._active_relays = {}
    # Ensure single active relay per job: replace any previous and let it clean up
    prev = start_relay._active_relays.get(job_id)
    start_relay._active_relays[job_id] = relay_manager
    
    # Check for stored generating status and apply it to the new relay
    if hasattr(send_global_job_status, '_global_statuses') and job_id in send_global_job_status._global_statuses:
        stored_status = send_global_job_status._global_statuses[job_id]
        # Apply generating flag if it's "generating"
        if stored_status["status"] == "generating":
            relay_manager.generation_started = True

    
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
            
        try:
            # Configure WebSocket with memory limits
            max_size = 2 * 1024 * 1024  # 2MB max message size
            
            # Connect to ComfyUI WebSocket (to receive progress)
            print(f"🔌 Connecting to ComfyUI WebSocket: {comfy_ws_url}")
            comfy_ws = await websockets.connect(
                comfy_ws_url,
                ping_interval=30,   # Keep connection alive
                ping_timeout=20,    # Avoid hanging connections
                close_timeout=10,   # Faster closes
                open_timeout=30,    # Bound connect attempts
                max_size=max_size,
                compression=None    # Disable compression to save memory
            )
            print(f"✅ Connected to ComfyUI WebSocket for job {job_id}")
            
            # Connect to progress broadcast WebSocket (to send progress) with retry logic
            print(f"🔌 Connecting to progress broadcast: {progress_ws_url}")
            
            # First, try to check if the progress server is healthy (try both endpoints like training app)
            try:
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
                    await broadcast_ws.send(_json.dumps({
                        "job_id": job_id,
                        "job_type": "inference",
                        "status": "starting",
                        "progress": 0,
                        "message": "Starting up",
                        "timestamp": int(time.time() * 1000)
                    }))
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
                                await broadcast_ws.send(_json.dumps({
                                    "job_id": job_id,
                                    "job_type": "inference",
                                    "status": "closed",
                                    "final": True,
                                    "close_connection": True,
                                    "timeout": True
                                }))
                                print(f"📤 Sent timeout close signal for job {job_id}")
                            except Exception as timeout_e:
                                print(f"⚠️ Failed to send timeout message: {timeout_e}")
                        break
                    
                    # Check for and send any custom messages
                    if hasattr(manager, '_custom_messages') and manager._custom_messages:
                        custom_messages = manager._custom_messages.copy()
                        manager._custom_messages.clear()  # Clear the queue
                        
                        for custom_msg in custom_messages:
                            if broadcast_ws:
                                try:
                                    await broadcast_ws.send(_json.dumps(custom_msg))
                                    print(f"📤 Sent custom message for job {job_id}: {custom_msg.get('status', 'unknown')}")
                                except Exception as custom_e:
                                    print(f"⚠️ Failed to send custom message for job {job_id}: {custom_e}")
                    
                    # Check if job completion was signaled externally
                    if hasattr(start_relay, '_completion_flags') and start_relay._completion_flags.get(job_id, False):
                        print(f"🏁 External completion signal received for job {job_id}")
                        # Send completion message and exit
                        try:
                            await broadcast_ws.send(_json.dumps({
                                "job_id": job_id,
                                "job_type": "inference",
                                "status": "completed",
                                "progress": 100,
                                "message": "Completed",
                                "timestamp": int(time.time() * 1000),
                                "final": True  # Signal that this is the final message
                            }))
                            print(f"📤 Sent external completion message for job {job_id}")
                            
                            # Send a close signal after a brief delay
                            # Do not close the broadcast mid-run; final close happens once per job
                            
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
                                # ComfyUI sends binary data with 8-byte header format
                                try:
                                    image_data = msg
                                    
                                    # Check if this is ComfyUI's 8-byte header format
                                    if len(msg) >= 8:
                                        # ComfyUI sends preview images as binary WS messages with 8-byte header
                                        # The header indicates message type + format, followed by actual image data
                                        header = msg[:8]
                                        potential_image_data = msg[8:]
                                        
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
                                                        # Switch to "generating" status on first base64 image
                                                        if not manager.generation_started:
                                                            manager.generation_started = True
                                                            print(f"🎯 SWITCHED TO GENERATING - LOCKED FOREVER for job {job_id}")
                                                        
                                                        preview_evt = {
                                                            "job_id": job_id,
                                                            "job_type": "inference",
                                                            "status": "generating",
                                                            "preview_images": [optimized_preview],
                                                            "image_index": manager.image_index,
                                                            "timestamp": int(time.time() * 1000)
                                                        }
                                                        await broadcast_ws.send(_json.dumps(preview_evt))
                                                        manager.last_preview_ts = now
                                                        # Note: image_index is now set explicitly for sequential generation
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
                            "message": "",
                            #"image_index": manager.image_index  # Include image_index for sequential generation
                        }
                        
                        if isinstance(data, dict):
                            # Handle ComfyUI progress updates (most accurate)
                            if "progress" in data:
                                progress_value = data.get("progress", 0)
                                evt["progress"] = min(100, max(0, progress_value))
                                
                                # Update activity time
                                manager.last_activity_time = time.time()
                                
                                # Simple flag check - if generating, ALWAYS use generating status
                                if manager.generation_started:
                                    evt["status"] = "generating"
                                    evt["message"] = "Generating"
                                else:
                                    evt["status"] = "starting"
                                    evt["message"] = "Starting up"
                                
                                # Check for preview data in progress messages (ComfyUI often sends previews here)
                                if "preview" in data:
                                    preview = data.get("preview")
                                    print(f"🔍 Found preview in progress message for job {job_id}")
                                elif "image" in data:
                                    preview = data.get("image")
                                    print(f"🔍 Found image in progress message for job {job_id}")
                            
                            # Handle ComfyUI execution updates with progressive progress
                            elif data.get("type") == "execution_start":
                                if manager.generation_started:
                                    evt["status"] = "generating"
                                    evt["message"] = "Generating"
                                else:
                                    evt["status"] = "starting"
                                    evt["message"] = "Starting up"
                                evt["progress"] = 5
                                
                            elif data.get("type") == "executing" or data.get("type") == "execution_cached":
                                if manager.generation_started:
                                    evt["status"] = "generating"
                                    evt["message"] = "Generating"
                                else:
                                    evt["status"] = "starting"
                                    evt["message"] = "Starting up"
                                evt["progress"] = 10
                                # Record prompt_id if present for graceful close
                                pid = (data.get("data", {}) or {}).get("prompt_id")
                                if isinstance(pid, str) and pid:
                                    manager.last_prompt_id = pid
                                # Track the currently executing node for diagnostics
                                exec_data = data.get("data", {}) or {}
                                node_id = exec_data.get("node") or exec_data.get("node_id")
                                if node_id is not None:
                                    manager.last_executing_node = {
                                        "node_id": node_id,
                                        "prompt_id": manager.last_prompt_id,
                                        "timestamp": int(time.time() * 1000)
                                    }
                                
                            elif data.get("type") == "executed":
                                if manager.generation_started:
                                    evt["status"] = "generating"
                                    evt["message"] = "Generating"
                                else:
                                    evt["status"] = "starting"
                                    evt["message"] = "Starting up"
                                evt["progress"] = 50
                                
                                # Check if this is a save node execution - strong signal of completion
                                executed_data = data.get("data", {})
                                node_id = executed_data.get("node")
                                if node_id and "output" in executed_data:
                                    output_data = executed_data.get("output") or {}
                                    # Check if this node produced saved images
                                    if output_data and any("images" in v for v in output_data.values() if isinstance(v, dict)):
                                        print(f"🎯 Detected save node execution for node {node_id}")
                                        # This is likely our completion event
                                        prompt_id = executed_data.get("prompt_id") or manager.last_prompt_id
                                        if prompt_id:
                                            manager.completed_prompt_ids.add(prompt_id)
                                            manager.completion_data[prompt_id] = executed_data
                                            manager.completion_event.set()
                                            print(f"🔔 Signaled completion via save node execution for prompt {prompt_id}")
                                
                            elif data.get("type") == "execution_complete" or data.get("type") == "execution_success":
                                event_type = data.get("type")
                                print(f"🏁 {event_type.upper()} received for job {job_id}!")
                                
                                # Don't send "completed" status for individual images - keep generating
                                evt["status"] = "generating"
                                evt["progress"] = 90  # High progress but not 100%
                                evt["message"] = f"Image {manager.image_index + 1} completed"
                                
                                # Signal completion for the current prompt
                                prompt_id = data.get("data", {}).get("prompt_id")
                                if prompt_id:
                                    manager.completed_prompt_ids.add(prompt_id)
                                    # Store the full completion data (contains outputs!)
                                    manager.completion_data[prompt_id] = data.get("data", {})
                                    # Track last seen prompt id
                                    manager.last_prompt_id = prompt_id
                                    print(f"🎯 Prompt {prompt_id} completed for job {job_id}")
                                    print(f"📊 Stored completion data keys: {list(data.get('data', {}).keys())}")
                                
                                # Signal the completion event (but don't break the loop)
                                manager.completion_event.set()
                                print(f"🔔 Signaled completion event for job {job_id}")
                                
                                # Send the completion message
                                if broadcast_ws:
                                    try:
                                        completion_message = _json.dumps(evt)
                                        await broadcast_ws.send(completion_message)
                                        print(f"📤 Sent completion message for job {job_id}")
                                    except Exception as completion_e:
                                        print(f"⚠️ Failed to send completion message: {completion_e}")
                                
                                # Don't break - keep relay alive for next images
                                print(f"🏁 Image completed but keeping relay alive for job {job_id}")
                            
                            # Capture ComfyUI execution errors when present
                            elif str(data.get("type", "")).lower() in [
                                "execution_error", "node_execution_error", "execution_failed", "error"
                            ]:
                                err_type = str(data.get("type") or "execution_error")
                                err_data = data.get("data", {}) or {}
                                # Try to extract commonly provided fields
                                prompt_id = err_data.get("prompt_id") or manager.last_prompt_id
                                node_id = err_data.get("node") or err_data.get("node_id")
                                exception_type = err_data.get("exception_type") or data.get("exception_type")
                                message = (
                                    err_data.get("message")
                                    or data.get("message")
                                    or str(err_data)[:500]
                                )
                                traceback_text = err_data.get("traceback") or data.get("traceback")
                                # Store error details
                                error_details = {
                                    "event_type": err_type,
                                    "prompt_id": prompt_id,
                                    "node_id": node_id,
                                    "exception_type": exception_type,
                                    "message": message,
                                    "traceback": traceback_text,
                                }
                                manager.last_error = error_details
                                if isinstance(prompt_id, str):
                                    manager.errors_by_prompt[prompt_id] = error_details
                                # Optional verbose traceback head
                                debug_env = os.environ.get("COMFYUI_DEBUG_ERRORS", "false").lower() in ("1", "true", "yes")
                                if debug_env and traceback_text:
                                    try:
                                        head = "\n".join(str(traceback_text).splitlines()[:20])
                                        print(f"❌ Captured ComfyUI error [{exception_type or 'Unknown'}]: {message}\n{head}")
                                    except Exception:
                                        print(f"❌ Captured ComfyUI error [{exception_type or 'Unknown'}]: {message}")
                                else:
                                    print(f"❌ Captured ComfyUI execution error for job {job_id}: {exception_type or 'Unknown'} - {message}")
                            
                            elif data.get("type") == "progress_state":
                                # Store progress state for fallback completion detection
                                progress_data = data.get("data", {})
                                nodes = progress_data.get("nodes", {})
                                
                                if nodes:
                                    # Store the latest progress state for fallback completion
                                    manager.last_progress_state = progress_data
                                
                                # Update activity time
                                manager.last_activity_time = time.time()
                            
                            else:
                                # Log ALL unhandled event types to debug missing execution_complete
                                event_type = data.get("type", "unknown")
                                
                                # Log the full data for unknown events to see if execution_complete is being missed
                                if event_type not in ["progress", "status", "progress_state"]:
                                    print(f"🔍 Full unknown event data: {data}")
                                
                                # Check if this might be a completion event with a different structure
                                if "complete" in event_type.lower() or "finish" in event_type.lower() or "done" in event_type.lower() or "success" in event_type.lower():
                                    print(f"🚨 POTENTIAL COMPLETION EVENT DETECTED: {event_type}")
                                    print(f"🚨 Full data: {data}")
                                    
                                    # Try to extract prompt_id and treat as completion
                                    prompt_id = None
                                    event_data = data.get("data", {})
                                    
                                    # Try different ways to get prompt_id
                                    if isinstance(event_data, dict):
                                        prompt_id = event_data.get("prompt_id") or event_data.get("id") or event_data.get("prompt")
                                    
                                    if prompt_id:
                                        print(f"🎯 Found prompt_id in {event_type} event: {prompt_id}")
                                        
                                        # Treat as completion but don't send "completed" status for individual images
                                        evt["status"] = "generating"
                                        evt["progress"] = 90  # High progress but not 100%
                                        evt["message"] = f"Image {manager.image_index + 1} completed"
                                        
                                        manager.completed_prompt_ids.add(prompt_id)
                                        manager.completion_data[prompt_id] = {
                                            "prompt_id": prompt_id,
                                            "outputs": {},  # Will use history fallback
                                            "completion_source": event_type
                                        }
                                        manager.completion_event.set()
                                        
                                        print(f"🔔 Signaled completion via {event_type} event for job {job_id}")
                                        
                                        # Send completion message
                                        if broadcast_ws:
                                            try:
                                                completion_message = _json.dumps(evt)
                                                await broadcast_ws.send(completion_message)
                                                print(f"📤 Sent completion message via {event_type} for job {job_id}")
                                            except Exception as completion_e:
                                                print(f"⚠️ Failed to send completion message: {completion_e}")
                                    else:
                                        print(f"⚠️ No prompt_id found in {event_type} event")
                            
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
                                    has_saved_images = False
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
                                                        has_saved_images = True
                                                        # This is a strong signal of completion - image has been saved
                                                        prompt_id = exec_data.get("prompt_id") or manager.last_prompt_id
                                                        if prompt_id and prompt_id not in manager.completed_prompt_ids:
                                                            print(f"🎯 Detected saved image output, marking prompt {prompt_id} as complete")
                                                            manager.completed_prompt_ids.add(prompt_id)
                                                            manager.completion_data[prompt_id] = exec_data
                                                            manager.completion_event.set()
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
                                        evt["image_index"] = manager.image_index
                                        manager.last_preview_ts = now
                                        
                                        # Switch to "generating" status on first base64 image
                                        if not manager.generation_started:
                                            manager.generation_started = True
                                            print(f"🎯 SWITCHED TO GENERATING - LOCKED FOREVER for job {job_id}")
                                        
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
                        # If job is marked complete, send final and exit; otherwise try graceful completion + reconnect
                        if start_relay._completion_flags.get(job_id, False):
                            try:
                                final_msg = {
                                    "job_id": job_id,
                                    "job_type": "inference",
                                    "status": "completed",
                                    "progress": 100,
                                    "message": "Completed (connection closed)",
                                    "timestamp": int(time.time() * 1000)
                                }
                                if broadcast_ws:
                                    await broadcast_ws.send(_json.dumps(final_msg))
                                    print(f"📤 Sent final message on connection close for job {job_id}")
                            except Exception as final_close_e:
                                print(f"⚠️ Failed to send final message on close for job {job_id}: {final_close_e}")
                            break
                        else:
                            print(f"🔌 ComfyUI connection closed but job {job_id} not flagged as complete - triggering graceful completion fallback")
                            # Graceful fallback: if we have a recent prompt_id, treat as completed
                            try:
                                if manager.last_prompt_id and manager.last_prompt_id not in manager.completed_prompt_ids:
                                    manager.completed_prompt_ids.add(manager.last_prompt_id)
                                    manager.completion_data[manager.last_prompt_id] = {
                                        "prompt_id": manager.last_prompt_id,
                                        "outputs": {},
                                        "completion_source": "connection_closed"
                                    }
                                    manager.completion_event.set()
                                    print(f"🔔 Set completion for prompt {manager.last_prompt_id} due to connection close")
                            except Exception as _grace_e:
                                print(f"⚠️ Graceful close completion fallback failed: {_grace_e}")
                            # Attempt to reconnect to ComfyUI WS and continue
                            try:
                                print(f"🔄 Attempting to reconnect to ComfyUI WebSocket for job {job_id}...")
                                comfy_ws = await websockets.connect(
                                    comfy_ws_url,
                                    ping_interval=30,
                                    ping_timeout=20,
                                    close_timeout=10,
                                    open_timeout=30,
                                    max_size=max_size,
                                    compression=None
                                )
                                print(f"🔄 Reconnected to ComfyUI WebSocket for job {job_id}")
                                continue
                            except Exception as re_ws_e:
                                print(f"❌ Reconnect to ComfyUI WS failed for job {job_id}: {re_ws_e}")
                                break
                    except Exception as msg_e:
                        print(f"⚠️ Message processing error for job {job_id}: {msg_e}")
                        continue
                        
            finally:
                # Send final completion message and single close before shutting down (only if connected)
                if broadcast_ws:
                    try:
                        final_msg = {
                            "job_id": job_id,
                            "job_type": "inference",
                            "status": "completed",
                            "progress": 100,
                            "message": "Completed",
                            "timestamp": int(time.time() * 1000)
                        }
                        await broadcast_ws.send(_json.dumps(final_msg))
                        print(f"📤 Sent final completion message for job {job_id}")
                        # Now send a single close signal marked final
                        close_msg = {
                            "job_id": job_id,
                            "job_type": "inference",
                            "status": "closed",
                            "final": True,
                            "close_connection": True
                        }
                        await asyncio.sleep(0.1)
                        await broadcast_ws.send(_json.dumps(close_msg))
                        print(f"📤 Sent final close signal for job {job_id}")
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
        
        # Clean up stored global status
        if hasattr(send_global_job_status, '_global_statuses') and job_id in send_global_job_status._global_statuses:
            send_global_job_status._global_statuses.pop(job_id, None)
            print(f"🧹 Cleaned up stored global status for job {job_id}")
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
        
        # Clean up all stored global statuses
        if hasattr(send_global_job_status, '_global_statuses'):
            send_global_job_status._global_statuses.clear()
            print(f"🧹 Cleaned up all stored global statuses")
        
        print(f"✅ Cleaned up all active relays")


def get_active_relays() -> list:
    """Get list of currently active relay job IDs."""
    if hasattr(start_relay, '_active_relays'):
        return list(start_relay._active_relays.keys())
    return []


def send_custom_message_to_job(job_id: str, message_data: dict) -> bool:
    """Send a custom message through the existing WebSocket relay for a job."""
    if not hasattr(start_relay, '_active_relays'):
        print(f"⚠️ No active relays found for custom message to job {job_id}")
        return False
    
    manager = start_relay._active_relays.get(job_id)
    if not manager:
        print(f"⚠️ No active relay found for job {job_id} to send custom message")
        return False
    
    try:
        # Send the custom message through the existing WebSocket connection
        manager.send_message(message_data)
        print(f"📤 Sent custom message to job {job_id}: {message_data.get('status', 'unknown')}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send custom message to job {job_id}: {e}")
        return False


def update_job_image_index(job_id: str, image_index: int):
    """Update the image index for an active job relay."""
    if hasattr(start_relay, '_active_relays') and job_id in start_relay._active_relays:
        manager = start_relay._active_relays[job_id]
        manager.update_image_index(image_index)
        print(f"📸 Updated image index to {image_index} for job {job_id}")
    else:
        print(f"⚠️ No active relay found to update image index for job {job_id}")

def get_prompt_completion_data(job_id: str, prompt_id: str) -> dict:
    """Get the completion data for a specific prompt (contains outputs, etc.)."""
    if not hasattr(start_relay, '_active_relays') or job_id not in start_relay._active_relays:
        print(f"⚠️ No active relay found for job {job_id} to get completion data")
        return {}
    
    manager = start_relay._active_relays[job_id]
    completion_data = manager.completion_data.get(prompt_id, {})
    
    if completion_data:
        print(f"📊 Retrieved completion data for prompt {prompt_id}: {list(completion_data.keys())}")
    else:
        print(f"⚠️ No completion data found for prompt {prompt_id}")
    
    return completion_data

def get_prompt_error_data(job_id: str, prompt_id: Optional[str] = None) -> dict:
    """Get the most recent ComfyUI error details. If prompt_id provided, returns for that prompt."""
    if not hasattr(start_relay, '_active_relays') or job_id not in start_relay._active_relays:
        print(f"⚠️ No active relay found for job {job_id} to get error data")
        return {}
    manager = start_relay._active_relays[job_id]
    if prompt_id and prompt_id in manager.errors_by_prompt:
        return manager.errors_by_prompt.get(prompt_id, {}) or {}
    return manager.last_error or {}

def get_last_executing_node(job_id: str) -> dict:
    """Get the last executing node info for diagnostics."""
    if not hasattr(start_relay, '_active_relays') or job_id not in start_relay._active_relays:
        print(f"⚠️ No active relay found for job {job_id} to get last executing node")
        return {}
    manager = start_relay._active_relays[job_id]
    return manager.last_executing_node or {}

def wait_for_prompt_completion(job_id: str, prompt_id: str, timeout: float = 600) -> bool:
    """Wait for a specific prompt to complete using event-based signaling instead of polling."""
    import time
    
    if not hasattr(start_relay, '_active_relays') or job_id not in start_relay._active_relays:
        print(f"⚠️ No active relay found for job {job_id} to wait for completion")
        return False
    
    manager = start_relay._active_relays[job_id]
    
    # Check if already completed
    if prompt_id in manager.completed_prompt_ids:
        print(f"✅ Prompt {prompt_id} already completed for job {job_id}")
        return True
    
    print(f"⏳ Waiting for prompt {prompt_id} completion via WebSocket events (timeout: {timeout}s)")
    
    # Wait for completion event with timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Wait for the event with a short timeout to allow checking prompt_id
        if manager.completion_event.wait(timeout=1.0):
            # Event was set, check if our specific prompt completed
            if prompt_id in manager.completed_prompt_ids:
                print(f"🎯 Prompt {prompt_id} completed for job {job_id}")
                # Clear the event so it's ready for the next image
                manager.completion_event.clear()
                print(f"🔄 Cleared completion event for next image in job {job_id}")
                return True
            elif len(manager.completed_prompt_ids) == 0:
                # Completion event was set but no specific prompt_id was recorded
                # This might be a fallback completion (e.g., from progress_state)
                print(f"🎯 Generic completion detected for job {job_id}, assuming prompt {prompt_id} completed")
                manager.completed_prompt_ids.add(prompt_id)  # Add it for consistency
                # Clear the event so it's ready for the next image
                manager.completion_event.clear()
                print(f"🔄 Cleared completion event for next image in job {job_id}")
                return True
            else:
                # Different prompt completed, clear event and continue waiting
                manager.completion_event.clear()
                print(f"🔄 Different prompt completed, continuing to wait for {prompt_id}")
        
        # Check for inactivity-based completion (fallback)
        current_time = time.time()
        time_since_activity = current_time - manager.last_activity_time
        
        # Check for stuck node detection (if same node executing for too long)
        if manager.last_executing_node:
            node_timestamp = manager.last_executing_node.get('timestamp', 0)
            if node_timestamp > 0:
                time_on_same_node = current_time - (node_timestamp / 1000.0)  # Convert ms to seconds
                if time_on_same_node > 120:  # 2 minutes on same node = likely stuck
                    print(f"🚨 Node {manager.last_executing_node.get('node_id')} has been executing for {time_on_same_node:.1f}s - likely stuck")
                    print(f"🚨 Failing job {job_id} due to stuck node")
                    # Don't mark as completed, let it timeout and fail properly
                    return False
        
        # If we've been generating and no activity for 5 seconds, check if we should complete
        if manager.generation_started and time_since_activity > 5:
            # Check if the last progress state indicates completion
            last_progress = manager.last_progress_state
            if last_progress:
                nodes = last_progress.get("nodes", {})
                progress_prompt_id = last_progress.get("prompt_id")
                
                # CRITICAL: Only use fallback if the progress_state prompt_id matches what we're waiting for
                if nodes and progress_prompt_id == prompt_id:
                    running_count = sum(1 for node in nodes.values() if node.get("state") == "running")
                    finished_count = sum(1 for node in nodes.values() if node.get("state") == "finished")
                    
                    # If we have finished nodes and no running nodes, assume completion
                    if finished_count > 0 and running_count == 0:
                        print(f"🕐 No activity for {time_since_activity:.1f}s + all nodes finished for prompt {prompt_id}, assuming completion")
                        print(f"🎯 Using fallback completion from progress_state: {finished_count} finished, {running_count} running")
                        
                        manager.completed_prompt_ids.add(prompt_id)
                        
                        # Store minimal completion data
                        manager.completion_data[prompt_id] = {
                            "prompt_id": prompt_id,
                            "outputs": {},  # Will use history fallback
                            "completion_source": "inactivity_fallback"
                        }
                        
                        manager.completion_event.set()
                        return True
                    else:
                        print(f"🔄 Progress state shows {running_count} running nodes for prompt {prompt_id}, still waiting...")
                elif progress_prompt_id != prompt_id:
                    print(f"🔄 Progress state is for different prompt ({progress_prompt_id}), waiting for {prompt_id}...")
            
            # Original fallback if no progress state available or timeout
            if time_since_activity > 15:
                print(f"🕐 No activity for {time_since_activity:.1f}s, assuming completion for prompt {prompt_id}")
                manager.completed_prompt_ids.add(prompt_id)
                manager.completion_event.set()
                return True
    
    print(f"⏰ Timeout waiting for prompt {prompt_id} completion for job {job_id}")
    return False

def send_global_job_status(job_id: str, status: str, message: str = "") -> bool:
    """Send a global job status message (not per-image status)."""
    global_status_data = {
        "job_id": job_id,
        "job_type": "inference",
        "status": status,
        "message": message,
        "timestamp": int(time.time() * 1000),
        "global": True,  # Flag to indicate this is job-level, not image-level
        "progress": 0 if status == "initializing" else 50 if status == "ready" else 100
    }
    
    # Store the global status for future relay initialization
    if not hasattr(send_global_job_status, '_global_statuses'):
        send_global_job_status._global_statuses = {}
    send_global_job_status._global_statuses[job_id] = {
        "status": status,
        "message": message,
        "timestamp": int(time.time() * 1000)
    }
    print(f"💾 Stored global status for job {job_id}: {status} - {message}")
    
    # Try to send through existing relay first
    if send_custom_message_to_job(job_id, global_status_data):
        return True
    
    # If no relay exists yet, try to send via HTTP progress endpoint
    try:
        import json as _json
        import os
        
        # Get progress WebSocket URL and convert to HTTP endpoint
        base = os.environ.get("PROGRESS_WS_URL") or "wss://creativebuild--primeshot-inference-progress.modal.run"
        
        if base and "{job_id}" not in base:
            # Convert WebSocket URL to HTTP progress endpoint
            post_url = (
                base.replace("wss://", "https://")
                .replace("ws://", "http://")
                .rstrip("/") + f"/api/progress/{job_id}"
            )
            
            progress_body = _json.dumps(global_status_data).encode('utf-8')
            progress_req = urllib.request.Request(
                url=post_url,
                data=progress_body,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(progress_req, timeout=5)
            print(f"📤 Sent global status '{status}' via HTTP for job {job_id}")
            return True
            
    except Exception as e:
        print(f"⚠️ Failed to send global status via HTTP for job {job_id}: {e}")
    
    print(f"⚠️ Could not send global status '{status}' for job {job_id}")
    return False


def start_direct_comfyui_relay(progress_ws_url: str, comfy_ws_url: str, job_id: str, image_index: int = 0, throttle_sec: float = 1.5) -> None:
    """Alias for start_relay - connects directly to ComfyUI WebSocket."""
    start_relay(progress_ws_url, comfy_ws_url, job_id, throttle_sec, image_index)


