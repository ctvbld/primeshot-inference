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
            
            # Connect to progress broadcast WebSocket (to send progress)
            print(f"🔌 Connecting to progress broadcast: {progress_ws_url}")
            broadcast_ws = await websockets.connect(
                progress_ws_url, 
                ping_interval=30,  # Keep connection alive
                max_size=max_size,
                compression=None
            )
            print(f"✅ Connected to progress broadcast for job {job_id}")
            
            # Send initial connection message
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
            
            try:
                message_count = 0
                
                while True:
                    # Check if manager still exists
                    manager = start_relay._active_relays.get(job_id)
                    if not manager:
                        print(f"⚠️ Relay manager cleaned up, stopping relay for job {job_id}")
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
                            # Handle ComfyUI progress updates
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
                            
                            # Handle ComfyUI execution updates
                            elif data.get("type") == "execution_start":
                                evt["status"] = "running"
                                evt["progress"] = 5
                                evt["message"] = "Starting generation..."
                                
                            elif data.get("type") == "executing":
                                node_id = data.get("data", {}).get("node")
                                evt["status"] = "running"
                                evt["progress"] = 25  # Estimated progress
                                evt["message"] = f"Processing node {node_id}..."
                                
                            elif data.get("type") == "execution_cached":
                                evt["status"] = "running"
                                evt["progress"] = 50
                                evt["message"] = "Using cached results..."
                                
                            elif data.get("type") == "execution_complete":
                                evt["status"] = "completed"
                                evt["progress"] = 100
                                evt["message"] = "Generation completed"
                            
                            # Handle preview images with memory management
                            preview = None
                            
                            # Check for different preview formats from ComfyUI
                            if data.get("type") == "preview":
                                # Direct preview message
                                preview = data.get("data", {}).get("image")
                            elif "image" in data:
                                # Direct image data
                                preview = data.get("image")
                            elif "preview" in data:
                                # Preview field
                                preview = data.get("preview")
                            elif data.get("type") == "executing" and "data" in data:
                                # Sometimes previews come with executing messages
                                exec_data = data.get("data", {})
                                if "preview" in exec_data:
                                    preview = exec_data.get("preview")
                            
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
                                
                                # Send to broadcast WebSocket
                                await broadcast_ws.send(message_json)
                                
                                # Log different types of updates
                                if "preview_images" in evt:
                                    print(f"📸 Sent preview update for job {job_id}")
                                elif "progress" in evt:
                                    print(f"📊 Sent progress update for job {job_id}: {evt.get('status', 'unknown')} - {evt.get('progress', 0)}%")
                                else:
                                    print(f"📤 Sent status update for job {job_id}: {evt.get('status', 'unknown')}")
                                
                                # Clean up message data
                                del message_json
                                
                            except websockets.exceptions.ConnectionClosed:
                                print(f"🔌 Broadcast WebSocket connection closed for job {job_id}, attempting reconnect...")
                                # Try to reconnect broadcast WebSocket
                                try:
                                    broadcast_ws = await websockets.connect(
                                        progress_ws_url, 
                                        ping_interval=30,
                                        max_size=max_size,
                                        compression=None
                                    )
                                    print(f"🔄 Reconnected broadcast WebSocket for job {job_id}")
                                    # Retry sending the message
                                    await broadcast_ws.send(message_json)
                                    print(f"📤 Resent message after reconnect for job {job_id}")
                                except Exception as reconnect_e:
                                    print(f"❌ Failed to reconnect broadcast WebSocket for job {job_id}: {reconnect_e}")
                                    break
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
                # Send final completion message before closing
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
                
                # Close WebSocket connections
                try:
                    await comfy_ws.close()
                    print(f"🔌 Closed ComfyUI WebSocket for job {job_id}")
                except:
                    pass
                    
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


