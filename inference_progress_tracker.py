"""
Progress tracking for ComfyUI inference jobs
Parses ComfyUI execution logs and streams progress via WebSocket in real-time
Also updates inference_jobs database table for persistence
"""
import time
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class InferenceProgressTracker:
    """Progress tracker for ComfyUI inference workflows with WebSocket streaming and database updates"""
    
    def __init__(self, job_id: str, total_images: int = 1, websocket_url: Optional[str] = None):
        self.job_id = job_id
        self.websocket_url = websocket_url
        self.websocket = None
        self.websocket_connected = False
        self.websocket_task = None  # Background task for WebSocket connection
        
        # Initialize message queue for thread-safe WebSocket communication
        import queue
        self.message_queue = queue.Queue()
        self.current_progress = 0
        self.is_completed = False  # Flag to prevent updates after completion
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        self.total_images = total_images
        self.database_started = False  # Track if we've marked inference as started in DB
        
        # Initialize Supabase client for database updates
        self.supabase_client = None
        try:
            from inference_supabase_client import get_inference_supabase_client
            self.supabase_client = get_inference_supabase_client()
            print(f"📊 Database integration enabled for inference job: {job_id}")
        except Exception as e:
            print(f"⚠️ Database integration failed: {e}")
            self.supabase_client = None
        
        # Don't initialize WebSocket connection in __init__ - do it when first needed
        print(f"🔧 Inference progress tracker initialized for job: {job_id}")
        if websocket_url:
            print(f"🔗 WebSocket URL configured: {websocket_url}")
        else:
            print("⚠️ No WebSocket URL provided - WebSocket tracking disabled")
        
        # ComfyUI workflow configuration
        self.current_phase = 'setup'
        self.workflow_nodes_completed = 0
        self.total_workflow_nodes = 0  # Will be detected from workflow
        self.has_upscaling = False  # Will be detected if workflow includes upscaling
        
        # Timing constants based on typical ComfyUI workflows
        self.PRE_EXECUTION_OVERHEAD = 15   # 15s for loading models and setup
        self.EXECUTION_TIME_PER_IMAGE = 8  # 8s per image for Flux inference
        self.UPSCALING_TIME_PER_IMAGE = 25 # 25s per image for 4K upscaling
        self.POST_EXECUTION_OVERHEAD = 5   # 5s for saving and cleanup
        
        # Calculate initial estimate
        base_execution_time = self.total_images * self.EXECUTION_TIME_PER_IMAGE
        self.total_estimated_duration = int(
            self.PRE_EXECUTION_OVERHEAD + 
            base_execution_time + 
            self.POST_EXECUTION_OVERHEAD
        )
        
    def _update_database_status(self, progress: int, message: str, status: str = 'processing'):
        """Update database with inference status"""
        if not self.supabase_client:
            return
            
        try:
            # Mark inference as started in database if this is the first real progress
            if not self.database_started and (progress > 0 or status == 'processing'):
                self.database_started = True
                print(f"📊 Marking inference job {self.job_id} as started in database")
                self.supabase_client.start_inference_job(job_id=self.job_id)
            
            # Update inference status only for major state changes
            if status != 'processing':  # Only update for major state changes
                self.supabase_client.update_inference_status(
                    job_id=self.job_id,
                    status=status,
                    progress=progress
                )
                print(f"📊 Updated inference job {self.job_id} status to: {status} ({progress}%)")
            
        except Exception as e:
            print(f"⚠️ Failed to update database status: {e}")
            logger.error(f"Database update failed: {e}")
    
    def estimate_remaining_time(self, progress: int) -> int:
        """Estimate remaining time based on workflow phase and progress"""
        if progress >= 100:
            return 0
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Phase 1: Pre-execution (progress < 10%)
        if progress < 10:
            remaining_pre_execution = max(0, self.PRE_EXECUTION_OVERHEAD - elapsed_time)
            execution_time = self.total_images * self.EXECUTION_TIME_PER_IMAGE
            if self.has_upscaling:
                execution_time += self.total_images * self.UPSCALING_TIME_PER_IMAGE
            remaining_time = remaining_pre_execution + execution_time + self.POST_EXECUTION_OVERHEAD
            
            logger.debug(f"Pre-execution estimate: {remaining_time:.0f}s")
            return max(0, int(remaining_time))
        
        # Phase 2: During execution (progress 10-90%)
        elif progress < 90:
            # Use linear interpolation based on elapsed time
            remaining_percentage = (100 - progress) / 100
            if elapsed_time > 0:
                total_estimated = elapsed_time / ((progress - 0) / 100) if progress > 0 else self.total_estimated_duration
                remaining_time = total_estimated * remaining_percentage
            else:
                remaining_time = self.total_estimated_duration * remaining_percentage
            
            logger.debug(f"Execution estimate: {remaining_time:.0f}s")
            return max(0, int(remaining_time))
        
        # Phase 3: Completion (progress >= 90%)
        else:
            remaining_time = self.POST_EXECUTION_OVERHEAD
            logger.debug(f"Completion estimate: {remaining_time}s")
            return max(0, int(remaining_time))
    
    def update_progress(self, progress: int, message: str) -> None:
        """Update progress with WebSocket streaming"""
        # Don't update if inference is already completed
        if self.is_completed:
            logger.info(f"Skipping progress update - job already completed: {progress}% - {message}")
            return
            
        if progress > self.current_progress:
            current_time = time.time()
            self.current_progress = progress
            
            # Ensure WebSocket connection is established before sending
            if self.websocket_url and not self.websocket_connected:
                self._ensure_websocket_connection()
            
            # Update phase tracking
            if progress >= 10 and progress < 90:
                self.current_phase = 'Generating...'
            elif progress >= 90:
                self.current_phase = 'Finalizing...'
            elif progress < 10:
                self.current_phase = 'Initializing...'
            
            # Calculate estimated remaining time
            estimated_remaining = self.estimate_remaining_time(progress)
            elapsed_time = int(current_time - self.start_time)
            
            # Prepare progress data for WebSocket
            progress_data = {
                "job_id": self.job_id,
                "progress": progress,
                "message": message,
                "timestamp": current_time,
                "estimated_remaining": estimated_remaining,
                "elapsed_time": elapsed_time,
                "phase": self.current_phase,
                "total_estimated_duration": self.total_estimated_duration,
                "total_images": self.total_images
            }
            
            # Mark as completed when reaching 100%
            if progress >= 100:
                self.is_completed = True
                progress_data["status"] = "completed"
                logger.info(f"[WEBSOCKET] Inference completed: {progress}% - {message}")
            else:
                progress_data["status"] = "processing"
            
            # Send via WebSocket (non-blocking) only if connected
            if self.websocket_url and self.websocket_connected:
                self._send_progress_sync(progress_data)
            
            self.last_progress_time = current_time
            
            # Update database status
            self._update_database_status(progress, message, progress_data["status"])
    
    def _ensure_websocket_connection(self):
        """Ensure WebSocket connection is established and start background task if needed"""
        if not self.websocket_url or self.websocket_task:
            return
            
        print(f"🔗 Starting WebSocket connection for inference job: {self.job_id}")
        
        # Start background WebSocket connection task
        import asyncio
        import threading
        
        def start_websocket_thread():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._websocket_connection())
            except Exception as e:
                print(f"❌ WebSocket thread error: {e}")
            finally:
                loop.close()
        
        self.websocket_task = threading.Thread(target=start_websocket_thread, daemon=True)
        self.websocket_task.start()
        self.websocket_connected = True
    
    async def _websocket_connection(self):
        """Maintain persistent WebSocket connection and handle sending queue"""
        import websockets
        import json
        import asyncio
        
        uri = f"{self.websocket_url}/ws/broadcast/{self.job_id}"
        
        while not self.is_completed:
            try:
                print(f"🔗 Connecting to WebSocket: {uri}")
                
                # Connect with proper ping/pong to maintain connection
                async with websockets.connect(
                    uri, 
                    ping_interval=15,   # Send ping every 15 seconds
                    ping_timeout=10,    # Wait 10 seconds for pong
                    close_timeout=5,    # Quick close timeout
                    max_size=2**20,     # 1MB max message size
                    open_timeout=10     # 10 second connection timeout
                ) as websocket:
                    print(f"✅ WebSocket connected for inference job: {self.job_id}")
                    
                    # Send any queued messages first
                    sent_queued = 0
                    while not self.message_queue.empty():
                        try:
                            message = self.message_queue.get_nowait()
                            json_data = json.dumps(message, ensure_ascii=False)
                            await websocket.send(json_data)
                            print(f"📤 Sent queued progress: {message.get('progress', 0)}%")
                            sent_queued += 1
                        except Exception as e:
                            print(f"❌ Failed to send queued message: {e}")
                            break
                    
                    if sent_queued > 0:
                        print(f"✅ Sent {sent_queued} queued messages")
                    
                    # Keep connection alive and process new messages
                    while not self.is_completed:
                        try:
                            # Check for new messages every 100ms for responsiveness
                            await asyncio.sleep(0.1)
                            
                            # Send any new messages from queue
                            messages_sent = 0
                            while not self.message_queue.empty() and messages_sent < 10:  # Batch limit
                                try:
                                    message = self.message_queue.get_nowait()
                                    
                                    # Ensure proper JSON formatting
                                    json_data = json.dumps(message, ensure_ascii=False, separators=(',', ':'))
                                    
                                    await websocket.send(json_data)
                                    print(f"📤 Sent progress: {message.get('progress', 0)}%")
                                    messages_sent += 1
                                    
                                    # Small delay between messages to avoid flooding
                                    if messages_sent > 1:
                                        await asyncio.sleep(0.01)
                                        
                                except websockets.exceptions.ConnectionClosed:
                                    print(f"⚠️ WebSocket connection closed during send")
                                    raise
                                except Exception as e:
                                    print(f"❌ Failed to send message: {e}")
                                    # Re-queue the message for retry
                                    self.message_queue.put(message)
                                    break
                                    
                        except asyncio.CancelledError:
                            print(f"🔄 WebSocket connection cancelled")
                            break
                        except websockets.exceptions.ConnectionClosed:
                            print(f"⚠️ WebSocket connection closed by server")
                            break
                        except Exception as e:
                            print(f"❌ WebSocket error in main loop: {e}")
                            break
                            
            except websockets.exceptions.ConnectionClosed as e:
                print(f"⚠️ WebSocket connection closed: {e}")
                await asyncio.sleep(2)  # Brief delay before reconnecting
            except Exception as e:
                print(f"❌ WebSocket connection error: {e}")
                await asyncio.sleep(2)  # Brief delay before reconnecting
        
        print(f"🔚 WebSocket connection loop ended for inference job: {self.job_id}")
    
    def _send_progress_sync(self, progress_data: dict):
        """Queue progress data for WebSocket sending"""
        if not self.websocket_url:
            return
            
        # Ensure WebSocket connection is running
        self._ensure_websocket_connection()
        
        # Add message to queue for WebSocket thread to send
        if hasattr(self, 'message_queue'):
            self.message_queue.put(progress_data)
            print(f"📋 Queued progress update: {progress_data.get('progress', 0)}%")
        else:
            print(f"⚠️ Message queue not ready, progress not sent: {progress_data.get('progress', 0)}%")
    
    def mark_completed(self, success: bool = True, error_message: str = None) -> None:
        """Mark the inference job as completed - should be called by main workflow"""
        if not self.is_completed:
            self.is_completed = True
            total_time = time.time() - self.start_time
            
            # Send final WebSocket update
            progress_data = {
                "job_id": self.job_id,
                "progress": 100,
                "message": "Inference completed successfully" if success else f"Inference failed: {error_message}",
                "timestamp": time.time(),
                "status": "completed" if success else "failed",
                "elapsed_time": int(total_time),
                "estimated_remaining": 0,
                "phase": "completed",
                "error_message": error_message,
                "total_images": self.total_images
            }
            
            if self.websocket_url and self.websocket_connected:
                self._send_progress_sync(progress_data)
            
            logger.info(f"Inference job {self.job_id} completed in {total_time:.1f}s - {'success' if success else 'failed'}")
            
            # Complete inference job in database
            if self.supabase_client:
                try:
                    self.supabase_client.complete_inference_job(
                        job_id=self.job_id,
                        success=success,
                        error_message=error_message
                    )
                    print(f"📊 Inference job {self.job_id} marked as {'completed' if success else 'failed'} in database")
                except Exception as e:
                    print(f"⚠️ Failed to complete inference job in database: {e}")
                    logger.error(f"Database completion failed: {e}")
    
    def detect_workflow_features(self, workflow: dict) -> None:
        """Analyze workflow to detect features that affect timing"""
        try:
            # Count total nodes for progress estimation
            self.total_workflow_nodes = len(workflow)
            
            # Detect upscaling nodes
            upscale_node_types = [
                "UpscaleModelLoader", "ImageUpscaleWithModel", "UltimateSDUpscale",
                "ControlNetApplyAdvanced", "LatentUpscale", "ImageScale"
            ]
            
            for node_id, node_data in workflow.items():
                class_type = node_data.get("class_type", "")
                if any(upscale_type in class_type for upscale_type in upscale_node_types):
                    self.has_upscaling = True
                    print(f"🔍 Detected upscaling workflow (node: {class_type})")
                    break
            
            # Update time estimate if upscaling detected
            if self.has_upscaling:
                upscaling_time = self.total_images * self.UPSCALING_TIME_PER_IMAGE
                self.total_estimated_duration += upscaling_time
                print(f"⏱️ Updated time estimate for upscaling: +{upscaling_time}s")
            
            print(f"🔍 Workflow analysis: {self.total_workflow_nodes} nodes, upscaling: {self.has_upscaling}")
            
        except Exception as e:
            logger.error(f"Failed to analyze workflow features: {e}")
    
    def update_progress_from_execution_log(self, log_line: str) -> None:
        """Update progress based on ComfyUI execution logs"""
        try:
            # Phase 1: Initialization (0-10%)
            if "Health check" in log_line or "poll_server_health" in log_line:
                self.update_progress(1, "Health check")
            elif "Loading workflow template" in log_line or "load_workflow_template" in log_line:
                self.update_progress(3, "Loading workflow")
            elif "Validating workflow" in log_line or "validate_workflow_request" in log_line:
                self.update_progress(5, "Validating workflow")
            elif "Setting up LoRA" in log_line or "setup_style_lora" in log_line:
                self.update_progress(7, "Setting up LoRA")
            elif "Injecting parameters" in log_line or "inject_parameters" in log_line:
                self.update_progress(9, "Configuring workflow")
            
            # Phase 2: Execution (10-85%)
            elif "Executing workflow" in log_line or "execute_workflow" in log_line:
                self.update_progress(10, "Starting generation")
            elif "prompt queued" in log_line.lower():
                self.update_progress(15, "Queued for processing")
            elif "executing" in log_line.lower() and "node" in log_line.lower():
                # Extract node execution progress if possible
                progress = self._extract_node_progress(log_line)
                if progress:
                    self.update_progress(progress, "Generating images")
            elif "sampling" in log_line.lower() or "denoise" in log_line.lower():
                self.update_progress(30, "Sampling")
            elif "latent" in log_line.lower() and ("decode" in log_line.lower() or "encode" in log_line.lower()):
                self.update_progress(60, "Processing latents")
            elif "upscal" in log_line.lower():
                self.update_progress(70, "Upscaling")
            elif "saving" in log_line.lower() and "image" in log_line.lower():
                self.update_progress(80, "Saving images")
            
            # Phase 3: Post-processing (85-100%)
            elif "Processing results" in log_line or "process_results" in log_line:
                self.update_progress(85, "Processing results")
            elif "Uploading to S3" in log_line or "upload" in log_line.lower() and "s3" in log_line.lower():
                self.update_progress(90, "Uploading results")
            elif "Workflow executed successfully" in log_line:
                self.update_progress(95, "Generation complete")
            elif "Generation completed" in log_line or "completed successfully" in log_line:
                self.update_progress(100, "Completed")
                    
        except Exception as e:
            logger.error(f"Error updating progress from execution log: {e}")
            # Don't raise - we don't want log parsing to break inference
    
    def _extract_node_progress(self, log_line: str) -> Optional[int]:
        """Extract node execution progress from ComfyUI logs"""
        try:
            # Look for patterns like "Executing node 5/20" or similar
            if "/" in log_line and "node" in log_line.lower():
                parts = log_line.split()
                for i, part in enumerate(parts):
                    if "/" in part and part.replace("/", "").replace(" ", "").isdigit():
                        current, total = part.split("/")
                        current = int(current.strip())
                        total = int(total.strip())
                        if total > 0:
                            # Map node progress to 10-80% range (70% range for execution)
                            node_progress = (current / total) * 70
                            return int(10 + node_progress)
            return None
        except Exception:
            return None


class InferenceLogStreamTracker:
    """Captures ComfyUI execution output and updates progress in real-time"""
    
    def __init__(self, job_id: str, total_images: int = 1, websocket_url: Optional[str] = None):
        self.progress_tracker = InferenceProgressTracker(job_id, total_images, websocket_url=websocket_url)
        self.buffer = ""
        self.line_count = 0
    
    def process_output(self, output: str) -> None:
        """
        Process ComfyUI execution output and update progress
        
        Args:
            output: Raw execution output (may contain multiple lines)
        """
        # Add to buffer and process complete lines
        self.buffer += output
        
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.strip()
            
            if line:  # Skip empty lines
                self.line_count += 1
                
                # Log the line for debugging (rate limited)
                if self.line_count <= 20 or self.line_count % 50 == 0:
                    logger.debug(f"ComfyUI log #{self.line_count}: {line}")
                
                # Update progress based on log content
                self.progress_tracker.update_progress_from_execution_log(line)