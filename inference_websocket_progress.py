"""
WebSocket Progress Server for ComfyUI Inference Jobs
Provides real-time progress streaming for image generation workflows
Features smart lifecycle management with auto-shutdown
"""
import modal
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Modal app for WebSocket server
app = modal.App("inference-websocket")

@app.function(
    image=modal.Image.debian_slim().pip_install([
        "fastapi[standard]==0.115.4",
        "websockets>=12.0",
        "uvicorn[standard]>=0.24.0"
    ]),
    max_containers=1,
    timeout=7200
)
@modal.concurrent(max_inputs=1000)  # Allow many concurrent connections on single container
@modal.asgi_app()
def progress():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import JSONResponse
    
    app = FastAPI()
    
    # Store active connections: job_id -> list of client websockets
    active_connections: Dict[str, List[WebSocket]] = {}
    
    # Store latest progress for each job with metadata
    progress_data_store: Dict[str, dict] = {}
    
    # Track broadcast connections: job_id -> WebSocket
    broadcast_connections: Dict[str, WebSocket] = {}
    
    # Job lifecycle management
    active_inference_jobs: set = set()
    shutdown_timer_task: Optional[asyncio.Task] = None
    server_start_time = time.time()
    
    # Configuration
    SHUTDOWN_DELAY = 3 * 60  # 3 minutes in seconds (shorter for inference)
    
    def validate_progress_data(data: str, job_id: str) -> Optional[dict]:
        """Validate and parse progress data"""
        try:
            progress_data = json.loads(data)
            logger.info(f"[{job_id}] Raw progress data received: {progress_data}")
            
            # Add timestamp if missing
            if "timestamp" not in progress_data:
                progress_data["timestamp"] = datetime.now().isoformat()
            
            # Validate required fields
            if "progress" not in progress_data:
                logger.warning(f"[{job_id}] Missing 'progress' field in data: {progress_data}")
                return None
                
            # Ensure progress is numeric
            try:
                progress_value = float(progress_data["progress"])
                progress_data["progress"] = progress_value
                logger.info(f"[{job_id}] Validated progress: {progress_value}%")
            except (ValueError, TypeError):
                logger.error(f"[{job_id}] Invalid progress value: {progress_data.get('progress')}")
                return None
            
            return progress_data
            
        except json.JSONDecodeError as e:
            logger.error(f"[{job_id}] JSON decode error: {e} - Raw data: {data}")
            return None
        except Exception as e:
            logger.error(f"[{job_id}] Unexpected error validating progress data: {e}")
            return None
    
    async def cancel_shutdown_timer():
        """Cancel any pending shutdown timer"""
        nonlocal shutdown_timer_task
        if shutdown_timer_task and not shutdown_timer_task.done():
            shutdown_timer_task.cancel()
            logger.info("Shutdown timer cancelled")
    
    async def schedule_shutdown():
        """Schedule server shutdown after delay if no active jobs"""
        nonlocal shutdown_timer_task
        
        async def shutdown_after_delay():
            logger.info(f"Starting shutdown timer for {SHUTDOWN_DELAY} seconds")
            await asyncio.sleep(SHUTDOWN_DELAY)
            
            if not active_inference_jobs:
                logger.info("No active inference jobs - initiating server shutdown")
                # Clean up all data
                active_connections.clear()
                progress_data_store.clear()
                broadcast_connections.clear()
                logger.info("Server cleanup completed - ready for shutdown")
            else:
                logger.info(f"Active jobs detected: {active_inference_jobs} - shutdown cancelled")
        
        shutdown_timer_task = asyncio.create_task(shutdown_after_delay())
    
    @app.post("/register/{job_id}")
    async def register_job(job_id: str):
        """Register a new inference job starting"""
        active_inference_jobs.add(job_id)
        await cancel_shutdown_timer()
        logger.info(f"[{job_id}] Inference job registered. Total active jobs: {len(active_inference_jobs)}")
        return {"status": "registered", "job_id": job_id, "active_jobs": len(active_inference_jobs)}
    
    @app.post("/unregister/{job_id}")
    async def unregister_job(job_id: str):
        """Unregister completed/failed job"""
        active_inference_jobs.discard(job_id)
        logger.info(f"[{job_id}] Inference job unregistered. Remaining active jobs: {len(active_inference_jobs)}")
        
        # Clean up job data
        if job_id in progress_data_store:
            logger.info(f"[{job_id}] Cleaning up progress data")
            del progress_data_store[job_id]
        
        if job_id in broadcast_connections:
            logger.info(f"[{job_id}] Cleaning up broadcast connection")
            try:
                await broadcast_connections[job_id].close()
            except:
                pass
            del broadcast_connections[job_id]
            
        if job_id in active_connections:
            logger.info(f"[{job_id}] Closing {len(active_connections[job_id])} client connections")
            # Close all connections for this job
            for ws in active_connections[job_id]:
                try:
                    await ws.close()
                except Exception as e:
                    logger.warning(f"[{job_id}] Error closing client connection: {e}")
            del active_connections[job_id]
        
        # Schedule shutdown if no active jobs
        if not active_inference_jobs:
            logger.info("No active inference jobs remaining - scheduling shutdown")
            await schedule_shutdown()
        
        return {"status": "unregistered", "job_id": job_id, "active_jobs": len(active_inference_jobs)}
    
    @app.get("/job/{job_id}/status")
    async def job_status(job_id: str):
        """Get detailed status for a specific job"""
        latest_progress = progress_data_store.get(job_id, {})
        
        return {
            "job_id": job_id,
            "is_registered": job_id in active_inference_jobs,
            "has_broadcast_connection": job_id in broadcast_connections,
            "client_connections": len(active_connections.get(job_id, [])),
            "latest_progress": latest_progress,
            "last_update": latest_progress.get("timestamp"),
            "progress_value": latest_progress.get("progress"),
            "status": latest_progress.get("status")
        }
    
    @app.get("/debug/connections")
    async def debug_connections():
        """Debug endpoint to inspect all connections"""
        return {
            "active_inference_jobs": list(active_inference_jobs),
            "broadcast_connections": list(broadcast_connections.keys()),
            "client_connections": {
                job_id: len(conns) for job_id, conns in active_connections.items()
            },
            "progress_data": {
                job_id: {
                    "progress": data.get("progress"),
                    "status": data.get("status"), 
                    "timestamp": data.get("timestamp")
                } for job_id, data in progress_data_store.items()
            }
        }
    
    @app.websocket("/ws/progress/{job_id}")
    async def websocket_endpoint(websocket: WebSocket, job_id: str):
        """Client endpoint for receiving progress updates"""
        try:
            await websocket.accept()
            logger.info(f"[{job_id}] Client connected to inference progress stream")
            
            # Add to active connections
            if job_id not in active_connections:
                active_connections[job_id] = []
            active_connections[job_id].append(websocket)
            
            # Send latest progress if available
            if job_id in progress_data_store:
                try:
                    latest_data = json.dumps(progress_data_store[job_id])
                    await websocket.send_text(latest_data)
                    logger.info(f"[{job_id}] Sent latest progress to new client: {progress_data_store[job_id].get('progress', 'unknown')}%")
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to send latest progress to new client: {e}")
            else:
                logger.info(f"[{job_id}] No stored progress data for new client")
            
            try:
                # Keep connection alive and handle heartbeats
                while True:
                    try:
                        # Wait for client messages (heartbeat or close)
                        message = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                        if message == "ping":
                            await websocket.send_text("pong")
                            logger.debug(f"[{job_id}] Heartbeat exchanged with client")
                    except asyncio.TimeoutError:
                        # No message received, but connection is still alive
                        continue
                    except Exception as e:
                        logger.info(f"[{job_id}] Client connection closed or error: {e}")
                        break
                        
            except WebSocketDisconnect:
                logger.info(f"[{job_id}] Client disconnected normally")
            except Exception as e:
                logger.warning(f"[{job_id}] Client connection error: {e}")
                
        except Exception as e:
            logger.error(f"[{job_id}] Failed to accept client connection: {e}")
        finally:
            # Remove from active connections
            if job_id in active_connections and websocket in active_connections[job_id]:
                active_connections[job_id].remove(websocket)
                if not active_connections[job_id]:
                    del active_connections[job_id]
                    # Clean up progress data for completed jobs
                    if job_id in progress_data_store:
                        progress_data = progress_data_store[job_id]
                        if progress_data.get("status") in ["completed", "failed"]:
                            logger.info(f"[{job_id}] Cleaning up completed job progress data")
                            del progress_data_store[job_id]
                logger.info(f"[{job_id}] Client removed from active connections")
    
    @app.websocket("/ws/broadcast/{job_id}")
    async def broadcast_endpoint(websocket: WebSocket, job_id: str):
        """ComfyUI container endpoint for sending progress updates"""
        try:
            await websocket.accept()
            broadcast_connections[job_id] = websocket
            logger.info(f"[{job_id}] ComfyUI container connected for broadcasting")
            
            try:
                while True:
                    try:
                        # Check if WebSocket is still connected before trying to receive
                        if websocket.client_state.name != 'CONNECTED':
                            logger.info(f"[{job_id}] WebSocket no longer connected (state: {websocket.client_state.name}), breaking loop")
                            break
                            
                        # Receive progress data from ComfyUI container
                        raw_data = await websocket.receive_text()
                        logger.info(f"[{job_id}] Received raw progress data: {raw_data}")
                        
                        # Validate progress data
                        progress_data = validate_progress_data(raw_data, job_id)
                        if progress_data is None:
                            logger.warning(f"[{job_id}] Skipping invalid progress data, continuing connection")
                            continue
                        
                        # Store latest progress
                        progress_data_store[job_id] = progress_data
                        logger.info(f"[{job_id}] Stored progress: {progress_data.get('progress', 'unknown')}% - Status: {progress_data.get('status', 'unknown')}")
                        
                        # Broadcast to all connected clients
                        if job_id in active_connections:
                            client_count = len(active_connections[job_id])
                            logger.info(f"[{job_id}] Broadcasting to {client_count} clients")
                            
                            disconnected_clients = []
                            successful_broadcasts = 0
                            
                            for i, client_ws in enumerate(active_connections[job_id]):
                                try:
                                    await client_ws.send_text(raw_data)
                                    successful_broadcasts += 1
                                    logger.debug(f"[{job_id}] Successfully sent to client {i+1}")
                                except Exception as e:
                                    logger.warning(f"[{job_id}] Failed to send to client {i+1}: {e}")
                                    disconnected_clients.append(client_ws)
                            
                            # Remove disconnected clients
                            for client_ws in disconnected_clients:
                                if client_ws in active_connections[job_id]:
                                    active_connections[job_id].remove(client_ws)
                            
                            logger.info(f"[{job_id}] Broadcast complete: {successful_broadcasts}/{client_count} successful, {len(disconnected_clients)} disconnected")
                            
                        else:
                            logger.warning(f"[{job_id}] No active client connections for broadcasting")
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"[{job_id}] Timeout receiving data, continuing...")
                        continue
                    except json.JSONDecodeError as e:
                        logger.error(f"[{job_id}] JSON decode error: {e}, continuing connection")
                        continue
                    except Exception as e:
                        logger.error(f"[{job_id}] Error processing message: {e}, continuing connection")
                        continue
                        
            except WebSocketDisconnect:
                logger.info(f"[{job_id}] ComfyUI container disconnected normally")
            except Exception as e:
                logger.error(f"[{job_id}] ComfyUI container connection error: {e}")
                
        except Exception as e:
            logger.error(f"[{job_id}] Failed to accept ComfyUI container connection: {e}")
        finally:
            if job_id in broadcast_connections:
                del broadcast_connections[job_id]
                logger.info(f"[{job_id}] ComfyUI container removed from broadcast connections")
    
    @app.get("/health")
    async def health_check():
        """Enhanced health check endpoint with lifecycle info"""
        uptime = time.time() - server_start_time
        shutdown_scheduled = shutdown_timer_task is not None and not shutdown_timer_task.done()
        
        return {
            "status": "healthy",
            "service": "ComfyUI Inference Progress",
            "uptime_seconds": round(uptime, 2),
            "active_inference_jobs": len(active_inference_jobs),
            "active_job_ids": list(active_inference_jobs),
            "broadcast_connections": len(broadcast_connections),
            "progress_data_jobs": len(progress_data_store),
            "active_client_connections": sum(len(conns) for conns in active_connections.values()),
            "shutdown_scheduled": shutdown_scheduled,
            "shutdown_delay_seconds": SHUTDOWN_DELAY
        }
    
    @app.post("/broadcast/{job_id}")
    async def http_broadcast(job_id: str, request: dict):
        """HTTP endpoint for sending progress updates (fallback for WebSocket issues)"""
        try:
            logger.info(f"[{job_id}] Received HTTP progress update: {request}")
            
            # Validate progress data
            if not isinstance(request, dict) or 'progress' not in request:
                return {"error": "Invalid progress data format"}
            
            # Store latest progress
            progress_data_store[job_id] = request
            logger.info(f"[{job_id}] Stored HTTP progress: {request.get('progress', 'unknown')}% - Status: {request.get('status', 'unknown')}")
            
            # Broadcast to all connected clients via WebSocket
            if job_id in active_connections:
                client_count = len(active_connections[job_id])
                logger.info(f"[{job_id}] Broadcasting HTTP update to {client_count} clients")
                
                disconnected_clients = []
                successful_broadcasts = 0
                
                # Convert to JSON string for WebSocket transmission
                message = json.dumps(request)
                
                for i, client_ws in enumerate(active_connections[job_id]):
                    try:
                        await client_ws.send_text(message)
                        successful_broadcasts += 1
                        logger.debug(f"[{job_id}] Successfully sent HTTP update to client {i+1}")
                    except Exception as e:
                        logger.warning(f"[{job_id}] Failed to send HTTP update to client {i+1}: {e}")
                        disconnected_clients.append(client_ws)
                
                # Remove disconnected clients
                for client_ws in disconnected_clients:
                    active_connections[job_id].remove(client_ws)
                
                logger.info(f"[{job_id}] HTTP broadcast complete: {successful_broadcasts}/{client_count} successful, {len(disconnected_clients)} disconnected")
                
                return {
                    "status": "success", 
                    "clients_notified": successful_broadcasts,
                    "clients_disconnected": len(disconnected_clients)
                }
            else:
                logger.warning(f"[{job_id}] No active client connections for HTTP broadcast")
                return {"status": "success", "clients_notified": 0, "message": "No active clients"}
                
        except Exception as e:
            logger.error(f"[{job_id}] HTTP broadcast error: {e}")
            return {"error": str(e)}

    @app.get("/status")
    async def server_status():
        """Detailed server status for monitoring"""
        uptime = time.time() - server_start_time
        shutdown_scheduled = shutdown_timer_task is not None and not shutdown_timer_task.done()
        
        return {
            "server": {
                "status": "running",
                "uptime_seconds": round(uptime, 2),
                "start_time": server_start_time
            },
            "jobs": {
                "active_inference_jobs": len(active_inference_jobs),
                "job_ids": list(active_inference_jobs),
                "broadcast_connections": len(broadcast_connections),
                "progress_data_stored": len(progress_data_store)
            },
            "connections": {
                "total_client_connections": sum(len(conns) for conns in active_connections.values()),
                "by_job": {job_id: len(conns) for job_id, conns in active_connections.items()},
                "broadcast_by_job": list(broadcast_connections.keys())
            },
            "lifecycle": {
                "shutdown_scheduled": shutdown_scheduled,
                "shutdown_delay_seconds": SHUTDOWN_DELAY,
                "will_shutdown_when_empty": True
            }
        }
    
    return app

if __name__ == "__main__":
    # For local development
    import uvicorn
    uvicorn.run("inference_websocket_progress:progress", host="0.0.0.0", port=8000)