"""
Smart WebSocket Server Manager for ComfyUI Inference Jobs
Handles server lifecycle with job registration and auto-shutdown
"""
import requests
import time
import os
import modal
from typing import Optional, Dict, Any

class InferenceWebSocketServerManager:
    """Manages WebSocket server lifecycle with smart job tracking for inference jobs"""
    
    def __init__(self):
        self.server_url = "https://creativebuild--inference-websocket-progress.modal.run"
        self.websocket_url = "wss://creativebuild--inference-websocket-progress.modal.run"
        
        print(f"🔧 Inference WebSocket Manager initialized:")
        print(f"   Server URL: {self.server_url}")
        print(f"   WebSocket URL: {self.websocket_url}")
        print(f"   Auth: Public endpoints (no auth required)")
        
    def is_server_running(self) -> bool:
        """Check if WebSocket server is currently running"""
        print(f"🔍 Checking if inference WebSocket server is running...")
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            print(f"   Health check response: {response.status_code}")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   Server status: {health_data}")
                return True
            else:
                print(f"   Server not healthy: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"   Health check failed: {e}")
            return False
    
    def start_server_if_needed(self) -> bool:
        """Ensure server is running. If down, attempt to start it automatically."""
        if self.is_server_running():
            print("✅ Inference WebSocket server is running")
            return True

        print("🏗️  Inference WebSocket server is not running – attempting auto-start via Modal…")

        if self._auto_start_server():
            print("✅ Inference WebSocket server started successfully")
            return True

        print("❌ Failed to start inference WebSocket server automatically")
        return False
    
    def register_job(self, job_id: str) -> bool:
        """Register a new inference job with the server"""
        print(f"📝 Registering inference job {job_id} with WebSocket server...")
        try:
            url = f"{self.server_url}/register/{job_id}"
            print(f"   POST URL: {url}")
            
            response = requests.post(url, timeout=10)
            
            print(f"   Registration response: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                print(f"   Response data: {response_data}")
                print(f"✅ Registered inference job {job_id} with WebSocket server")
                return True
            else:
                print(f"❌ Failed to register inference job {job_id}: {response.status_code}")
                print(f"   Response text: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Failed to register inference job {job_id}: {e}")
            return False
    
    def unregister_job(self, job_id: str) -> bool:
        """Unregister an inference job from the server"""
        print(f"🗑️ Unregistering inference job {job_id} from WebSocket server...")
        try:
            response = requests.post(f"{self.server_url}/unregister/{job_id}", timeout=10)
            print(f"   Unregistration response: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                print(f"   Response data: {response_data}")
                print(f"✅ Unregistered inference job {job_id} from WebSocket server")
                return True
            else:
                print(f"❌ Failed to unregister inference job {job_id}: {response.status_code}")
                print(f"   Response text: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Failed to unregister inference job {job_id}: {e}")
            return False
    
    def get_server_status(self) -> Optional[Dict[str, Any]]:
        """Get detailed server status"""
        print(f"📊 Getting detailed inference server status...")
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            print(f"   Status response: {response.status_code}")
            if response.status_code == 200:
                status_data = response.json()
                print(f"   Status data: {status_data}")
                return status_data
            else:
                print(f"   Status check failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"   Status check error: {e}")
            return None
    
    def ensure_server_for_job(self, job_id: str) -> tuple[bool, str]:
        """Ensure WebSocket server is running and register inference job"""
        print(f"\n🚀 Setting up inference WebSocket server for job: {job_id}")
        
        # Check if server is running
        print("Step 1: Checking server status...")
        if not self.start_server_if_needed():
            print("❌ Server not available")
            return False, ""
        
        # Register the job
        print("Step 2: Registering inference job...")
        if self.register_job(job_id):
            print(f"✅ Inference WebSocket setup complete for job {job_id}")
            print(f"   WebSocket URL: {self.websocket_url}")
            return True, self.websocket_url
        else:
            print(f"❌ Failed to register inference job {job_id}")
            return False, ""
    
    def cleanup_job(self, job_id: str):
        """Clean up inference job registration"""
        print(f"\n🧹 Cleaning up inference job: {job_id}")
        # WebSocket connections clean up automatically when containers shut down
        # No need to explicitly unregister - this prevents timeout errors
        print(f"✅ Inference job {job_id} cleanup completed (WebSocket auto-cleanup)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_start_server(self) -> bool:
        """Attempt to spawn the Modal WebSocket server and wait until healthy."""
        try:
            # Lookup the deployed function by name (<app>, <function>)
            # NB: App name must match inference_websocket_progress.py
            fn = modal.Function.lookup("inference-websocket", "progress")

            print("🚀 Spawning Modal inference WebSocket server container…")
            handle = fn.spawn()
            print(f"   Spawn handle: {handle}")

            # Wait up to 60 s for health endpoint to pass
            max_attempts = 20
            for attempt in range(max_attempts):
                time.sleep(3)
                if self.is_server_running():
                    return True
                print(f"   Waiting for server to become healthy… ({attempt + 1}/{max_attempts})")

            print("❌ Timed-out waiting for inference WebSocket server health check")
            return False

        except Exception as e:
            print(f"❌ Auto-start failed: {e}")
            return False

# Global instance
inference_websocket_manager = InferenceWebSocketServerManager()