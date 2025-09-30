"""ComfyUI server management and health checking."""

import os
import subprocess
import time
import urllib.request
import urllib.error
import socket
from typing import Dict, Any, Optional
import logging
import shutil

from .constants import (
    COMFYUI_PORT,
    COMFYUI_STARTUP_TIMEOUT,
    COMFYUI_PATH,
    MODELS_PATH,
    HTTP_REQUEST_TIMEOUT
)

logger = logging.getLogger(__name__)


class ComfyUIServer:
    """Manages ComfyUI server lifecycle and configuration."""
    
    def __init__(self, port: int = COMFYUI_PORT):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        
    def setup_model_paths(self) -> bool:
        """Configure ComfyUI model paths via extra_model_paths.yaml."""
        logger.info("🔧 Configuring ComfyUI model paths...")
        
        source_config = "/extra_model_paths.yaml"
        target_config = f"{COMFYUI_PATH}/extra_model_paths.yaml"
        
        try:
            if os.path.exists(source_config):
                shutil.copy2(source_config, target_config)
                logger.info(f"✅ Copied model paths config: {source_config} -> {target_config}")
            else:
                logger.warning(f"⚠️ Model paths config not found: {source_config}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to copy model paths config: {e}")
            return False
        
        # Create LoRA directory
        lora_dir = f"{COMFYUI_PATH}/models/loras"
        os.makedirs(lora_dir, exist_ok=True)
        
        logger.info("✅ ComfyUI configured to use /models volume via extra_model_paths.yaml")
        return True
    
    def launch(self, **kwargs) -> None:
        """Launch ComfyUI server with optimized settings."""
        logger.info(f"🔄 Launching ComfyUI on port {self.port}...")
        
        # Setup environment
        env = self._get_optimized_env()
        
        # Build command
        cmd = self._build_launch_command(**kwargs)
        
        # Launch server
        subprocess.run(cmd, shell=True, check=True, env=env)
        logger.info("✅ ComfyUI server launched successfully")
        
        # Wait for server to be ready
        self.wait_until_ready()
    
    def _get_optimized_env(self) -> Dict[str, str]:
        """Get optimized environment variables for ComfyUI."""
        env = os.environ.copy()
        env.update({
            # Performance optimization
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,backend:cudaMallocAsync',
            'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
            'NVIDIA_TF32_OVERRIDE': '1',
            'SAGE_ATTENTION_BACKEND': 'triton',
            'XFORMERS_ENABLE_TRITON': '1',
            
            # Force GPU usage
            'COMFYUI_MODEL_DEVICE': 'cuda',
            'COMFYUI_VAE_DEVICE': 'cuda',
            'COMFYUI_CLIP_DEVICE': 'cuda',
            'COMFYUI_LOWVRAM': 'false',
            'COMFYUI_NOVRAM': 'false',
            
            # Threading optimization
            'OMP_NUM_THREADS': '8',
            'MKL_NUM_THREADS': '8',
            
            # CUDA settings
            'TORCH_CUDA_ARCH_LIST': '9.0',  # H100 compute capability
            'FORCE_CUDA': '1',
            'CUDA_LAUNCH_BLOCKING': '0',
            
            # Memory optimization
            'PYTORCH_NO_CUDA_MEMORY_CACHING': '0',
            'TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT': '32',
            'TORCH_CUDNN_V8_API_ENABLED': '1',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            
            # Locale settings
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
            'LANGUAGE': 'en_US:en',
            
            # Non-interactive mode
            'DEBIAN_FRONTEND': 'noninteractive',
            'PIP_NO_INPUT': '1',
            'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        })
        return env
    
    def _build_launch_command(self, **kwargs) -> str:
        """Build ComfyUI launch command with options."""
        output_dir = kwargs.get('output_dir', f'{COMFYUI_PATH}/output')
        preview_method = kwargs.get('preview_method', 'auto')
        listen_addr = kwargs.get('listen', '127.0.0.1')
        
        cmd_parts = [
            'comfy launch',
            '--background' if kwargs.get('background', True) else '',
            '--',
            f'--port {self.port}',
            '--use-sage-attention',
            '--gpu-only',
            '--bf16-unet',
            '--bf16-vae',
            f'--output-directory {output_dir}',
            f'--preview-method {preview_method}',
        ]
        
        # Add listen address for dev server
        if listen_addr != '127.0.0.1':
            cmd_parts.append(f'--listen {listen_addr}')
        
        return ' '.join(filter(None, cmd_parts))
    
    def wait_until_ready(self, timeout: int = COMFYUI_STARTUP_TIMEOUT) -> bool:
        """Wait for ComfyUI server to become ready."""
        logger.info(f"⏳ Waiting for ComfyUI to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_health():
                logger.info("✅ ComfyUI is ready and responding")
                return True
            time.sleep(1)
        
        logger.error(f"❌ ComfyUI failed to become ready within {timeout} seconds")
        return False
    
    def check_health(self) -> bool:
        """Check if ComfyUI server is healthy."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            urllib.request.urlopen(req, timeout=HTTP_REQUEST_TIMEOUT)
            return True
        except (socket.timeout, urllib.error.URLError, Exception):
            return False
    
    def submit_prompt(
        self,
        workflow: Dict[str, Any],
        client_id: str,
        workflow_key: str = ""
    ) -> Dict[str, Any]:
        """Submit a workflow prompt to ComfyUI.
        
        Returns:
            Response data including prompt_id
        """
        import json
        
        body = {
            "prompt": workflow,
            "client_id": client_id,
            "workflow_key": workflow_key
        }
        
        req = urllib.request.Request(
            url=f"{self.base_url}/prompt",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            response = urllib.request.urlopen(req, timeout=30)
            response_data = response.read().decode('utf-8')
            return json.loads(response_data)
            
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            logger.error(f"❌ ComfyUI /prompt HTTP {e.code}. Body: {error_body[:2000]}")
            raise RuntimeError(f"HTTP {e.code} from ComfyUI /prompt: {error_body[:200]}")
    
    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get prompt history from ComfyUI."""
        import json
        
        try:
            response = urllib.request.urlopen(
                f"{self.base_url}/history/{prompt_id}",
                timeout=HTTP_REQUEST_TIMEOUT
            )
            history_data = json.loads(response.read().decode('utf-8'))
            return history_data.get(prompt_id)
        except Exception as e:
            logger.warning(f"Failed to get history for {prompt_id}: {e}")
            return None
    
    def poll_history(
        self,
        prompt_id: str,
        max_attempts: int = 8,
        backoff_base: float = 0.25
    ) -> Optional[Dict[str, Any]]:
        """Poll history API with exponential backoff."""
        for attempt in range(max_attempts):
            result = self.get_history(prompt_id)
            if result and result.get("outputs"):
                return result
            
            if attempt < max_attempts - 1:
                backoff = min(backoff_base * (2 ** attempt), 2.0)
                time.sleep(backoff)
        
        return None
    
    @staticmethod
    def summarize_workflow(workflow: Dict[str, Any]) -> None:
        """Log a summary of the workflow for debugging."""
        try:
            class_types = sorted({
                node.get("class_type", "")
                for node in workflow.values()
                if isinstance(node, dict)
            })
            
            titles = [
                node.get("_meta", {}).get("title")
                for node in workflow.values()
                if isinstance(node, dict) and node.get("_meta", {}).get("title")
            ]
            
            logger.info(f"🔎 Workflow summary: nodes={len(workflow)}, classes={class_types[:12]}")
            if titles:
                logger.info(f"🔎 Titles include: {titles[:12]}")
            
            # Check for required nodes
            from .constants import LATENT_NODE_TYPES
            has_latent = any(
                isinstance(n, dict) and n.get("class_type") in LATENT_NODE_TYPES
                for n in workflow.values()
            )
            
            if not has_latent:
                logger.warning(f"⚠️ No latent node found ({'/'.join(LATENT_NODE_TYPES)})")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to summarize workflow: {e}")


def stop_if_unhealthy(port: int = COMFYUI_PORT) -> None:
    """Check server health and stop container if unhealthy."""
    server = ComfyUIServer(port)
    if not server.check_health():
        logger.error("❌ Server health check failed")
        import modal.experimental
        modal.experimental.stop_fetching_inputs()
        raise Exception("ComfyUI server is not healthy, stopping container")
