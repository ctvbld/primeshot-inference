# ---
# deploy: true  
# cmd: ["modal", "serve", "comfyui_app.py"]
# ---


import json
import subprocess
import uuid
import os
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import modal
import modal.experimental
from fastapi import Query

sys.path.insert(0, "/root/lib")

app = modal.App(name="primeshot-inference")

# Create Modal volumes for persistent storage
models_volume = modal.Volume.from_name("models-vol", create_if_missing=True)
aws_secret = modal.Secret.from_name("aws-secret")
inference_secret = modal.Secret.from_name("inference-secret")
supabase_secret = modal.Secret.from_name("supabase-secret")

# Define paths
MODELS_PATH = "/models"
PORT: int = 8000

# S3 mount for user LoRAs and outputs
# Mount user-images/ prefix at /data for all relevant functions
user_images_mount = {"/data": modal.CloudBucketMount(
    bucket_name="primeshot-uploads-01",
    key_prefix="user-images/",
    secret=aws_secret,
    read_only=False
)}

# Mount workflows/ prefix at /workflows for local file access to workflow JSON
workflows_mount = {"/workflows": modal.CloudBucketMount(
    bucket_name="primeshot-uploads-01",
    key_prefix="workflows/",
    secret=aws_secret,
    read_only=True
)}

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # Remove verbose logging by base image on entry
    .apt_install([
        "git", "build-essential", "cmake", "ninja-build",
        "locales", "pkg-config", "curl", "wget",
        # Graphics libraries for ComfyUI
        "libgl1-mesa-glx", "libglib2.0-0", "libfontconfig1",
        "libxrender1", "libxtst6", "libxi6", "libxrandr2", 
        "libasound2", "libgtk-3-0", "libsm6", "libxext6",
        "mesa-utils", "libgl1-mesa-dev", "libgles2-mesa-dev"
    ])

    .run_commands("locale-gen en_US.UTF-8")  # Generate English locale
    .env({
        # Locale settings
        "LANG": "en_US.UTF-8", 
        "LC_ALL": "en_US.UTF-8", 
        "LANGUAGE": "en_US:en",
        # Prevent interactive prompts in pip and other tools
        "DEBIAN_FRONTEND": "noninteractive",
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        # Performance GPU environment variables for H100 optimization
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,backend:cudaMallocAsync",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", 
        "NVIDIA_TF32_OVERRIDE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_LAUNCH_BLOCKING": "0",
        # Threading optimizations for inference
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        # Compilation settings for H100 architecture
        "TORCH_CUDA_ARCH_LIST": "9.0",  # H100 compute capability
        "FORCE_CUDA": "1",
        # Memory optimization
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
        "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "32",
        # SageAttention optimization flags
        "SAGE_ATTENTION_BACKEND": "triton",
        # Prevent CUDA issues with xformers
        "XFORMERS_ENABLE_TRITON": "1",  # Prevents xformers from calling torch.cuda.get_device_capability on import
        # H100 Performance: Force models to stay on GPU
        "COMFYUI_MODEL_DEVICE": "cuda",
        "COMFYUI_VAE_DEVICE": "cuda", 
        "COMFYUI_CLIP_DEVICE": "cuda",
        "COMFYUI_LOWVRAM": "false",
        "COMFYUI_NOVRAM": "false"
    })
    # Install optimized PyTorch 2.5+ with CUDA 12.1 (compatible with 12.8)
    .pip_install(
        "torch>=2.5.1", "torchvision>=0.20.1", "torchaudio>=2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121"
    )
    # Install Triton for SageAttention optimization
    .pip_install("triton>=3.2.0")
    # Install SageAttention 2.2.0+ using run_commands (same as working AI Toolkit)
    .run_commands([
        "pip install sageattention>=2.2.0 --no-deps"
    ])
    # Install web dependencies and ComfyUI
    .pip_install("fastapi[standard]==0.115.4")  # web dependencies
    .pip_install("comfy-cli==1.4.1")  # Install latest version of ComfyUI
    # Pre-install OpenCV to avoid conflicts with custom nodes
    .pip_install("opencv-python-headless==4.8.1.78")  # OpenCV without GUI dependencies
    # Install optimized xformers for additional attention acceleration
    .pip_install("xformers>=0.0.28")  # Latest xformers for attention optimizations
    .pip_install("websockets>=12.0")  # For ComfyUI WS preview/progress relay
    # Common dependencies required by various custom nodes
    .pip_install("diffusers>=0.30.0", "transformers>=4.42.0", "accelerate>=0.30.0", "safetensors>=0.4.3", "psutil>=6.0.0")
    .run_commands(  # install ComfyUI with NVIDIA support
        "comfy --skip-prompt install --fast-deps --nvidia"
    )
    # Install core custom nodes first (known to be stable)
    .run_commands(
        "comfy node install --fast-deps was-node-suite-comfyui@latest"
    )
    # Install rgthree for UI controls and labels (latest version for compatibility)
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/rgthree/rgthree-comfy.git",
        "cd /root/comfy/ComfyUI/custom_nodes/rgthree-comfy && pip install -r requirements.txt --no-input"
    )
    # Install ControlAltAI nodes for resolution controls (uses pyproject.toml)
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/gseth/ControlAltAI-Nodes.git",
        "cd /root/comfy/ComfyUI/custom_nodes/ControlAltAI-Nodes && pip install . --no-input"
    )
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes && pip install -r requirements.txt --no-input"
    )
    .run_commands(
       "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/yolain/ComfyUI-Easy-Use",
       "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Easy-Use && pip install -r requirements.txt --no-input"
    )
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/vrgamegirl19/comfyui-vrgamedevgirl",
        "cd /root/comfy/ComfyUI/custom_nodes/comfyui-vrgamedevgirl && pip install -r requirements.txt --no-input"
    )
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl"
    )
    # Install RES4LYF advanced sampling nodes
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/ClownsharkBatwing/RES4LYF.git",
        "cd /root/comfy/ComfyUI/custom_nodes/RES4LYF && pip install -r requirements.txt --no-input 2>/dev/null || echo 'No requirements.txt found for RES4LYF'"
    )
    # Install bilbox-comfyui photo prompt and post-processing nodes
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/syllebra/bilbox-comfyui.git", 
        "cd /root/comfy/ComfyUI/custom_nodes/bilbox-comfyui && pip install -r requirements.txt --no-input"
    )
    # Post-installation compatibility fixes for attention mask issues
    .run_commands(
        # Clear any cached model files that might cause conflicts
        "find /root/comfy/ComfyUI -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        # Verify Python environment is clean
        "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); import xformers; print(f\"xformers: {xformers.__version__}\")'"
    )

    # Add workflow templates directory and job tracker
    .add_local_dir("workflows", "/root/workflows", copy=True)
    .add_local_dir("lib", "/root/lib", copy=True)
    .add_local_file("job_tracker.py", "/root/job_tracker.py")
    .add_local_file("extra_model_paths.yaml", "/extra_model_paths.yaml")
)

# ## Model Setup Functions
def setup_model_paths_config():
    """Configure ComfyUI model paths by copying extra_model_paths.yaml config file."""
    import os
    import shutil
    
    print("🔧 Configuring ComfyUI model paths...")
    
    # Copy extra_model_paths.yaml from project root to ComfyUI directory
    # ComfyUI automatically loads extra_model_paths.yaml from its root directory
    source_config = "/extra_model_paths.yaml"
    target_config = "/root/comfy/ComfyUI/extra_model_paths.yaml"
    
    try:
        if os.path.exists(source_config):
            shutil.copy2(source_config, target_config)
            print(f"✅ Copied model paths config: {source_config} -> {target_config}")
        else:
            print(f"⚠️ Model paths config not found: {source_config}")
            return False
    except Exception as e:
        print(f"❌ Failed to copy model paths config: {str(e)}")
        return False
    
    # Models will be discovered by ComfyUI during initialization
    print("✅ ComfyUI configured to use /models volume via extra_model_paths.yaml")
    
    return True


# Shared runtime launcher so both Fast and Slow classes stay DRY
def _launch_inference_runtime(port: int) -> None:
    print("🔄 Initializing ComfyUI production environment...")
    
    # Send global status: container is starting up (cold start)
    try:
        from lib.ws_preview_relay import send_global_job_status
        # Note: We don't have job_id here, so this will be sent for the first job that connects
        print("📤 Container starting up - will send 'initializing' status for first job")
    except Exception as e:
        print(f"⚠️ Failed to import global status function: {e}")
    
    if not setup_model_paths_config():
        raise RuntimeError("Failed to configure model paths")
    os.makedirs("/root/comfy/ComfyUI/models/loras", exist_ok=True)
    
    # LoRA models directory will be created and populated on-demand per job
    
    # Set up environment variables for performance optimization
    env = os.environ.copy()
    env.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,backend:cudaMallocAsync',
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
        'NVIDIA_TF32_OVERRIDE': '1',
        'SAGE_ATTENTION_BACKEND': 'triton',
        'COMFYUI_MODEL_DEVICE': 'cuda',
        'COMFYUI_VAE_DEVICE': 'cuda', 
        'COMFYUI_CLIP_DEVICE': 'cuda',
        'COMFYUI_LOWVRAM': 'false',
        'COMFYUI_NOVRAM': 'false'
    })
    
    # Launch ComfyUI with optimized settings and preview support
    # Use default ComfyUI output directory, job isolation handled via filename_prefix
    cmd = (
        f"comfy launch --background -- --port {port} --use-sage-attention --gpu-only "
        f"--bf16-unet --bf16-vae --output-directory /root/comfy/ComfyUI/output --preview-method auto"
    )
    subprocess.run(cmd, shell=True, check=True, env=env)
    print("✅ ComfyUI server running with SageAttention and performance optimizations")

    # ComfyUI is now ready for direct API calls
    # Wait for ComfyUI to be ready
    print("🔄 Waiting for ComfyUI to be ready...")
    try:
        import urllib.request as _rq
        import time
        
        for _ in range(30):  # 30 second timeout
            try:
                _rq.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3)
                print("✅ ComfyUI is ready and responding")
                break
            except Exception as e:
                print(f"⏳ ComfyUI not ready yet: {e}")
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI failed to become ready within 30 seconds")
    except Exception as e:
        print(f"❌ Failed to verify ComfyUI readiness: {e}")


def poll_server_health(port: int) -> None:
    import socket, urllib.request, urllib.error
    
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/system_stats")
        urllib.request.urlopen(req, timeout=5)
        print("✅ ComfyUI server is healthy")
    except (socket.timeout, urllib.error.URLError) as e:
        print(f"❌ Server health check failed: {str(e)}")
        modal.experimental.stop_fetching_inputs()
        raise Exception("ComfyUI server is not healthy, stopping container")
    

def process_and_save_single_image(img_info, image_index, job_id, user_id, bucket, progress_ws_url, failure_tracker=None):
    """
    Process and save a single generated image to S3, then send WebSocket notification.
    This runs asynchronously to not block the next generation.
    Returns True if successful, False if failed.
    """
    print(f"🔄 Processing image {image_index + 1} for job {job_id}")
    try:
        import os
        import shutil
        import json
        import time
        import urllib.request
        from PIL import Image
        
        filename = img_info.get("filename")
        subfolder = img_info.get("subfolder", "")
        
        if not filename:
            if failure_tracker:
                failure_tracker['failed_images'].add(image_index)
                failure_tracker['errors'].append(f"Image {image_index + 1}: No filename provided")
            return False
            
        # Skip temporary preview files
        if "temp_" in filename.lower() or filename.startswith("ComfyUI_temp"):
            return True  # Not a failure, just skipping
            
        # Construct full path to image file from ComfyUI's output directory
        # ComfyUI saves to: /root/comfy/ComfyUI/output/{job_id}/IMG-{counter}_{timestamp}.png
        if subfolder:
            image_path = f"/root/comfy/ComfyUI/output/{subfolder}/{filename}"
        else:
            image_path = f"/root/comfy/ComfyUI/output/{filename}"
            
        if not os.path.exists(image_path):
            if failure_tracker:
                failure_tracker['failed_images'].add(image_index)
                failure_tracker['errors'].append(f"Image {image_index + 1}: File not found at {image_path}")
            return False
            

        
        # Extract base name and extension from the image file
        base_name = os.path.splitext(os.path.basename(filename))[0]  # e.g., "IMG-_00001_"
        original_ext = os.path.splitext(image_path)[1]  # .png, .jpg, etc.
        
        # Clean up the base name to get proper IMG-XX format
        # ComfyUI generates: "IMG-_00001_" -> we want "IMG-01"
        if base_name.startswith("IMG-_") and base_name.endswith("_"):
            # Extract the number: "IMG-_00001_" -> "00001"
            number_str = base_name[5:-1]  # Remove "IMG-_" and trailing "_"
            try:
                # Convert to int and back to get clean number: "00001" -> 1 -> "01"
                image_num = int(number_str)
                base_name = f"IMG-{image_num:02d}"  # Format as IMG-01, IMG-02, etc.
            except ValueError:
                # Fallback: use image_index if parsing fails
                base_name = f"IMG-{image_index + 1:02d}"
        else:
            # Fallback: use image_index for any other format
            base_name = f"IMG-{image_index + 1:02d}"
        
        print(f"🖼️ Processing ComfyUI image: {image_path}")
        print(f"🔍 Cleaned base name: {base_name}")
        
        # Copy original to job-specific S3 directory
        orig_dir = f"/data/{user_id}/inference/{job_id}/orig"
        os.makedirs(orig_dir, exist_ok=True)
        
        orig_filename = f"{base_name}{original_ext}"
        orig_path = f"{orig_dir}/{orig_filename}"
        shutil.copy(image_path, orig_path)
        
        print(f"  ✅ Copied original to S3: {orig_path}")
        
        # Create web versions directly on mounted S3 filesystem
        web_dir = f"/data/{user_id}/inference/{job_id}/web"
        os.makedirs(web_dir, exist_ok=True)
        
        final_image_url = None
        
        with Image.open(image_path) as img:
            # Convert to RGB if needed (for WebP compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create 480px, 720px, and 1024px versions (480px for retina 240px displays)
            for size, size_name in [(480, '480'), (720, '720'), (1024, '1024')]:
                # Create a copy for resizing
                web_img = img.copy()
                web_img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Determine file path on mounted filesystem
                if size == 1024:
                    # For 1024px, use base name without suffix (expected by frontend)
                    web_filename = f"{base_name}.webp"
                else:
                    # For smaller sizes, use -w{size} suffix (expected by frontend)
                    web_filename = f"{base_name}-w{size}.webp"
                
                web_path = f"{web_dir}/{web_filename}"
                
                # Save directly to mounted S3 filesystem with retries for reliability
                save_ok = False
                for attempt in range(3):
                    try:
                        web_img.save(web_path, format='WEBP', quality=max(65, 85 - attempt * 10), method=6, optimize=True)
                        if os.path.exists(web_path) and os.path.getsize(web_path) > 0:
                            save_ok = True
                            break
                    except Exception as _s_e:
                        pass  # Will retry
                    # brief delay before retry
                    import time as _t
                    _t.sleep(0.1 * (attempt + 1))
                if not save_ok:
                    if failure_tracker:
                        failure_tracker['failed_images'].add(image_index)
                        failure_tracker['errors'].append(f"Image {image_index + 1}: Failed to save {size}px WebP to S3")
                    continue
                
                # Store the 1024px version URL for WebSocket notification
                if size == 1024:
                    # Construct the S3 URL that the frontend expects
                    final_image_url = f"user-images/{user_id}/inference/{job_id}/web/{web_filename}"
        

        
        # Send WebSocket notification with final image URL through existing broadcast connection
        if final_image_url:
            try:
                # Import the WebSocket relay functions
                from lib.ws_preview_relay import send_custom_message_to_job
                
                final_image_data = {
                    "job_id": job_id,
                    "job_type": "inference",
                    "image_index": image_index,
                    "final_image_url": final_image_url,
                    "status": "image_completed",
                    "timestamp": int(time.time() * 1000),
                    "message": f"Image {image_index + 1} completed and saved"
                }
                
                print(f"🔍 Sending final image notification via existing broadcast connection:")
                print(f"🔍 Data: {final_image_data}")
                
                # Send through the existing WebSocket broadcast connection
                send_custom_message_to_job(job_id, final_image_data)
                print(f"📤 Sent final image WebSocket notification for image {image_index + 1}: {final_image_url}")
            except Exception as ws_e:
                print(f"⚠️ Failed to send final image WebSocket notification for image {image_index + 1}: {ws_e}")
                import traceback
                print(f"📊 WebSocket error: {traceback.format_exc()}")
        
        # Save image to database via Edge Function (retry if needed)
        try:
            import requests
            import os
            
            # Get Supabase credentials
            supabase_url = os.environ.get('SUPABASE_URL_DEV') or os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY_DEV') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
            if supabase_url and service_role_key and final_image_url:
                # Call inference-save-image Edge Function
                ef_url = f"{supabase_url}/functions/v1/inference-save-image"
                ef_headers = {
                    'Authorization': f'Bearer {service_role_key}',
                    'Content-Type': 'application/json',
                    'apikey': service_role_key,
                }
                
                ef_body = {
                    'job_id': job_id,
                    'image_index': image_index,
                    'user_id': user_id,
                    'original_path': f"user-images/{user_id}/inference/{job_id}/orig/{base_name}{original_ext}",
                    'web_path': final_image_url,
                    'width': 1024,  # We know this from our processing
                    'height': 1024,  # We know this from our processing
                    'format': 'webp',
                    'bytes': 0  # We could calculate this but it's not critical
                }
                
                ef_ok = False
                for attempt in range(3):
                    try:
                        ef_resp = requests.post(ef_url, json=ef_body, headers=ef_headers, timeout=10)
                        if ef_resp.ok:
                            ef_ok = True

                            break
                        else:
                            pass  # Will retry
                    except Exception as _ef_e:
                        pass  # Will retry
                    import time as _t
                    _t.sleep(0.2 * (attempt + 1))
                if not ef_ok:
                    if failure_tracker:
                        failure_tracker['failed_images'].add(image_index)
                        failure_tracker['errors'].append(f"Image {image_index + 1}: Failed to save to database after retries")
                
        except Exception as ef_e:
            if failure_tracker:
                failure_tracker['failed_images'].add(image_index)
                failure_tracker['errors'].append(f"Image {image_index + 1}: Edge Function error: {str(ef_e)}")
        
        # If we got here and have a final_image_url, consider it successful
        if final_image_url:
            return True
        else:
            if failure_tracker:
                failure_tracker['failed_images'].add(image_index)
                failure_tracker['errors'].append(f"Image {image_index + 1}: No final image URL generated")
            return False
                
    except Exception as e:
        if failure_tracker:
            failure_tracker['failed_images'].add(image_index)
            failure_tracker['errors'].append(f"Image {image_index + 1}: Processing error: {str(e)}")
        return False


def main(input_data: Dict[str, Any]) -> Dict[str, Any]:
    import sys; sys.path.append("/root")
    
    # Initialize variables that are used in exception handlers
    created_lora_filenames: list[str] = []
    
    # Choose Supabase creds based on env flag in input_data (same pattern as training)
    env_tag = (input_data.get("env") or "dev").lower()
    if env_tag not in {"dev", "prod"}:
        env_tag = "prod"

    # Override generic names so the rest of the code picks them up
    os.environ["SUPABASE_URL"] = os.getenv(f"SUPABASE_URL_{env_tag.upper()}") or os.environ.get("SUPABASE_URL", "")
    # Also switch service role key if provided
    env_service_key = os.getenv(f"SUPABASE_SERVICE_ROLE_KEY_{env_tag.upper()}")
    if env_service_key:
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = env_service_key
    
    from job_tracker import get_job_tracker
    tracker = get_job_tracker()
    job_id = input_data.get("job_id") or str(uuid.uuid4())
    user_id = input_data.get("user_id", "unknown")
    
    try:
        tracker.create_job(input_data, job_id)
        tracker.mark_processing(job_id)
        
        print(f"🎯 Starting generation job: {job_id}")
        
        # Detect if this is a cold start (first job on container)
        container_start_time = getattr(main, '_container_start_time', None)
        if container_start_time is None:
            # This is the first job on this container - mark as cold start
            main._container_start_time = time.time()
            is_cold_start = True
        else:
            # Container has been running, this is a warm start
            is_cold_start = False
        
        # Send initial "starting" status for all jobs
        try:
            from lib.ws_preview_relay import send_global_job_status
            send_global_job_status(job_id, "starting", "Starting up")
        except Exception as e:
            pass  # Non-critical
        
        # Update job status to 'running' via inference-start EF
        try:
            import urllib.request as _rq
            import json
            
            supabase_url = os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
            if supabase_url and service_role_key:
                start_url = f"{supabase_url}/functions/v1/inference-start"
                start_body = {"job_id": job_id}
                start_req = _rq.Request(
                    url=start_url,
                    data=json.dumps(start_body).encode('utf-8'),
                    headers={
                        'Authorization': f'Bearer {service_role_key}',
                        'Content-Type': 'application/json',
                        'apikey': service_role_key,
                    },
                    method='POST'
                )
                _rq.urlopen(start_req, timeout=10)

            else:
                print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY; skipping status update")
        except Exception as e:
            print(f"⚠️ Failed to update job status to running: {e}")
        
        poll_server_health(PORT)

        # Expect prepared payload from EF
        prepared = input_data.get("prepared")
        

        
        if not prepared:
            raise RuntimeError("Missing 'prepared' payload from inference-create EF")

        # Load workflow JSON (prefer mounted /workflows). Default to WAN2.1.json when not provided
        from lib.workflow_loader import load_workflow_from_s3
        wf_key = prepared.get("workflow") or "WAN2.1.json"
        wf = load_workflow_from_s3(wf_key)
        
        from lib.workflow_patcher import compute_dimensions, patch_workflow
        p = input_data.get("params", {})
        width, height = compute_dimensions(p.get("quality", "1K"), p.get("aspect_ratio", "1:1"))
        
        # Character/style LoRAs are absolute paths under /data from EF; link into models/loras
        char_lora = prepared.get("character_lora")
        style_lora = prepared.get("style_lora")
        
        # Track unique filenames we create so we can clean them up after the job finishes
        # (created_lora_filenames is initialized at function start)

        def _link_lora(abs_path: str | None) -> str | None:
            if not abs_path:
                return None
            try:
                from pathlib import Path as _P
                import shutil as _sh
                p = _P(abs_path)
                
                if not p.exists():
                    return None
        
                dest_dir = _P("/root/comfy/ComfyUI/models/loras")
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Use a job-scoped unique filename to avoid cross-job collisions when multiple
                # jobs link LoRAs with the same basename concurrently
                unique_name = f"{job_id}__{p.name}"
                dest = dest_dir / unique_name
        
                try:
                    if dest.exists() or dest.is_symlink():
                        dest.unlink()
                except Exception:
                    pass
                try:
                    dest.symlink_to(p)
                except Exception as e:
                    _sh.copy2(str(p), str(dest))
                
                created_lora_filenames.append(unique_name)
                return unique_name
        
            except Exception as _e:
                return None
        

        
        char_name = _link_lora(char_lora)
        style_name = _link_lora(style_lora)
        


        # Handle seed generation - if seed is -1 or None, generate a random seed
        seed_value = p.get("seed")
        if seed_value is None or seed_value == -1:
            import random
            seed_value = random.randint(0, 2**32 - 1)
        
        # Get number of images to generate
        nb_takes = int(p.get("nb_takes", 1))

        # 3) Generate client ID for ComfyUI API
        comfyui_base = f"http://127.0.0.1:{PORT}"
        
        # 4) Start direct ComfyUI WebSocket monitoring for progress (non-blocking)
        try:
            import threading
            from lib.ws_preview_relay import start_direct_comfyui_relay
        
            base = os.environ.get("PROGRESS_WS_URL") or "wss://creativebuild--primeshot-inference-progress.modal.run"
        
            if base:
                progress_ws_url = base.replace("{job_id}", job_id) if "{job_id}" in base else base.rstrip("/") + f"/ws/broadcast/{job_id}"
                

        
        except Exception as e:
            print(f"⚠️ Progress relay setup failed: {e}")
            import traceback
            print(f"📊 Full error: {traceback.format_exc()}")
            base = None
            progress_ws_url = None

        # 5) Sequential image generation loop
        all_generated_images = []
        
        # Track S3 processing failures and threads
        s3_failure_tracker = {
            'failed_images': set(),
            'errors': []
        }
        s3_processing_threads = []
        import urllib.request, urllib.error
        
        # Get AWS bucket early since it's needed in job metadata
        bucket = os.environ.get("AWS_BUCKET")
        
        # Store enhanced job metadata for webhook retrieval and debugging
        job_metadata = {
            "user_id": user_id,
            "character_id": input_data.get("character_id"),
            "style_id": input_data.get("style_id"),
            "workflow_key": wf_key,
            "dimensions": f"{width}x{height}",
            "nb_takes": nb_takes,
            "quality": p.get("quality", "1K"),
            "aspect_ratio": p.get("aspect_ratio", "1:1"),
            "seed": seed_value,
            "character_lora": char_name,
            "style_lora": style_name,
            "comfyui_base": comfyui_base,
            "s3_configured": bucket is not None,
            "started_at": time.time(),
            "gpu_type": input_data.get("gpu_type", "unknown"),
            "env": env_tag
        }
        
        # Store in a simple in-memory cache
        if not hasattr(tracker, '_job_metadata'):
            tracker._job_metadata = {}
        tracker._job_metadata[job_id] = job_metadata
        
        print(f"🔍 Job metadata: {job_metadata}")
        
        # Start ONE WebSocket relay for the ENTIRE job (not per image)
        if progress_ws_url:
            try:
                import threading
                from lib.ws_preview_relay import start_relay
                
                # Use a single client ID for the entire job
                job_client_id = str(uuid.uuid4())
                comfy_ws_url = f"ws://127.0.0.1:{PORT}/ws?clientId={job_client_id}"
                
                print(f"🔌 WebSocket Configuration for job {job_id}:")
                print(f"  📤 Progress Broadcast: {progress_ws_url}")
                print(f"  🌐 Base URL: {progress_ws_url.split('/ws/broadcast/')[0]}")
                
                # Start the relay for the entire job
                start_relay(progress_ws_url, comfy_ws_url, job_id, throttle_sec=1.5, image_index=0)
                print(f"🔄 Started WebSocket relay for entire job {job_id}")
            except Exception as e:
                print(f"⚠️ WebSocket relay failed for job {job_id}: {e}")
        
        # Sequential generation loop
        for image_index in range(nb_takes):
            print(f"🎯 === GENERATING IMAGE {image_index + 1}/{nb_takes} ===")
            
            # Update the WebSocket relay with the current image index
            if progress_ws_url:
                try:
                    from lib.ws_preview_relay import update_job_image_index
                    update_job_image_index(job_id, image_index)
                except Exception as e:
                    print(f"⚠️ Failed to update image index: {e}")
            
            # Send global status when starting first image generation
            # Status will be automatically switched to "generating" when first base64 image is received
            
            # Generate unique seed for each image (if original seed was random)
            current_seed = seed_value + image_index if seed_value != -1 else None
            if current_seed is None:
                import random
                current_seed = random.randint(0, 2**32 - 1)
            
            print(f"🎲 Using seed {current_seed} for image {image_index + 1}")
            
            # Send progress update for this image
            if progress_ws_url:
                try:
                    import urllib.request as _rq
                    import json
                    
                    progress_data = {
                        "job_id": job_id,
                        "status": "running",
                        "progress": int((image_index / nb_takes) * 100),
                        "image_index": image_index,
                        "message": f"Generating image {image_index + 1} of {nb_takes}",
                        "timestamp": time.time()
                    }
                    
                    # Send to HTTP progress endpoint (fallback). Convert scheme and path.
                    if "/ws/broadcast/" in progress_ws_url:
                        post_url = (
                            progress_ws_url
                            .replace("wss://", "https://")
                            .replace("ws://", "http://")
                            .replace("/ws/broadcast/", "/api/progress/")
                        )
                        progress_body = json.dumps(progress_data).encode('utf-8')
                        progress_req = _rq.Request(
                            url=post_url,
                            data=progress_body,
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        )
                        _rq.urlopen(progress_req, timeout=5)
                    print(f"📊 Sent progress update: image {image_index + 1}/{nb_takes}")
                except Exception as e:
                    print(f"⚠️ Failed to send progress update: {e}")
            
            # Patch workflow for single image with current seed
            patched = patch_workflow(
                wf,
                prompt=prepared.get("prompt", ""),
                negative_prompt=prepared.get("negative_prompt", ""),
                width=width,
                height=height,
                seed=current_seed,
                images_count=1,  # Always generate 1 image per call
                quality=p.get("quality", "1K"),
                lora_filename=None,
                character_lora=char_name,
                style_lora=style_name,
                bypass_nodes=prepared.get("bypass_nodes"),
                job_id=job_id,  # Pass job_id for output path isolation
                user_id=user_id,  # Pass user_id for output path isolation
            )
            
            # Use the same client ID for all images in this job
            client_id = job_client_id
            
            # Submit to ComfyUI
            body = {
                "prompt": patched,
                "client_id": client_id
            }
            
            print(f"🔧 Submitting image {image_index + 1} to ComfyUI...")
            
            # Single readiness check
            try:
                urllib.request.urlopen(f"{comfyui_base}/system_stats", timeout=3.0)
                print(f"✅ ComfyUI ready for image {image_index + 1}")
            except Exception as e:
                print(f"⚠️ ComfyUI readiness check failed for image {image_index + 1}: {e}")

            # Submit to ComfyUI for this specific image
            prompt_url = f"{comfyui_base}/prompt"
            
            try:
                req = urllib.request.Request(
                    url=prompt_url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                response = urllib.request.urlopen(req, timeout=30)
                response_data = response.read().decode('utf-8')
                response_json = json.loads(response_data)
                
                print(f"📨 Submitted image {image_index + 1} to ComfyUI: {response_json}")
                
                # Extract prompt_id from response
                prompt_id = response_json.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"No prompt_id in ComfyUI response for image {image_index + 1}: {response_json}")
                
                print(f"🎯 Image {image_index + 1} submitted! Prompt ID: {prompt_id}")
                
                # Wait for this specific image to complete using WebSocket events
                print(f"⏳ Waiting for image {image_index + 1} to complete...")
                
                # Use event-based completion instead of polling
                from lib.ws_preview_relay import wait_for_prompt_completion, get_prompt_completion_data
                completed = wait_for_prompt_completion(job_id, prompt_id, timeout=600)
                
                if completed:
                    print(f"✅ Image {image_index + 1} completed!")
                    
                    # Get the completion data directly from WebSocket (no HTTP request needed!)
                    completion_data = get_prompt_completion_data(job_id, prompt_id)
                    
                    if completion_data and completion_data.get("outputs"):
                        # Process the completed image using WebSocket data
                        outputs = completion_data.get("outputs", {})
                        print(f"🔍 DEBUG: Using WebSocket completion data for image {image_index + 1}")
                    else:
                        # Fallback to history endpoint for outputs (even if we have completion confirmation)
                        completion_source = completion_data.get("completion_source", "unknown") if completion_data else "none"
                        print(f"⚠️ WebSocket completion data has no outputs (source: {completion_source}), falling back to history endpoint")
                        
                        try:
                            history_url = f"{comfyui_base}/history/{prompt_id}"
                            history_response = urllib.request.urlopen(history_url, timeout=10)
                            history_data = json.loads(history_response.read().decode('utf-8'))
                            
                            if prompt_id in history_data:
                                image_data = history_data[prompt_id]
                                outputs = image_data.get("outputs", {})
                                print(f"🔍 DEBUG: Using fallback history data for image {image_index + 1}")
                            else:
                                print(f"⚠️ Prompt {prompt_id} not found in history for image {image_index + 1}")
                                print(f"🔍 DEBUG: Available history keys: {list(history_data.keys()) if isinstance(history_data, dict) else 'Not a dict'}")
                                
                                # If we have completion confirmation but no history, the image might still be processing
                                # Let's wait a bit and try again
                                if completion_data:
                                    print(f"🔄 Completion detected but no history yet, waiting 2s and retrying...")
                                    time.sleep(2)
                                    
                                    # Retry once
                                    history_response = urllib.request.urlopen(history_url, timeout=10)
                                    history_data = json.loads(history_response.read().decode('utf-8'))
                                    
                                    if prompt_id in history_data:
                                        image_data = history_data[prompt_id]
                                        outputs = image_data.get("outputs", {})
                                        print(f"🔍 DEBUG: Using retry history data for image {image_index + 1}")
                                    else:
                                        print(f"⚠️ Prompt {prompt_id} still not found in history after retry")
                                        raise RuntimeError(f"Prompt {prompt_id} not found in ComfyUI history after retry")
                                else:
                                    raise RuntimeError(f"Prompt {prompt_id} not found in ComfyUI history")
                        except Exception as e:
                            print(f"⚠️ Error getting fallback history data for image {image_index + 1}: {e}")
                            raise RuntimeError(f"Failed to get completion data for image {image_index + 1}: {e}")
                    
                    print(f"🔍 DEBUG: ComfyUI outputs for image {image_index + 1}:")
                    print(f"🔍 DEBUG: Available output nodes: {list(outputs.keys())}")
                    for node_id, node_output in outputs.items():
                        print(f"🔍 DEBUG: Node {node_id} output keys: {list(node_output.keys())}")
                        if "images" in node_output:
                            print(f"🔍 DEBUG: Node {node_id} images: {node_output['images']}")
                    
                    # Find generated images in outputs
                    generated_images = []
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img in node_output["images"]:
                                generated_images.append({
                                    "filename": img.get("filename"),
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output"),
                                    "image_index": image_index
                                })
                    
                    print(f"🖼️ Found {len(generated_images)} images for image {image_index + 1}")
                    if generated_images:
                        print(f"🔍 DEBUG: First image details: {generated_images[0]}")
                        
                        # DEBUG: Check if files exist in ComfyUI output directory
                        comfyui_output_dir = "/root/comfy/ComfyUI/output"
                        print(f"🔍 DEBUG: Checking ComfyUI output directory: {comfyui_output_dir}")
                        try:
                            if os.path.exists(comfyui_output_dir):
                                files_in_dir = os.listdir(comfyui_output_dir)
                                print(f"🔍 DEBUG: Files in ComfyUI output: {files_in_dir}")
                                
                                # Check job-specific subdirectory
                                job_subdir = f"{comfyui_output_dir}/{job_id}"
                                if os.path.exists(job_subdir):
                                    job_files = os.listdir(job_subdir)
                                    print(f"🔍 DEBUG: Files in job subdirectory {job_id}: {job_files}")
                                else:
                                    print(f"🔍 DEBUG: Job subdirectory {job_id} does not exist!")
                            else:
                                print(f"🔍 DEBUG: ComfyUI output directory does not exist!")
                        except Exception as e:
                            print(f"🔍 DEBUG: Error checking ComfyUI output directory: {e}")
                        
                        all_generated_images.extend(generated_images)
                        
                        # Immediately process and save this image to S3 (async)
                        if generated_images:
                            try:
                                import threading
                                print(f"🔍 About to start async S3 processing for image {image_index + 1}")
                                print(f"🔍 Generated image info: {generated_images[0]}")
                                print(f"🔍 Parameters: image_index={image_index}, job_id={job_id}, user_id={user_id}, bucket={bucket}")
                                
                                # Start async S3 processing for this image
                                thread = threading.Thread(
                                    target=process_and_save_single_image,
                                    args=(generated_images[0], image_index, job_id, user_id, bucket, progress_ws_url, s3_failure_tracker),
                                    daemon=True
                                )
                                thread.start()
                                s3_processing_threads.append(thread)
                            except Exception as save_e:
                                pass  # S3 processing will be tracked by failure_tracker
                            
                    else:
                        raise RuntimeError(f"No generated images found for image {image_index + 1}")
                else:
                    raise RuntimeError(f"Image {image_index + 1} generation timed out")
                    
            except Exception as e:
                raise RuntimeError(f"Image {image_index + 1} generation failed: {e}")
        
        # Wait for all S3 processing threads to complete and check for failures
        for thread in s3_processing_threads:
            thread.join(timeout=30)  # Wait up to 30 seconds per thread
        
        # Check if any images failed to save
        if s3_failure_tracker['failed_images']:
            failed_count = len(s3_failure_tracker['failed_images'])
            total_count = nb_takes
            error_summary = "; ".join(s3_failure_tracker['errors'][:3])  # First 3 errors
            if len(s3_failure_tracker['errors']) > 3:
                error_summary += f" (and {len(s3_failure_tracker['errors']) - 3} more errors)"
            
            raise RuntimeError(f"Failed to save {failed_count}/{total_count} images to S3: {error_summary}")
        
        # Images are now processed asynchronously during generation
        # We only need to call inference-complete Edge Function to mark the job as done
        if not all_generated_images:
            raise RuntimeError("No images were generated")
        

        
        # Call inference-complete Edge Function to mark the job as completed
        # (Individual images were already saved to S3 asynchronously during generation)
        try:
            env_tag = job_metadata.get('env', 'dev')
            supabase_url = os.environ.get(f'SUPABASE_URL_{env_tag.upper()}') or os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get(f'SUPABASE_SERVICE_ROLE_KEY_{env_tag.upper()}') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
            if supabase_url and service_role_key:
                import requests
                ef_url = f"{supabase_url}/functions/v1/inference-complete"
                ef_headers = {
                    'Authorization': f'Bearer {service_role_key}',
                    'Content-Type': 'application/json',
                    'apikey': service_role_key,
                }
                # Simple completion call - images were processed individually
                ef_body = {
                    'job_id': job_id,
                    'success': True
                }
                ef_resp = requests.post(ef_url, json=ef_body, headers=ef_headers, timeout=20)
                if ef_resp.ok:
                    print(f"✅ Called inference-complete Edge Function successfully")
                    print(f"🎯 Inference pipeline completed successfully!")
                else:
                    print(f"⚠️ inference-complete EF error: {ef_resp.status_code} {ef_resp.text}")
            else:
                print(f"⚠️ Missing Supabase credentials for {env_tag} environment")
                
        except Exception as ef_e:
            print(f"⚠️ Failed to call inference-complete Edge Function: {ef_e}")
            import traceback
            print(f"📊 Full error: {traceback.format_exc()}")
        
        print(f"✅ Successfully completed sequential generation job {job_id}")
        
        # Signal WebSocket relay that job is complete
        try:
            from lib.ws_preview_relay import signal_job_completion, get_active_relays
            
            print(f"📊 Active relays before completion: {get_active_relays()}")
            signal_job_completion(job_id)
            print(f"📊 Active relays after completion: {get_active_relays()}")
            print(f"✅ WebSocket relay cleanup completed for job {job_id}")
            
        except Exception as signal_e:
            print(f"⚠️ Failed to signal job completion: {signal_e}")
            # Try to force cleanup anyway
            try:
                from lib.ws_preview_relay import cleanup_all_relays
                cleanup_all_relays()
            except Exception as cleanup_e:
                print(f"⚠️ Failed to force cleanup relays: {cleanup_e}")
        
        # Clean up job-scoped LoRA links
        try:
            from pathlib import Path as _P
            for fname in created_lora_filenames:
                try:
                    f = _P(f"/root/comfy/ComfyUI/models/loras/{fname}")
                    if f.exists() or f.is_symlink():
                        f.unlink()
                        print(f"🧹 Removed job-scoped LoRA link: {fname}")
                except Exception as _e:
                    print(f"⚠️ Failed to remove LoRA link {fname}: {_e}")
        except Exception as _cleanup_e:
            print(f"⚠️ LoRA cleanup failed: {_cleanup_e}")
        
        # 5) Return accepted; completion goes via webhook -> EF -> DB
        return {"status": "accepted", "job_id": job_id}
    
    except Exception as e:
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "job_id": job_id,
            "user_id": user_id
        }
        
        # Categorize error types for better debugging
        if "connection refused" in str(e).lower():
            error_details["category"] = "connection_error"
            error_details["suggestion"] = "ComfyUI server may not be running or ready"
        elif "timeout" in str(e).lower():
            error_details["category"] = "timeout_error"
            error_details["suggestion"] = "Request timed out - server may be overloaded"
        elif "404" in str(e):
            error_details["category"] = "endpoint_error"
            error_details["suggestion"] = "ComfyUI endpoint not found - check server configuration"
        elif "prompt_id" in str(e).lower():
            error_details["category"] = "generation_error"
            error_details["suggestion"] = "Issue with ComfyUI prompt generation or processing"
        else:
            error_details["category"] = "general_error"
            error_details["suggestion"] = "Check logs for detailed error information"
        
        print(f"❌ Generation failed: {error_details}")
        
        try:
            tracker.mark_failed(job_id, str(e))
        except Exception as tracker_e:
            print(f"⚠️ Failed to update job tracker: {tracker_e}")
        
        # Send failure notification via WebSocket before cleanup
        try:
            from lib.ws_preview_relay import send_custom_message_to_job
            failure_message = {
                "job_id": job_id,
                "job_type": "inference",
                "status": "failed",
                "progress": 0,
                "message": f"Generation failed: {error_details.get('suggestion', str(e))}",
                "timestamp": int(time.time() * 1000),
                "error": str(e),
                "error_type": error_details.get("error_type", "Unknown")
            }
            send_custom_message_to_job(job_id, failure_message)
            print(f"📤 Sent failure notification via WebSocket for job {job_id}")
        except Exception as ws_fail_e:
            print(f"⚠️ Failed to send failure notification via WebSocket: {ws_fail_e}")
        
        # Clean up WebSocket relay on failure
        try:
            from lib.ws_preview_relay import signal_job_completion, get_active_relays
            print(f"📊 Active relays before failure cleanup: {get_active_relays()}")
            signal_job_completion(job_id)
            print(f"📊 Active relays after failure cleanup: {get_active_relays()}")
        except Exception as cleanup_e:
            print(f"⚠️ Failed to cleanup relay on failure: {cleanup_e}")
        
        # Attempt to remove job-scoped LoRA links on failure
        try:
            from pathlib import Path as _P
            for fname in created_lora_filenames:
                try:
                    f = _P(f"/root/comfy/ComfyUI/models/loras/{fname}")
                    if f.exists() or f.is_symlink():
                        f.unlink()
                        print(f"🧹 Removed job-scoped LoRA link after failure: {fname}")
                except Exception as _e:
                    print(f"⚠️ Failed to remove LoRA link {fname} after failure: {_e}")
        except Exception as _cleanup_fail_e:
            print(f"⚠️ LoRA cleanup on failure failed: {_cleanup_fail_e}")
    
        return {
            "job_id": job_id, 
            "status": "failed", 
            "error": str(e),
            "error_details": error_details
        }
    
    finally:
        pass


@app.cls(
    gpu="H100",
    image=cuda_image,
    secrets=[aws_secret, inference_secret, supabase_secret],
    volumes={**user_images_mount, **workflows_mount, MODELS_PATH: models_volume},
    timeout=30000,
    scaledown_window=300,  # 5 minute keep-alive (will be tuned later)
    max_containers=30,
    retries=3,
)
@modal.concurrent(max_inputs=1)
class Fast:
    """Production ComfyUI class for optimized batch image generation."""

    @modal.enter()
    def setup_environment(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        _launch_inference_runtime(PORT)

    @modal.method()
    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return main(input_data)


@app.cls(
    gpu="A10G",
    volumes={**user_images_mount, **workflows_mount, MODELS_PATH: models_volume},
    secrets=[aws_secret, inference_secret, supabase_secret],
    timeout=30000,
    scaledown_window=300,  # 5 minute keep-alive (will be tuned later)
    max_containers=10,
    retries=3,
)
@modal.concurrent(max_inputs=1)
class Slow:
    @modal.enter()
    def setup_environment(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        _launch_inference_runtime(PORT)

    @modal.method()
    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return main(input_data)
    

@app.function(
    image=modal.Image.debian_slim().pip_install([
        "fastapi[standard]==0.115.4",
        "websockets>=12.0",
        "uvicorn[standard]>=0.24.0",
        "python-dotenv>=1.0.0",
        "anthropic>=0.34.0"
    ]),
    secrets=[aws_secret, inference_secret, supabase_secret],
    scaledown_window=300,  # 300 seconds (5 minutes)
)
@modal.concurrent(max_inputs=100, target_inputs=80)
@modal.fastapi_endpoint(method="POST", label="primeshot-inference", requires_proxy_auth=True)
def api_endpoint(request_data: Dict[str, Any]):
    """Accept prepared inference request and spawn GPU worker.

    Contract: caller must provide `user_id`, `prepared`, and optional `params`.
    This endpoint does NOT call inference-create EF; it just enqueues work.
    """
    from fastapi import HTTPException

    user_id = request_data.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing required parameter: user_id")
    
    prepared = request_data.get("prepared")
    
    if not prepared:
        raise HTTPException(status_code=400, detail="Missing required parameter: prepared")

    job_id = request_data.get("job_id") or str(uuid.uuid4())
    params = request_data.get("params", {})

    print(f"Received prepared: {prepared}")

    # Spawn on available GPU class
    gpu_classes = [("H100", Fast), ("A10G", Slow)]
    
    for gpu_type, gpu_class in gpu_classes:
        try:
            gpu = gpu_class()
            payload = {
                "user_id": user_id,
                "job_id": job_id,
                "prepared": prepared,
                "params": params,
                "gpu_type": gpu_type,
            }
            handle = gpu.run_inference.spawn(payload)
    
            return {
                "status": "accepted",
                "gpu_type": gpu_type,
                "job_handle": str(handle),
                "job_id": job_id,
            }
    
        except Exception as e:
            print(f"Spawn failed for {gpu_type}: {e}. Trying next class...")
            continue

    raise HTTPException(status_code=503, detail="Submission failed for all GPU classes")


# Webhook endpoint removed - no longer using webhooks, only synchronous S3 mode


# WebSocket Progress Server for Inference
@app.function(
    image=modal.Image.debian_slim().pip_install([
        "fastapi[standard]==0.115.4",
        "websockets>=12.0",
        "uvicorn[standard]>=0.24.0",
        "python-dotenv>=1.0.0"
    ]),
    scaledown_window=300,
    timeout=720,
)
@modal.concurrent(max_inputs=600, target_inputs=480)
@modal.asgi_app()
def progress():
    """
    WebSocket Progress Server for real-time inference updates.
    Endpoints:
      - /ws/progress/{job_id}: clients subscribe to receive updates
      - /ws/broadcast/{job_id}: producers send updates to listeners
    """
    import asyncio
    import json
    import logging
    from typing import Dict, List
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    # Logging (quiet by default)
    log_level_name = os.getenv("INFERENCE_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, log_level_name, logging.WARNING)
    logger = logging.getLogger("inference_ws")
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    web_app = FastAPI(title="Primeshot Inference Progress Server")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    active_connections: Dict[str, List[WebSocket]] = {}
    connection_timestamps: Dict[str, float] = {}  # Track when connections were created

    class ConnectionManager:
        @staticmethod
        async def connect_listener(websocket: WebSocket, job_id: str) -> None:
            await websocket.accept()
            active_connections.setdefault(job_id, []).append(websocket)
            connection_timestamps[job_id] = time.time()  # Track connection time

        @staticmethod
        def disconnect(websocket: WebSocket) -> None:
            for job_id, sockets in list(active_connections.items()):
                if websocket in sockets:
                    sockets.remove(websocket)
                if not sockets:
                    active_connections.pop(job_id, None)
                    connection_timestamps.pop(job_id, None)  # Clean up timestamp

        @staticmethod
        async def broadcast(job_id: str, message: str) -> None:
            for ws in list(active_connections.get(job_id, [])):
                try:
                    await ws.send_text(message)
                except Exception:
                    ConnectionManager.disconnect(ws)
        
        @staticmethod
        def cleanup_stale_connections(max_age_minutes: int = 15) -> None:
            """Clean up connections older than max_age_minutes."""
            import time
            current_time = time.time()
            max_age_seconds = max_age_minutes * 60
            
            stale_jobs = []
            for job_id, timestamp in list(connection_timestamps.items()):
                if current_time - timestamp > max_age_seconds:
                    stale_jobs.append(job_id)
            
            for job_id in stale_jobs:
                logger.info(f"Cleaning up stale connection for job {job_id}")
                active_connections.pop(job_id, None)
                connection_timestamps.pop(job_id, None)
            
            if stale_jobs:
                logger.info(f"Cleaned up {len(stale_jobs)} stale connections")
        
        @staticmethod
        def get_connection_stats() -> dict:
            """Get statistics about active connections."""
            import time
            current_time = time.time()
            
            stats = {
                "total_jobs": len(active_connections),
                "total_connections": sum(len(sockets) for sockets in active_connections.values()),
                "jobs_by_age": {}
            }
            
            for job_id, timestamp in connection_timestamps.items():
                age_minutes = int((current_time - timestamp) / 60)
                age_bucket = f"{age_minutes//5 * 5}-{age_minutes//5 * 5 + 4}min"
                stats["jobs_by_age"][age_bucket] = stats["jobs_by_age"].get(age_bucket, 0) + 1
            
            return stats

    manager = ConnectionManager()

    # Periodic cleanup task
    async def periodic_cleanup():
        """Periodically clean up stale connections."""
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            try:
                manager.cleanup_stale_connections(max_age_minutes=15)
                stats = manager.get_connection_stats()
                logger.info(f"WebSocket stats: {stats}")
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    # Store background tasks for proper cleanup
    background_tasks = set()
    
    # Startup event to initialize background tasks
    @web_app.on_event("startup")
    async def startup_event():
        """Initialize background tasks when the app starts"""
        task = asyncio.create_task(periodic_cleanup())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        logger.info("Started periodic cleanup task")
    
    # Shutdown event to clean up background tasks
    @web_app.on_event("shutdown")
    async def shutdown_event():
        """Clean up background tasks when the app shuts down"""
        logger.info("Shutting down background tasks...")
        for task in background_tasks:
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)
        logger.info("Background tasks cleaned up")

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint for the progress server."""
        return {"status": "healthy", "service": "primeshot-inference-progress"}

    @web_app.get("/api/health")
    async def api_health_check():
        """API health check endpoint (matches training app pattern)."""
        return {"status": "healthy", "service": "primeshot-inference-progress", "version": "1.0.0"}

    @web_app.get("/stats")
    async def get_stats():
        """Get WebSocket connection statistics."""
        return manager.get_connection_stats()

    from fastapi import Request
    @web_app.post("/api/progress/{job_id}")
    async def http_progress(job_id: str, request: Request):
        """HTTP fallback: accept JSON progress and rebroadcast to WS listeners."""
        try:
            payload_bytes = await request.body()
            # Validate minimally by ensuring it's JSON
            import json as _j
            _j.loads(payload_bytes)
            payload_text = payload_bytes.decode("utf-8")
            await manager.broadcast(job_id, payload_text)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @web_app.websocket("/ws/progress/{job_id}")
    async def websocket_progress_endpoint(websocket: WebSocket, job_id: str):
        await manager.connect_listener(websocket, job_id)
        try:
            # Keep the connection open; listeners don't send messages
            while True:
                await asyncio.sleep(60)
    
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @web_app.websocket("/ws/broadcast/{job_id}")
    async def websocket_broadcast_endpoint(websocket: WebSocket, job_id: str):
        await websocket.accept()
    
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    # Ensure JSON payload
                    json.loads(data)
                    await manager.broadcast(job_id, data)
                except Exception as e:
                    logger.error(f"Invalid broadcast payload: {e}")
    
        except WebSocketDisconnect:
            pass

    return web_app



@app.function(
    gpu="H100",  # Cost-effective for UI development A10G
    image=cuda_image,
    volumes={**user_images_mount, **workflows_mount, MODELS_PATH: models_volume},
    max_containers=1,
    timeout=1300
)
@modal.concurrent(max_inputs=1)
@modal.web_server(8000, startup_timeout=120)
def dev_server():
    """Interactive ComfyUI development server for workflow creation."""
    print("🚀 Starting ComfyUI development server...")
    
    # Configure ComfyUI with extra_model_paths.yaml
    if not setup_model_paths_config():
        print("❌ Failed to configure model paths")
        return
    
    # Create LoRA directory for symlinks
    lora_models_dir = "/root/comfy/ComfyUI/models/loras"
    os.makedirs(lora_models_dir, exist_ok=True)
    
    # Link all LoRAs from S3 bucket for dev server
    def link_all_loras():
        """Link all LoRAs from S3 bucket to make them available in dev server."""
        s3_loras_dir = "/data"
        linked_count = 0
        
        if not os.path.exists(s3_loras_dir):
            print(f"⚠️ S3 bucket not found at {s3_loras_dir}")
            return 0
        
        print(f"🔍 Scanning S3 bucket for LoRAs: {s3_loras_dir}")
        
        # Walk through all directories in S3 bucket
        for root, dirs, files in os.walk(s3_loras_dir):
            for file in files:
                # Link .safetensors files (LoRA format)
                if file.endswith('.safetensors'):
                    source_path = os.path.join(root, file)
                    relative_path = os.path.relpath(source_path, s3_loras_dir)
                    
                    # Create unique filename to avoid conflicts
                    safe_filename = relative_path.replace('/', '_').replace('\\', '_')
                    target_path = os.path.join(lora_models_dir, safe_filename)
                    
                    try:
                        # Remove existing link if it exists
                        if os.path.exists(target_path) or os.path.islink(target_path):
                            os.unlink(target_path)
                        
                        # Create symlink
                        os.symlink(source_path, target_path)
                        linked_count += 1
                        print(f"🔗 Linked LoRA: {safe_filename}")
    
                    except Exception as e:
                        print(f"⚠️ Failed to link {file}: {str(e)}")
        
        return linked_count
    
    # Link all LoRAs for dev testing
    linked_loras = link_all_loras()
    print(f"✅ Linked {linked_loras} LoRAs from S3 bucket for dev server")
    
    # Set comprehensive environment variables for English locale and performance optimization
    env = os.environ.copy()
    env.update({
        # Performance optimization environment variables
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,backend:cudaMallocAsync',
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
        'NVIDIA_TF32_OVERRIDE': '1',
        'SAGE_ATTENTION_BACKEND': 'triton',
        # H100 Performance: Force models to stay on GPU
        'COMFYUI_MODEL_DEVICE': 'cuda',
        'COMFYUI_VAE_DEVICE': 'cuda', 
        'COMFYUI_CLIP_DEVICE': 'cuda',
        'COMFYUI_LOWVRAM': 'false',
        'COMFYUI_NOVRAM': 'false'
    })
    
    # Launch ComfyUI UI server with SageAttention, H100 performance optimizations, and preview support
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000 --use-sage-attention --gpu-only --bf16-unet --bf16-vae --output-directory /data/outputs --preview-method auto",
        shell=True,
        env=env
    )
    print("🌐 ComfyUI UI available at the development server URL with English locale")