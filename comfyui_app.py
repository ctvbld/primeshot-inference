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
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Tuple
import modal
import modal.experimental

sys.path.insert(0, "/root/lib")

app = modal.App(name="primeshot-inference")

# Create Modal volumes for persistent storage
models_volume = modal.Volume.from_name("models-vol", create_if_missing=True)
aws_secret = modal.Secret.from_name("aws-secret")
inference_secret = modal.Secret.from_name("inference-secret")
supabase_secret = modal.Secret.from_name("supabase-secret")

# Central GPU names (hardcoded routing)
DEFAULT_GPU_TYPE = "H200"

# Define paths
MODELS_PATH = "/models"
PORT: int = 8000
# GPU type for dev_server UI (H100 queues can be long). Override via DEV_SERVER_GPU.
DEV_SERVER_GPU_TYPE = os.getenv("DEV_SERVER_GPU", "H100")

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
       "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/yolain/ComfyUI-Easy-Use",
       "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Easy-Use && pip install -r requirements.txt --no-input"
    )
    # START TO UNUSED NODES
    # .run_commands(
    #     "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes",
    #     "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes && pip install -r requirements.txt --no-input"
    # )
    # .run_commands(
    #     "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/vrgamegirl19/comfyui-vrgamedevgirl",
    #     "cd /root/comfy/ComfyUI/custom_nodes/comfyui-vrgamedevgirl && pip install -r requirements.txt --no-input"
    # )
    # END TO UNUSED NODES
    # Install RES4LYF advanced sampling nodes
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/ClownsharkBatwing/RES4LYF.git",
        "cd /root/comfy/ComfyUI/custom_nodes/RES4LYF && pip install -r requirements.txt --no-input 2>/dev/null || echo 'No requirements.txt found for RES4LYF'"
    )
    # Install post-processing nodes
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/orion4d/ComfyUI-Image-Effects.git", 
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Image-Effects && pip install -r requirements.txt --no-input"
    )
    # NAG restores effective negative prompting in few-step diffusion models, and complements CFG in multi-step sampling for improved quality and control.
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/ChenDarYen/ComfyUI-NAG"
    )
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/Goktug/comfyui-saveimage-plus.git"
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
        import time
        
        for _ in range(30):  # 30 second timeout
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3)
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
    

def find_generated_images(
    job_id: str,
    output_dir: str = "/root/comfy/ComfyUI/output",
    max_attempts: int = 10,
    delay_seconds: float = 1.0
) -> list[dict]:
    """Find generated images in ComfyUI output directory with retries.
    
    Returns list of image info dicts with filename, subfolder, type fields.
    """
    import glob
    
    found_images = []
    job_output_dir = os.path.join(output_dir, job_id)
    
    for attempt in range(max_attempts):
        # Check job-specific directory first
        if os.path.exists(job_output_dir):
            files = [f for f in os.listdir(job_output_dir) 
                    if f.endswith(('.png', '.webp', '.jpg', '.jpeg'))]
            if files:
                print(f"✅ Found {len(files)} files in job directory after {attempt + 1} attempts")
                for filename in files:
                    found_images.append({
                        "filename": filename,
                        "subfolder": job_id,
                        "type": "output"
                    })
                return found_images
        
        # Check parent directory for job-prefixed files
        if os.path.exists(output_dir):
            parent_files = [f for f in os.listdir(output_dir) 
                          if f.startswith(job_id) and f.endswith(('.png', '.webp', '.jpg', '.jpeg'))]
            if parent_files:
                print(f"✅ Found {len(parent_files)} files in parent directory")
                for filename in parent_files:
                    found_images.append({
                        "filename": filename,
                        "subfolder": "",
                        "type": "output"
                    })
                return found_images
            
            # Also check for web_/orig_ prefixed files (recent files only)
            all_files = glob.glob(os.path.join(output_dir, "*"))
            image_files = [f for f in all_files 
                         if os.path.splitext(f)[1].lower() in {'.png', '.webp', '.jpg', '.jpeg'}]
            
            # Sort by modification time and check recent files
            if image_files:
                image_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                current_time = time.time()
                for filepath in image_files[:20]:  # Check only 20 most recent
                    # If file was created in last 30 seconds
                    if current_time - os.path.getmtime(filepath) < 30:
                        basename = os.path.basename(filepath)
                        if basename.startswith(('web_', 'orig_')):
                            found_images.append({
                                "filename": basename,
                                "subfolder": "",
                                "type": "output"
                            })
                if found_images:
                    print(f"✅ Found {len(found_images)} recent files")
                    return found_images
        
        if attempt < max_attempts - 1:
            print(f"⏳ Directory scan attempt {attempt + 1}/{max_attempts}, waiting {delay_seconds}s...")
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.5, 3.0)  # Exponential backoff
    
    return found_images


def get_image_outputs(
    job_id: str,
    prompt_id: str,
    comfyui_base: str,
    timeout: int = 600
) -> tuple[dict, str]:
    """Get image outputs using multiple methods with fallback.
    
    Returns (outputs_dict, source) where source indicates the method used.
    """
    from lib.ws_preview_relay import wait_for_prompt_completion, get_prompt_completion_data
    
    start_time = time.time()
    
    # Method 1: WebSocket completion data
    print(f"⏳ Waiting for completion via WebSocket...")
    completed = wait_for_prompt_completion(job_id, prompt_id, timeout=timeout)
    
    if completed:
        print(f"✅ WebSocket reported completion (took {time.time() - start_time:.1f}s)")
        completion_data = get_prompt_completion_data(job_id, prompt_id)
        
        if completion_data and completion_data.get("outputs"):
            return completion_data["outputs"], "websocket"
    
    # Method 2: ComfyUI History API
    print(f"⚠️ WebSocket data incomplete, trying history API...")
    history_start = time.time()
    history_url = f"{comfyui_base}/history/{prompt_id}"
    
    for attempt in range(8):
        try:
            response = urllib.request.urlopen(history_url, timeout=10)
            history_data = json.loads(response.read().decode('utf-8'))
            
            if prompt_id in history_data:
                outputs = history_data[prompt_id].get("outputs", {})
                if outputs:
                    print(f"✅ Got outputs from history (took {time.time() - history_start:.1f}s)")
                    return outputs, "history"
        except Exception as e:
            if attempt == 0:
                print(f"⚠️ History API error: {e}")
        
        backoff = min(0.25 * (2 ** attempt), 2.0)
        time.sleep(backoff)
    
    # Method 3: Directory scanning
    print(f"⚠️ History unavailable, scanning output directory...")
    dir_start = time.time()
    found_images = find_generated_images(job_id)
    
    if found_images:
        print(f"✅ Found images via directory scan (took {time.time() - dir_start:.1f}s)")
        # Build outputs structure
        outputs = {
            "directory_scan": {"images": found_images}
        }
        return outputs, "directory"
    
    # All methods failed
    total_time = time.time() - start_time
    raise RuntimeError(f"Failed to get outputs after {total_time:.1f}s - all methods exhausted")


def categorize_and_process_images(
    generated_images: list[dict],
    image_index: int,
    job_id: str,
    user_id: str,
    bucket: str,
    progress_ws_url: str,
    failure_tracker: dict
) -> list:
    """Categorize images by type and start async S3 upload threads.
    
    Returns list of threading.Thread objects to wait on.
    """
    import threading
    
    threads = []
    
    # Categorize images
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
    
    # Process web image
    if web_img:
        print(f"🔍 Starting async S3 processing (web) for image {image_index + 1}")
        thread = threading.Thread(
            target=process_and_save_single_image,
            args=(web_img, image_index, job_id, user_id, bucket, progress_ws_url, failure_tracker),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Process orig image if different from web
    if orig_img and orig_img != web_img:
        print(f"🔍 Starting async S3 processing (orig) for image {image_index + 1}")
        thread = threading.Thread(
            target=process_and_save_single_image,
            args=(orig_img, image_index, job_id, user_id, bucket, progress_ws_url, failure_tracker),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    return threads


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
        
        # Use deterministic per-take numbering regardless of ComfyUI's internal counters
        # This guarantees IMG-01..IMG-0N even if multiple Save nodes increment counters
        base_name = f"IMG-{image_index + 1:02d}"
        # Detect which Save node produced this file
        # Convention from workflow: orig_*.png (original PNG), web_*.webp (1024px webp)
        variant = "unknown"
        try:
            lower_name = filename.lower()
            if lower_name.startswith("orig_"):
                variant = "orig"
            elif lower_name.startswith("web_"):
                variant = "web"
        except Exception:
            pass
        
        print(f"🖼️ Processing ComfyUI image: {image_path}")
        print(f"🔍 Cleaned base name: {base_name}")
        
        # Prepare target folders
        orig_dir = f"/data/{user_id}/inference/{job_id}/orig"
        web_dir = f"/data/{user_id}/inference/{job_id}/web"
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(web_dir, exist_ok=True)

        final_image_url = None

        if variant == "orig":
            # Just copy to orig folder, no processing
            orig_filename = f"{base_name}{original_ext}"
            orig_path = f"{orig_dir}/{orig_filename}"
            shutil.copy(image_path, orig_path)
            print(f"  ✅ Copied original PNG to S3 (orig variant): {orig_path}")
        elif variant == "web":
            # Use ComfyUI 1024px webp as the main web image; also make 720/480 from it
            # Copy 1024 as IMG-XX.webp
            web_1024_filename = f"{base_name}.webp"
            web_1024_path = f"{web_dir}/{web_1024_filename}"
            shutil.copy(image_path, web_1024_path)
            print(f"  ✅ Saved 1024px WebP from workflow: {web_1024_path}")
            # Create downscaled variants from the 1024 image
            try:
                with Image.open(web_1024_path) as img1024:
                    if img1024.mode in ('RGBA', 'LA', 'P'):
                        img1024 = img1024.convert('RGB')
                    for size in (720, 480):
                        web_copy = img1024.copy()
                        web_copy.thumbnail((size, size), Image.Resampling.LANCZOS)
                        web_path = f"{web_dir}/{base_name}-w{size}.webp"
                        save_ok = False
                        for attempt in range(3):
                            try:
                                web_copy.save(web_path, format='WEBP', quality=max(65, 85 - attempt * 10), method=6, optimize=True)
                                if os.path.exists(web_path) and os.path.getsize(web_path) > 0:
                                    save_ok = True
                                    break
                            except Exception:
                                pass
                            import time as _t
                            _t.sleep(0.1 * (attempt + 1))
                        if not save_ok and failure_tracker:
                            failure_tracker['errors'].append(f"Image {image_index + 1}: Failed to save {size}px WebP to S3")
            except Exception as _web_e:
                print(f"⚠️ Failed to generate downscaled WEBP variants: {_web_e}")

            final_image_url = f"user-images/{user_id}/inference/{job_id}/web/{web_1024_filename}"
        else:
            # Legacy path: generate from the given file (kept for compatibility)
            # Copy original
            orig_filename = f"{base_name}{original_ext}"
            orig_path = f"{orig_dir}/{orig_filename}"
            shutil.copy(image_path, orig_path)
            print(f"  ✅ Copied original to S3: {orig_path}")
            # Generate 1024/720/480 as before
            with Image.open(image_path) as img:
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                for size in (1024, 720, 480):
                    web_img = img.copy()
                    web_img.thumbnail((size, size), Image.Resampling.LANCZOS)
                    web_filename = f"{base_name}.webp" if size == 1024 else f"{base_name}-w{size}.webp"
                    web_path = f"{web_dir}/{web_filename}"
                    save_ok = False
                    for attempt in range(3):
                        try:
                            web_img.save(web_path, format='WEBP', quality=max(65, 85 - attempt * 10), method=6, optimize=True)
                            if os.path.exists(web_path) and os.path.getsize(web_path) > 0:
                                save_ok = True
                                break
                        except Exception:
                            pass
                        import time as _t
                        _t.sleep(0.1 * (attempt + 1))
                    if not save_ok and failure_tracker:
                        failure_tracker['errors'].append(f"Image {image_index + 1}: Failed to save {size}px WebP to S3")
                final_image_url = f"user-images/{user_id}/inference/{job_id}/web/{base_name}.webp"
        

        
        # Send WebSocket notification only for the 1024px web image
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
        
        # Save image to database via Edge Function (retry if needed). Only on web variant/legacy where final_image_url is set.
        try:
            import requests
            import os
            
            # Get Supabase credentials (use the overridden environment variables)
            supabase_url = os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
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
                    'original_path': f"user-images/{user_id}/inference/{job_id}/orig/{base_name}.png",
                    'web_path': final_image_url,
                    'width': 1024,  # TODO: Change this to the actual width of the image when we have the final workflow
                    'height': 1024,  # TODO: Change this to the actual height of the image when we have the final workflow
                    'format': 'png',
                    'bytes': 0  # We could calculate this but it's not critical
                }
                
                ef_ok = False
                ef_status = None
                ef_text = None
                for attempt in range(3):
                    try:
                        # Increase timeout to better tolerate dev/ngrok slowness
                        ef_resp = requests.post(ef_url, json=ef_body, headers=ef_headers, timeout=30)
                        ef_status = ef_resp.status_code
                        if ef_resp.ok:
                            ef_ok = True
                            break
                        else:
                            ef_text = ef_resp.text
                    except Exception as _ef_e:
                        ef_text = str(_ef_e)
                    import time as _t
                    _t.sleep(0.5 * (attempt + 1))
                if not ef_ok:
                    # Non-fatal: we already saved to S3 and notified via WS. Log warning only.
                    warn_msg = (
                        f"Image {image_index + 1}: Edge Function save warning (status={ef_status}): {ef_text}"
                    )
                    print(f"⚠️ {warn_msg}")
                    if failure_tracker:
                        # Do not mark as failed; record warning for diagnostics.
                        failure_tracker['errors'].append(warn_msg)
                
        except Exception as ef_e:
            if failure_tracker:
                failure_tracker['failed_images'].add(image_index)
                failure_tracker['errors'].append(f"Image {image_index + 1}: Edge Function error: {str(ef_e)}")
        
        # If we got here and have a final_image_url, consider it successful
        if final_image_url:
            return True
        else:
            # If this was an orig-only save (no web variant), do not treat as failure
            try:
                if variant == "orig":
                    return True
            except Exception:
                pass
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
    if env_tag not in {"dev", "staging", "prod"}:
        env_tag = "prod"

    print(f"🔍 DEBUG: Environment resolution:")
    print(f"🔍 DEBUG: input env = {input_data.get('env')}")
    print(f"🔍 DEBUG: resolved env_tag = {env_tag}")
    
    # Get original values before override
    original_url = os.environ.get("SUPABASE_URL", "")
    original_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Get environment-specific values
    env_url = os.getenv(f"SUPABASE_URL_{env_tag.upper()}")
    env_key = os.getenv(f"SUPABASE_SERVICE_ROLE_KEY_{env_tag.upper()}")
    
    print(f"🔍 DEBUG: Original SUPABASE_URL = {original_url}")
    print(f"🔍 DEBUG: Environment-specific SUPABASE_URL_{env_tag.upper()} = {env_url}")
    print(f"🔍 DEBUG: Original SUPABASE_SERVICE_ROLE_KEY = {'***' + original_key[-4:] if original_key else 'None'}")
    print(f"🔍 DEBUG: Environment-specific SUPABASE_SERVICE_ROLE_KEY_{env_tag.upper()} = {'***' + env_key[-4:] if env_key else 'None'}")

    # Override generic names so the rest of the code picks them up
    os.environ["SUPABASE_URL"] = env_url or original_url
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = env_key or original_key
    
    print(f"🔍 DEBUG: Final SUPABASE_URL = {os.environ.get('SUPABASE_URL')}")
    print(f"🔍 DEBUG: Final SUPABASE_SERVICE_ROLE_KEY = {'***' + os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')[-4:] if os.environ.get('SUPABASE_SERVICE_ROLE_KEY') else 'None'}")
    
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
            import json
            
            supabase_url = os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
            print(f"🔍 DEBUG: Environment variables for inference-start:")
            print(f"🔍 DEBUG: env_tag = {env_tag}")
            print(f"🔍 DEBUG: SUPABASE_URL = {supabase_url}")
            print(f"🔍 DEBUG: SUPABASE_SERVICE_ROLE_KEY = {'***' + service_role_key[-4:] if service_role_key else 'None'}")
            
            if supabase_url and service_role_key:
                start_url = f"{supabase_url}/functions/v1/inference-start"
                start_body = {"job_id": job_id}
                
                print(f"🔍 DEBUG: Calling inference-start at: {start_url}")
                print(f"🔍 DEBUG: Request body: {start_body}")
                
                start_req = urllib.request.Request(
                    url=start_url,
                    data=json.dumps(start_body).encode('utf-8'),
                    headers={
                        'Authorization': f'Bearer {service_role_key}',
                        'Content-Type': 'application/json',
                        'apikey': service_role_key,
                    },
                    method='POST'
                )
                
                response = urllib.request.urlopen(start_req, timeout=10)
                response_text = response.read().decode('utf-8')
                print(f"✅ inference-start response: {response.status} - {response_text}")

            else:
                print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY; skipping status update")
        except Exception as e:
            print(f"⚠️ Failed to update job status to running: {e}")
            import traceback
            print(f"🔍 DEBUG: Full error traceback: {traceback.format_exc()}")
        
        poll_server_health(PORT)

        # Expect prepared payload from EF
        prepared = input_data.get("prepared")
        

        
        if not prepared:
            raise RuntimeError("Missing 'prepared' payload from inference-create EF")

        # Load workflow JSON (prefer mounted /workflows). Default to WAN2.1.json when not provided
        from lib.workflow_loader import load_workflow_from_s3
        # Select workflow by quality; fall back to prepared.workflow only if quality-based key is missing
        p = input_data.get("params", {})
        req_quality = str((p or {}).get("quality", "1K")).upper()
        quality_wf = "V1.0_1K.json" if req_quality == "1K" else "V1.0.json"
        chosen_key = quality_wf
        try:
            wf = load_workflow_from_s3(chosen_key)
            print(f"🎯 Using quality-selected workflow: {chosen_key} (quality={req_quality})")
        except FileNotFoundError as _e:
            alt_key = prepared.get("workflow") or "V1.0_1K.json"
            print(f"⚠️ Quality-selected workflow not found: {chosen_key}. Falling back to prepared key: {alt_key}")
            wf = load_workflow_from_s3(alt_key)
            chosen_key = alt_key
        
        from lib.workflow_patcher import compute_dimensions, patch_workflow
        settings_override = input_data.get("settings_override") or {}
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
        
        # Get AWS bucket early since it's needed in job metadata
        bucket = os.environ.get("AWS_BUCKET")
        
        # Store enhanced job metadata for webhook retrieval and debugging
        job_metadata = {
            "user_id": user_id,
            "character_id": input_data.get("character_id"),
            "style_id": input_data.get("style_id"),
            "workflow_key": chosen_key,
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
                        progress_req = urllib.request.Request(
                            url=post_url,
                            data=progress_body,
                            headers={'Content-Type': 'application/json'},
                            method='POST'
                        )
                        urllib.request.urlopen(progress_req, timeout=5)
                    print(f"📊 Sent progress update: image {image_index + 1}/{nb_takes}")
                except Exception as e:
                    print(f"⚠️ Failed to send progress update: {e}")
            
            # Title-based overrides are handled centrally in patch_workflow now

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
                aspect_ratio=p.get("aspect_ratio", "1:1"),
                settings_override=settings_override,
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
                "client_id": client_id,
                "workflow_key": chosen_key
            }
            
            # Debug: summarize workflow before submitting to ComfyUI
            try:
                class_types = sorted({
                    node.get("class_type", "")
                    for node in patched.values()
                    if isinstance(node, dict)
                })
                titles = [
                    node.get("_meta", {}).get("title")
                    for node in patched.values()
                    if isinstance(node, dict) and node.get("_meta", {}).get("title")
                ]
                print(f"🔎 Workflow summary: nodes={len(patched)}, classes={class_types[:12]}")
                if titles:
                    print(f"🔎 Titles include: {titles[:12]}")
                # Basic latent-node presence check
                if not any(
                    isinstance(n, dict) and n.get("class_type") in (
                        "EmptyHunyuanLatentVideo",
                        "EmptyLatentImage",
                        "EmptySD3LatentImage",
                        "EmptyLTXVLatentVideo",
                    )
                    for n in patched.values()
                ):
                    print("⚠️ No latent node found (EmptyHunyuanLatentVideo/EmptyLatentImage/EmptySD3LatentImage/EmptyLTXVLatentVideo)")
            except Exception as _summ_e:
                print(f"⚠️ Failed to summarize workflow: {_summ_e}")

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
                try:
                    response = urllib.request.urlopen(req, timeout=30)
                    response_data = response.read().decode('utf-8')
                    response_json = json.loads(response_data)
                except urllib.error.HTTPError as http_err:
                    err_body = ""
                    try:
                        err_body = http_err.read().decode('utf-8', errors='replace')
                    except Exception:
                        pass
                    print(f"❌ ComfyUI /prompt HTTP {getattr(http_err, 'code', 'unknown')}. Body: {err_body[:2000]}")
                    raise RuntimeError(f"HTTP {getattr(http_err, 'code', 'unknown')} from ComfyUI /prompt: {err_body[:200]}")
                
                print(f"📨 Submitted image {image_index + 1} to ComfyUI: {response_json}")
                
                # Extract prompt_id from response
                prompt_id = response_json.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"No prompt_id in ComfyUI response for image {image_index + 1}: {response_json}")
                
                print(f"🎯 Image {image_index + 1} submitted! Prompt ID: {prompt_id}")
                
                # Get image outputs using our optimized helper
                print(f"⏳ Waiting for image {image_index + 1} to complete...")
                
                try:
                    outputs, source = get_image_outputs(job_id, prompt_id, comfyui_base, timeout=600)
                    print(f"✅ Image {image_index + 1} completed! (source: {source})")
                    
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
                        all_generated_images.extend(generated_images)
                        
                        # Categorize and process images asynchronously
                        threads = categorize_and_process_images(
                            generated_images, image_index, job_id, user_id, 
                            bucket, progress_ws_url, s3_failure_tracker
                        )
                        s3_processing_threads.extend(threads)
                            
                    else:
                        # If outputs structure is empty, do one final directory scan
                        print(f"⚠️ No images in outputs, attempting final directory scan...")
                        final_images = find_generated_images(job_id, max_attempts=5)
                        
                        if final_images:
                            # Found images via directory scan
                            for img in final_images:
                                img["image_index"] = image_index
                                all_generated_images.append(img)
                            
                            print(f"✅ Recovered {len(final_images)} images via final directory scan")
                            
                            # Process these images too
                            threads = categorize_and_process_images(
                                final_images, image_index, job_id, user_id,
                                bucket, progress_ws_url, s3_failure_tracker
                            )
                            s3_processing_threads.extend(threads)
                        else:
                            raise RuntimeError(f"No generated images found for image {image_index + 1}")
                except Exception as e:
                    # Handle timeout or other errors from get_image_outputs
                    error_msg = str(e)
                    if "timed out" in error_msg.lower():
                        raise RuntimeError(f"Image {image_index + 1} generation timed out")
                    else:
                        raise RuntimeError(f"Image {image_index + 1} generation failed: {error_msg}")
                    
            except Exception as e:
                # Re-raise with image index context if not already included
                error_msg = str(e)
                if f"Image {image_index + 1}" not in error_msg:
                    raise RuntimeError(f"Image {image_index + 1} generation failed: {error_msg}")
                else:
                    raise
        
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
            # Use the overridden environment variables (already set based on env_tag in main function)
            supabase_url = os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            
            print(f"🔍 DEBUG: inference-complete call:")
            print(f"🔍 DEBUG: SUPABASE_URL = {supabase_url}")
            print(f"🔍 DEBUG: SUPABASE_SERVICE_ROLE_KEY = {'***' + service_role_key[-4:] if service_role_key else 'None'}")
            
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
                
                print(f"🔍 DEBUG: Calling inference-complete at: {ef_url}")
                print(f"🔍 DEBUG: Request body: {ef_body}")
                
                ef_resp = requests.post(ef_url, json=ef_body, headers=ef_headers, timeout=20)
                if ef_resp.ok:
                    print(f"✅ Called inference-complete Edge Function successfully")
                    print(f"🎯 Inference pipeline completed successfully!")
                else:
                    print(f"⚠️ inference-complete EF error: {ef_resp.status_code} {ef_resp.text}")
            else:
                print(f"⚠️ Missing Supabase credentials for environment")
                
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
        
        # Update database job status to failed via inference-complete EF
        try:
            import requests
            env = (env_tag if 'env_tag' in locals() else (input_data.get("env") or "dev")).lower()
            supabase_url = os.environ.get(f'SUPABASE_URL_{env.upper()}') or os.environ.get('SUPABASE_URL')
            service_role_key = os.environ.get(f'SUPABASE_SERVICE_ROLE_KEY_{env.upper()}') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            if supabase_url and service_role_key:
                ef_url = f"{supabase_url}/functions/v1/inference-complete"
                ef_headers = {
                    'Authorization': f'Bearer {service_role_key}',
                    'Content-Type': 'application/json',
                    'apikey': service_role_key,
                }
                ef_body = {
                    'job_id': job_id,
                    'success': False,
                    'error_message': str(e)
                }
                ef_resp = requests.post(ef_url, json=ef_body, headers=ef_headers, timeout=20)
                if ef_resp.ok:
                    print(f"✅ Marked job {job_id} as failed via inference-complete EF")
                else:
                    print(f"⚠️ inference-complete EF failure update error: {ef_resp.status_code} {ef_resp.text}")
            else:
                print("⚠️ Missing Supabase credentials; cannot mark job failed in DB")
        except Exception as ef_fail:
            print(f"⚠️ Failed to update job failure via inference-complete EF: {ef_fail}")
        
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
    gpu="H200",  # Fast
    image=cuda_image,
    secrets=[aws_secret, inference_secret, supabase_secret],
    volumes={**user_images_mount, **workflows_mount, MODELS_PATH: models_volume},
    timeout=30000,
    scaledown_window=300,
    max_containers=40,
    retries=3,
)
@modal.concurrent(max_inputs=1)
class Fast:
    """H200 worker (Fast)."""

    @modal.enter()
    def setup_environment(self):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        _launch_inference_runtime(PORT)

    @modal.method()
    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return main(input_data)


@app.cls(
    gpu="H100",  # Quick
    image=cuda_image,
    secrets=[aws_secret, inference_secret, supabase_secret],
    volumes={**user_images_mount, **workflows_mount, MODELS_PATH: models_volume},
    timeout=30000,
    scaledown_window=300,
    max_containers=40,
    retries=3,
)
@modal.concurrent(max_inputs=1)
class Quick:
    """H100 worker (Quick)."""

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
    settings_override = request_data.get("settings_override")
    env = request_data.get("env")  # Get env parameter from request

    print(f"Received prepared: {prepared}")
    print(f"Received env: {env}")

    # Try Fast (H200) first, fall back to Quick (H100) if spawn fails
    try:
        gpu = Fast()
        payload = {
            "user_id": user_id,
            "job_id": job_id,
            "prepared": prepared,
            "params": params,
            "settings_override": settings_override,
            "env": env,  # Pass env to GPU worker
            "gpu_type": "H200",
        }
        handle = gpu.run_inference.spawn(payload)
        print(f"✅ Submitted inference to Fast (H200) for job {job_id}")
        return {
            "status": "accepted",
            "gpu_type": "H200",
            "job_handle": str(handle),
            "job_id": job_id,
        }
    except Exception as e:
        print(f"Spawn failed for Fast (H200): {e}. Falling back to Quick (H100)...")
        try:
            gpu = Quick()
            payload = {
                "user_id": user_id,
                "job_id": job_id,
                "prepared": prepared,
                "params": params,
                "settings_override": settings_override,
                "env": env,  # Pass env to GPU worker
                "gpu_type": "H100",
            }
            handle = gpu.run_inference.spawn(payload)
            print(f"✅ Submitted inference to Quick (H100) for job {job_id}")
            return {
                "status": "accepted",
                "gpu_type": "H100",
                "job_handle": str(handle),
                "job_id": job_id,
            }
        except Exception as e2:
            raise HTTPException(status_code=503, detail=f"Submission failed for all GPU classes: {e2}")


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

    # --- Lightweight LoRA linker UI & endpoints for development ---
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi import Form

    LORA_MODELS_DIR = "/root/comfy/ComfyUI/models/loras"

    def _resolve_lora_source(path: str) -> str:
        p = (path or "").strip()
        if p.startswith("s3://"):
            # Our mounts:
            #   /data -> s3://primeshot-uploads-01/user-images/
            #   /workflows -> s3://primeshot-uploads-01/workflows/
            remainder = p[len("s3://"):]
            bucket = remainder
            key = ""
            if "/" in remainder:
                bucket, key = remainder.split("/", 1)
            # Normalize known prefixes regardless of bucket name
            if key.startswith("user-images/"):
                return f"/data/{key[len('user-images/') :]}"
            if key.startswith("workflows/"):
                return f"/workflows/{key[len('workflows/') :]}"
            # Fallback: assume it's a path under user-images prefix
            if key:
                return f"/data/{key}"
            return "/data"
        return p

    def _safe_link_name(source_path: str) -> str:
        # Create deterministic, conflict-free link name
        if source_path.startswith("/data/"):
            rel = source_path[len("/data/"):]
            return rel.replace("/", "_").replace("\\", "_")
        return os.path.basename(source_path)

    @web_app.get("/lora", response_class=HTMLResponse)
    async def lora_form():
        html = """
        <!doctype html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            <title>LoRA Linker</title>
            <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; }
                input[type=text] { width: 600px; padding: 8px; }
                button { padding: 8px 12px; }
                code { background: #f3f3f3; padding: 2px 4px; }
                .note { color: #555; margin-top: 8px; }
                .links { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h2>LoRA Linker (Dev)</h2>
            <form id=\"linkForm\" method=\"post\" action=\"/lora/link\">
                <label>Paste S3 path (e.g. <code>s3://primeshot-uploads-01/path/model.safetensors</code>) or an existing <code>/data/...</code> path:</label><br/>
                <input type=\"text\" name=\"path\" placeholder=\"s3://primeshot-uploads-01/lora/foo.safetensors\" />
                <button type=\"submit\">Link</button>
            </form>
            <div class=\"note\">Links are created in <code>/root/comfy/ComfyUI/models/loras</code>. Refresh ComfyUI after linking to see new LoRAs.</div>
            <div class=\"links\">
                <h3>Current Links</h3>
                <ul id=\"list\"></ul>
            </div>
            <script>
                async function refreshList() {
                    const res = await fetch('/lora/list');
                    const data = await res.json();
                    const ul = document.getElementById('list');
                    ul.innerHTML = '';
                    (data.links || []).forEach(name => {
                        const li = document.createElement('li');
                        const btn = document.createElement('button');
                        btn.textContent = 'unlink';
                        btn.onclick = async () => {
                            const fd = new FormData();
                            fd.append('name', name);
                            await fetch('/lora/unlink', { method: 'POST', body: fd });
                            refreshList();
                        };
                        li.textContent = name + ' ';
                        li.appendChild(btn);
                        ul.appendChild(li);
                    });
                }
                refreshList();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    @web_app.post("/lora/link")
    async def lora_link(request: Request, path: str = Form(None)):
        try:
            # Accept JSON payloads as well as form-encoded
            if path is None:
                try:
                    payload_bytes = await request.body()
                    if payload_bytes:
                        data = json.loads(payload_bytes.decode("utf-8"))
                        path = (data or {}).get("path")
                except Exception:
                    path = None
            if not path:
                return JSONResponse({"ok": False, "error": "Missing 'path'"}, status_code=400)

            source_path = _resolve_lora_source(path)
            if not os.path.exists(source_path):
                return JSONResponse({"ok": False, "error": f"Not found: {source_path}"}, status_code=404)

            os.makedirs(LORA_MODELS_DIR, exist_ok=True)
            link_name = _safe_link_name(source_path)
            target_path = os.path.join(LORA_MODELS_DIR, link_name)

            if os.path.exists(target_path) or os.path.islink(target_path):
                os.unlink(target_path)
            os.symlink(source_path, target_path)

            return JSONResponse({"ok": True, "linked": link_name, "source": source_path})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @web_app.get("/lora/list")
    async def lora_list():
        try:
            os.makedirs(LORA_MODELS_DIR, exist_ok=True)
            links = sorted([name for name in os.listdir(LORA_MODELS_DIR)])
            return {"ok": True, "links": links}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @web_app.post("/lora/unlink")
    async def lora_unlink(name: str = Form(...)):
        try:
            target_path = os.path.join(LORA_MODELS_DIR, name)
            if os.path.exists(target_path) or os.path.islink(target_path):
                os.unlink(target_path)
                return {"ok": True, "removed": name}
            return {"ok": False, "error": "Not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return web_app



@app.function(
    gpu=DEV_SERVER_GPU_TYPE,
    image=cuda_image,
    volumes={**user_images_mount, **workflows_mount, MODELS_PATH: models_volume},
    max_containers=1,
    timeout=60 * 60,  # 1 hour total timeout
    scaledown_window=1800  # keep container warm for 30 minutes of idle
)
@modal.concurrent(max_inputs=999)
@modal.asgi_app()
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
    
    # Always link LoRAs from /data/style_loras (fast, specific directory)
    def link_style_loras():
        source_dir = "/data/style_loras"
        linked_count = 0
        if not os.path.exists(source_dir):
            print(f"ℹ️ style_loras directory not found at {source_dir} (skipping)")
            return 0
        print(f"🔍 Scanning for style LoRAs in: {source_dir}")
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.safetensors'):
                    source_path = os.path.join(root, file)
                    relative_path = os.path.relpath(source_path, source_dir)
                    safe_filename = relative_path.replace('/', '_').replace('\\', '_')
                    target_path = os.path.join(lora_models_dir, safe_filename)
                    try:
                        if os.path.exists(target_path) or os.path.islink(target_path):
                            os.unlink(target_path)
                        os.symlink(source_path, target_path)
                        linked_count += 1
                        print(f"🔗 Linked style LoRA: {safe_filename}")
                    except Exception as e:
                        print(f"⚠️ Failed to link style LoRA {file}: {str(e)}")
        return linked_count

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
    
    # First, always link the specific style_loras
    try:
        style_count = link_style_loras()
        print(f"✅ Linked {style_count} style LoRAs from /data/style_loras")
    except Exception as e:
        print(f"⚠️ Error linking style_loras: {e}")

    # Optionally link all LoRAs for dev testing if enabled
    if os.getenv("DEV_LINK_ALL_LORAS", "0") in ("1", "true", "True"): 
        linked_loras = link_all_loras()
        print(f"✅ Linked {linked_loras} LoRAs from S3 bucket for dev server")
    else:
        print("⏭️ Skipping bulk LoRA linking. Visit /lora to link specific files.")
    
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
    
    # Launch ComfyUI on 8001; we'll serve a FastAPI app on 8000 that proxies to it
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8001 --use-sage-attention --gpu-only --bf16-unet --bf16-vae --output-directory /data/outputs --preview-method auto",
        shell=True,
        env=env
    )
    print("🌐 ComfyUI UI launched on 8001; proxy with /lora served on 8000")

    # Build FastAPI proxy app with /lora endpoints
    from fastapi import FastAPI, Request, WebSocket
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi import Form
    import httpx
    import websockets
    import asyncio

    app_proxy = FastAPI(title="ComfyUI Dev Server with LoRA Linker")

    LORA_MODELS_DIR = "/root/comfy/ComfyUI/models/loras"

    def _resolve_lora_source_dev(path: str) -> str:
        p = (path or "").strip()
        if p.startswith("s3://"):
            remainder = p[len("s3://"):]
            bucket = remainder
            key = ""
            if "/" in remainder:
                bucket, key = remainder.split("/", 1)
            if key.startswith("user-images/"):
                return f"/data/{key[len('user-images/') :]}"
            if key.startswith("workflows/"):
                return f"/workflows/{key[len('workflows/') :]}"
            if key:
                return f"/data/{key}"
            return "/data"
        return p

    def _safe_link_name_dev(source_path: str) -> str:
        if source_path.startswith("/data/"):
            rel = source_path[len("/data/"):]
            return rel.replace("/", "_").replace("\\", "_")
        return os.path.basename(source_path)

    @app_proxy.get("/lora", response_class=HTMLResponse)
    async def lora_form_dev():
        html = """
        <!doctype html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            <title>LoRA Linker</title>
            <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; }
                input[type=text] { width: 600px; padding: 8px; }
                button { padding: 8px 12px; }
                code { background: #f3f3f3; padding: 2px 4px; }
                .note { color: #555; margin-top: 8px; }
                .links { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h2>LoRA Linker (Dev)</h2>
            <form id=\"linkForm\" method=\"post\" action=\"/lora/link\"> 
                <label>Paste S3 path (e.g. <code>s3://primeshot-uploads-01/user-images/...</code>) or an existing <code>/data/...</code> path:</label><br/>
                <input type=\"text\" name=\"path\" placeholder=\"s3://primeshot-uploads-01/user-images/..../model.safetensors\" />
                <button type=\"submit\">Link</button>
            </form>
            <div class=\"note\">Links are created in <code>/root/comfy/ComfyUI/models/loras</code>. Refresh ComfyUI after linking to see new LoRAs.</div>
            <div class=\"links\"> <h3>Current Links</h3> <ul id=\"list\"></ul> </div>
            <script>
                async function refreshList() {
                    const res = await fetch('/lora/list');
                    const data = await res.json();
                    const ul = document.getElementById('list');
                    ul.innerHTML = '';
                    (data.links || []).forEach(name => {
                        const li = document.createElement('li');
                        const btn = document.createElement('button');
                        btn.textContent = 'unlink';
                        btn.onclick = async () => {
                            const fd = new FormData();
                            fd.append('name', name);
                            await fetch('/lora/unlink', { method: 'POST', body: fd });
                            refreshList();
                        };
                        li.textContent = name + ' ';
                        li.appendChild(btn);
                        ul.appendChild(li);
                    });
                }
                refreshList();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    @app_proxy.post("/lora/link")
    async def lora_link_dev(request: Request, path: str = Form(None)):
        try:
            if path is None:
                try:
                    payload_bytes = await request.body()
                    if payload_bytes:
                        data = json.loads(payload_bytes.decode("utf-8"))
                        path = (data or {}).get("path")
                except Exception:
                    path = None
            if not path:
                return JSONResponse({"ok": False, "error": "Missing 'path'"}, status_code=400)

            source_path = _resolve_lora_source_dev(path)
            if not os.path.exists(source_path):
                return JSONResponse({"ok": False, "error": f"Not found: {source_path}"}, status_code=404)

            os.makedirs(LORA_MODELS_DIR, exist_ok=True)
            link_name = _safe_link_name_dev(source_path)
            target_path = os.path.join(LORA_MODELS_DIR, link_name)
            if os.path.exists(target_path) or os.path.islink(target_path):
                os.unlink(target_path)
            os.symlink(source_path, target_path)
            return JSONResponse({"ok": True, "linked": link_name, "source": source_path})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    @app_proxy.get("/lora/list")
    async def lora_list_dev():
        try:
            os.makedirs(LORA_MODELS_DIR, exist_ok=True)
            links = sorted([name for name in os.listdir(LORA_MODELS_DIR)])
            return {"ok": True, "links": links}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app_proxy.post("/lora/unlink")
    async def lora_unlink_dev(name: str = Form(...)):
        try:
            target_path = os.path.join(LORA_MODELS_DIR, name)
            if os.path.exists(target_path) or os.path.islink(target_path):
                os.unlink(target_path)
                return {"ok": True, "removed": name}
            return {"ok": False, "error": "Not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Reverse proxy all other paths to ComfyUI on 8001
    UPSTREAM = "http://127.0.0.1:8001"

    def _filtered_headers(headers: dict) -> dict:
        hop_by_hop = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    @app_proxy.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]) 
    async def proxy_http(request: Request, path: str):
        if path.startswith("lora"):
            return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
        url = f"{UPSTREAM}/{path}"
        method = request.method
        headers = _filtered_headers(dict(request.headers))
        body = await request.body()
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                upstream_resp = await client.request(method, url, headers=headers, content=body, params=dict(request.query_params))
                resp_headers = _filtered_headers(dict(upstream_resp.headers))
                return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers=resp_headers)
        except httpx.ConnectError:
            return JSONResponse({"ok": False, "error": "ComfyUI upstream not ready"}, status_code=503)

    @app_proxy.websocket("/{path:path}")
    async def proxy_ws(websocket: WebSocket, path: str):
        if path.startswith("lora"):
            await websocket.close()
            return
        await websocket.accept()
        target = f"ws://127.0.0.1:8001/{path}"
        try:
            async with websockets.connect(target) as upstream:
                async def client_to_upstream():
                    try:
                        while True:
                            message = await websocket.receive()
                            if message.get("type") == "websocket.receive":
                                if "text" in message and message["text"] is not None:
                                    await upstream.send(message["text"])
                                elif "bytes" in message and message["bytes"] is not None:
                                    await upstream.send(message["bytes"])
                    except Exception:
                        try:
                            await upstream.close()
                        except Exception:
                            pass

                async def upstream_to_client():
                    try:
                        while True:
                            data = await upstream.recv()
                            if isinstance(data, (bytes, bytearray)):
                                await websocket.send_bytes(data)
                            else:
                                await websocket.send_text(str(data))
                    except Exception:
                        try:
                            await websocket.close()
                        except Exception:
                            pass

                await asyncio.gather(client_to_upstream(), upstream_to_client())
        except Exception:
            try:
                await websocket.close()
            except Exception:
                pass

    @app_proxy.on_event("startup")
    async def _wait_for_upstream():
        # Give ComfyUI a moment to boot to reduce initial 503s
        for _ in range(60):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    await client.get(f"{UPSTREAM}/")
                    break
            except Exception:
                await asyncio.sleep(2)

    return app_proxy