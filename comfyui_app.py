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
    # Install Node.js 20 LTS via NodeSource (for comfyui-api)
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get update && apt-get install -y nodejs && npm --version && node --version"
    )
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
    # Common dependencies required by various custom nodes and S3 access
    .pip_install("diffusers>=0.30.0", "transformers>=4.42.0", "accelerate>=0.30.0", "safetensors>=0.4.3", "boto3>=1.34.0", "psutil>=6.0.0")
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
    # Clone and build comfyui-api (fork: ctvbld/comfyui-api@main)
    .run_commands(
        "cd /root && git clone --depth=1 https://github.com/ctvbld/comfyui-api.git",
        "cd /root/comfyui-api && npm ci",
        "cd /root/comfyui-api && npm run build",
        "cd /root/comfyui-api && npm prune --omit=dev"
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
    
    # Launch ComfyUI with optimized settings
    cmd = (
        f"comfy launch --background -- --port {port} --use-sage-attention --gpu-only "
        f"--bf16-unet --bf16-vae --output-directory /data/outputs"
    )
    subprocess.run(cmd, shell=True, check=True, env=env)
    print("✅ ComfyUI server running with SageAttention and performance optimizations")

    # Setup comfyui-api compatibility and launch
    try:
        # Some comfyui-api builds expect ComfyUI at /opt/ComfyUI. Our install lives at /root/comfy/ComfyUI.
        # Provide both COMFY_HOME and a compatibility symlink so description scraping works.
        try:
            if os.path.exists("/root/comfy/ComfyUI") and not os.path.exists("/opt/ComfyUI"):
                os.symlink("/root/comfy/ComfyUI", "/opt/ComfyUI")
        except Exception:
            pass

        # Ensure /bin/sh understands 'source' by pointing it to bash (comfyui-api uses 'source' in its shell command)
        try:
            if os.path.exists("/bin/bash"):
                import pathlib
                sh_path = pathlib.Path("/bin/sh")
                try:
                    current = os.readlink(str(sh_path)) if sh_path.is_symlink() else ""
                except OSError:
                    current = ""
                if "bash" not in current:
                    subprocess.run("ln -sf /bin/bash /bin/sh", shell=True, check=False)
        except Exception:
            pass

        # Provide minimal ai-dock compatibility shims expected by comfyui-api (no-op env + venv)
        try:
            os.makedirs("/opt/ai-dock/etc", exist_ok=True)
            os.makedirs("/opt/ai-dock/bin", exist_ok=True)
            os.makedirs("/opt/ai-dock/venvs/comfyui/bin", exist_ok=True)

            env_sh = "/opt/ai-dock/etc/environment.sh"
            venv_set = "/opt/ai-dock/bin/venv-set.sh"
            activate = "/opt/ai-dock/venvs/comfyui/bin/activate"

            if not os.path.exists(env_sh):
                with open(env_sh, "w") as f:
                    f.write("#!/bin/bash\n# ai-dock env shim\n")
                subprocess.run(f"chmod +x {env_sh}", shell=True, check=False)
            if not os.path.exists(venv_set):
                with open(venv_set, "w") as f:
                    f.write("#!/bin/bash\nexport COMFYUI_VENV=/opt/ai-dock/venvs/comfyui\n")
                subprocess.run(f"chmod +x {venv_set}", shell=True, check=False)
            if not os.path.exists(activate):
                with open(activate, "w") as f:
                    f.write("#!/bin/bash\n# no-op activate\n")
                subprocess.run(f"chmod +x {activate}", shell=True, check=False)
        except Exception:
            pass

        api_env = os.environ.copy()
        # Explicitly configure comfyui-api per docs
        api_env.setdefault("COMFYUI_BASE_URL", f"http://127.0.0.1:{port}")  # for forks expecting base URL
        api_env.setdefault("COMFYUI_PORT_HOST", str(port))                   # salad tech config expects host port
        api_env.setdefault("DIRECT_ADDRESS", "127.0.0.1")
        api_env.setdefault("HOST", "::")
        api_env.setdefault("PORT", "3000")                                  # wrapper port
        api_env.setdefault("COMFY_HOME", "/root/comfy/ComfyUI")
        # Set OUTPUT_DIR to match where ComfyUI is actually saving files
        api_env["OUTPUT_DIR"] = "/data/outputs"
        # Disable comfyui-api's internal ComfyUI spawn (we already launched it)
        # Use /bin/true so child_process.spawn has a valid file instead of empty string
        api_env["CMD"] = "/bin/true"
        
        # Set all required AWS environment variables for S3 functionality  
        # According to comfyui-api docs: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
        aws_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_DEFAULT_REGION"]
        for key in aws_vars:
            if key not in api_env and key in os.environ:
                api_env[key] = os.environ[key]
        
        # Also set AWS_BUCKET for our S3 configuration
        if "AWS_BUCKET" in os.environ:
            api_env["AWS_BUCKET"] = os.environ["AWS_BUCKET"]
        
        # comfyui-api default wrapper port per docs is 3000
        print("🚀 Starting comfyui-api (expect default on http://127.0.0.1:3000) ...")
        
        # Prefer running the built entry directly to avoid missing npm scripts
        js_candidates = [
            "/root/comfyui-api/dist/index.js",
            "/root/comfyui-api/dist/server.js",
            "/root/comfyui-api/build/index.js",
            "/root/comfyui-api/build/server.js",
            "/root/comfyui-api/index.js",
        ]
        ts_candidates = [
            "/root/comfyui-api/src/index.ts",
            "/root/comfyui-api/src/server.ts",
        ]
        start_cmd = None
        
        for entry in js_candidates:
            if os.path.exists(entry):
                start_cmd = f"node {entry}"
                break
        
        if start_cmd is None:
            for entry in ts_candidates:
                if os.path.exists(entry):
                    # Run TypeScript entry with tsx via npx to avoid depending on package.json scripts
                    start_cmd = f"npx --yes tsx {entry}"
                    break
        
        if start_cmd is None:
            # Fail fast with a clear message; do not rely on npm scripts that don't exist
            raise RuntimeError(
                "comfyui-api entry not found. Tried JS: dist/index.js, dist/server.js, build/index.js, build/server.js, "
                "index.js and TS: src/index.ts, src/server.ts under /root/comfyui-api."
            )
        
        # Start comfyui-api with more logging
        print(f"🚀 Starting: {start_cmd}")
        print(f"🔧 Environment: PORT={api_env.get('PORT')}, HOST={api_env.get('HOST')}, CMD={api_env.get('CMD')}")
        
        # Debug AWS credentials being passed to comfyui-api
        aws_keys = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_BUCKET']
        aws_env_debug = {key: ('***HIDDEN***' if 'SECRET' in key else api_env.get(key, 'NOT_SET')) for key in aws_keys}
        print(f"🔧 AWS Environment: {aws_env_debug}")
        
        # Create a log file for comfyui-api output
        log_file_path = "/tmp/comfyui_api.log"
        
        process = subprocess.Popen(
            start_cmd, 
            shell=True, 
            env=api_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start a thread to capture and log comfyui-api output in real-time
        import threading
        import time
        
        def log_output():
            try:
                with open(log_file_path, 'w') as log_file:
                    log_file.write("=== ComfyUI-API Startup Log ===\n")
                    log_file.flush()
                    
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            # Write to log file
                            log_file.write(line)
                            log_file.flush()
                            
                            # Also print to Modal logs with prefix
                            print(f"[comfyui-api] {line.strip()}")
                            
                            # Break if process has ended
                            if process.poll() is not None:
                                break
            except Exception as e:
                print(f"⚠️ Error capturing comfyui-api logs: {e}")
        
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
        
        # Give it a moment and check if process is still alive
        time.sleep(2)
        
        if process.poll() is not None:
            # Process has already exited
            stdout, _ = process.communicate()
            print(f"❌ comfyui-api exited early with code {process.returncode}")
            print(f"Output: {stdout}")
            raise RuntimeError(f"comfyui-api failed to start: exit code {process.returncode}")
        else:
            print(f"✅ comfyui-api process started (PID: {process.pid})")

        # Quick warmup: wait for comfyui-api to accept connections (max ~10s)
        import urllib.request as _rq, urllib.error as _err
        
        # Probe common ports: 3000 (wrapper), 9000 (older), configurable via COMFY_API_BASE
        probe_ports = [
            os.environ.get("COMFY_API_BASE", "http://127.0.0.1:3000").rstrip("/"),
            "http://127.0.0.1:3000",
            "http://127.0.0.1:9000",
        ]
        ready = False
        
        for _ in range(10):  # ~10s max with faster individual probes
            for base in probe_ports:
                try:
                    # Prefer "ready"/"health" when available
                    resp = _rq.urlopen(base + "/ready", timeout=1.0)
                    if getattr(resp, "status", 200) == 200:
                        os.environ["COMFY_API_BASE"] = base
                        ready = True
                        break
                except Exception:
                    try:
                        resp2 = _rq.urlopen(base + "/health", timeout=1.0)
                        if getattr(resp2, "status", 200) == 200:
                            os.environ["COMFY_API_BASE"] = base
                            ready = True
                            break
                    except Exception:
                        try:
                            # Fall back to root: 404/405 means server is up
                            _rq.urlopen(base + "/", timeout=1.0)
                            os.environ["COMFY_API_BASE"] = base
                            ready = True
                            break
                        except _err.HTTPError as he:
                            if he.code in (404, 405):
                                os.environ["COMFY_API_BASE"] = base
                                ready = True
                                break
                        except Exception:
                            continue
                if ready:
                    break
        
            if ready:
                print(f"✅ comfyui-api is up at {os.environ.get('COMFY_API_BASE')}")
                break
            time.sleep(1.0)
        
        else:
            print("⚠️ comfyui-api did not become ready within timeout; will rely on submit-side retries")
    
    except Exception as e:
        print(f"❌ Failed to start comfyui-api: {e}")

    # Warmup both ComfyUI and comfyui-api
    try:
        import urllib.request as _rq
        _rq.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3)
        api_base = os.environ.get("COMFY_API_BASE", "http://127.0.0.1:3000")
        _rq.urlopen(api_base, timeout=3)
    except Exception:
        pass


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
    

def main(input_data: Dict[str, Any]) -> Dict[str, Any]:
    import sys; sys.path.append("/root")
    
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
                print(f"✅ Updated job {job_id} status to 'running'")
            else:
                print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY; skipping status update")
        except Exception as e:
            print(f"⚠️ Failed to update job status to running: {e}")
        
        poll_server_health(PORT)

        # Expect prepared payload from EF
        prepared = input_data.get("prepared")
        
        # 🔍 DETAILED LOGGING FOR DEBUGGING
        print(f"🔍 === INFERENCE JOB {job_id} DETAILS ===")
        print(f"📋 Input data keys: {list(input_data.keys())}")
        print(f"👤 User ID: {user_id}")
        print(f"🎭 Character ID: {input_data.get('character_id')}")
        print(f"🎨 Style ID: {input_data.get('style_id')}")
        print(f"⚙️ Environment: {env_tag}")
        print(f"🌐 Supabase URL: {os.environ.get('SUPABASE_URL', 'NOT_SET')[:50]}...")
        print(f"📦 Prepared payload: {'✅ Present' if prepared else '❌ Missing'}")
        if prepared:
            print(f"🔧 Prepared keys: {list(prepared.keys())}")
        print(f"🔍 === END DETAILS ===")
        
        if not prepared:
            raise RuntimeError("Missing 'prepared' payload from inference-create EF")

        # Load workflow JSON (prefer mounted /workflows). Default to WAN2.1.json when not provided
        from lib.workflow_loader import load_workflow_from_s3
        wf_key = prepared.get("workflow") or "WAN2.1.json"
        wf = load_workflow_from_s3(wf_key)
        
        from lib.workflow_patcher import compute_dimensions, patch_workflow
        p = input_data.get("params", {})
        width, height = compute_dimensions(p.get("quality", "1K"), p.get("aspect_ratio", "1:1"))
        
        # Character/style LoRAs are absolute paths under /data from EF; link basenames into models/loras
        char_lora = prepared.get("character_lora")
        style_lora = prepared.get("style_lora")
        
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
                dest = dest_dir / p.name
        
                try:
                    if dest.exists() or dest.is_symlink():
                        dest.unlink()
                except Exception:
                    pass
                try:
                    dest.symlink_to(p)
                except Exception:
                    _sh.copy2(str(p), str(dest))
                return p.name
        
            except Exception as _e:
                print(f"⚠️ LoRA link failed: {_e}")
                return None
        
        char_name = _link_lora(char_lora)
        style_name = _link_lora(style_lora)

        patched = patch_workflow(
            wf,
            prompt=prepared.get("prompt", ""),
            negative_prompt=prepared.get("negative_prompt", ""),
            width=width,
            height=height,
            seed=p.get("seed"),
            images_count=int(p.get("nb_takes", 1)),
            lora_filename=None,
            character_lora=char_name,
            style_lora=style_name,
            bypass_nodes=prepared.get("bypass_nodes"),
        )

        # 3) Start WS relay for progress/preview (non-blocking)
        try:
            import threading
            from lib.ws_preview_relay import start_relay
        
            client_id = str(uuid.uuid4())
            base = os.environ.get("PROGRESS_WS_URL")
        
            if base:
                progress_ws_url = base.replace("{job_id}", job_id) if "{job_id}" in base else base.rstrip("/") + f"/ws/broadcast/{job_id}"
                comfy_ws_url = f"ws://127.0.0.1:{PORT}/ws?clientId={client_id}"
                threading.Thread(target=start_relay, args=(progress_ws_url, comfy_ws_url, job_id), daemon=True).start()
        
        except Exception as e:
            print(f"⚠️ Progress relay not started: {e}")

        # 4) Submit to comfyui-api /prompt endpoint for execution
        import urllib.request, urllib.error
        
        # ComfyUI API always runs on port 3000
        api_base = "http://127.0.0.1:3000"
        # Use explicit override only if valid (/prompt or /workflow/*); else force /prompt
        _env_route = os.environ.get("COMFY_WORKFLOW_ENDPOINT")
        route = _env_route if (_env_route == "/prompt" or (_env_route or "").startswith("/workflow/")) else "/prompt"
        submit_paths = [route]
        webhook_url = os.environ.get("WEBHOOK_URL")
        webhook_secret = os.environ.get("WEBHOOK_SECRET")
        
        # Get AWS bucket early since it's needed in job metadata
        bucket = os.environ.get("AWS_BUCKET")
        
        # 🔍 API CALL LOGGING (without body - will log body details after it's created)
        print(f"🌐 === COMFYUI API CALL SETUP ===")
        print(f"🎯 API Base: {api_base}")
        print(f"📍 Route: {route}")
        print(f"🔗 Webhook URL: {webhook_url[:50] + '...' if webhook_url and len(webhook_url) > 50 else webhook_url}")
        print(f"🔐 Webhook Secret: {'✅ Set' if webhook_secret else '❌ Missing'}")
        print(f"🗄️ S3 Bucket: {'✅ Set' if bucket else '❌ Missing'}")
        print(f"🌐 === END API SETUP ===")
        
        # Store enhanced job metadata for webhook retrieval and debugging
        job_metadata = {
            "user_id": user_id,
            "character_id": input_data.get("character_id"),
            "style_id": input_data.get("style_id"),
            "workflow_key": wf_key,
            "dimensions": f"{width}x{height}",
            "nb_takes": int(p.get("nb_takes", 1)),
            "quality": p.get("quality", "1K"),
            "aspect_ratio": p.get("aspect_ratio", "1:1"),
            "seed": p.get("seed"),
            "character_lora": char_name,
            "style_lora": style_name,
            "api_base": api_base,
            "webhook_url": webhook_url is not None,
            "s3_configured": bucket is not None,
            "started_at": time.time(),
            "gpu_type": input_data.get("gpu_type", "unknown"),
            "env": env_tag  # Store environment for webhook to use correct Supabase instance
        }
        
        # Store in a simple in-memory cache (could use Redis in production)
        if not hasattr(tracker, '_job_metadata'):
            tracker._job_metadata = {}
        tracker._job_metadata[job_id] = job_metadata
        
        print(f"🔍 Job metadata: {job_metadata}")
        print(f"🌐 Environment for webhook: {env_tag}")
        
        # Map to comfyui-api /prompt schema
        body = {
            "id": job_id,
            "user_id": user_id,  # Include user_id for webhook
            "prompt": patched,
        }
        
        if bucket:
            # Save originals to /orig/ directory - we'll manually call EF after getting S3 URLs
            s3_config = {
                "bucket": bucket, 
                "prefix": f"user-images/{user_id}/inference/{job_id}/orig/", 
                "async": False  # Upload to S3 synchronously, return S3 URLs in response
            }
            body["s3"] = s3_config
            print(f"🔧 Using SYNCHRONOUS S3 mode - NO webhook in request")
            print(f"🔧 ComfyUI API will upload to S3 and return URLs in response")
        elif webhook_url:
            # Only use webhook if no S3 config (fallback to async webhook mode)
            if webhook_secret:
                separator = "&" if "?" in webhook_url else "?"
                body["webhook"] = f"{webhook_url}{separator}secret={webhook_secret}"
            else:
                body["webhook"] = webhook_url
            print(f"🔧 Using ASYNC webhook mode - no S3 upload")
            print(f"🔧 AWS Environment Check:")
            print(f"  - AWS_ACCESS_KEY_ID: {'✅ Set' if os.environ.get('AWS_ACCESS_KEY_ID') else '❌ Missing'}")
            print(f"  - AWS_SECRET_ACCESS_KEY: {'✅ Set' if os.environ.get('AWS_SECRET_ACCESS_KEY') else '❌ Missing'}")
            print(f"  - AWS_REGION: {os.environ.get('AWS_REGION', 'NOT_SET')}")
            print(f"  - AWS_DEFAULT_REGION: {os.environ.get('AWS_DEFAULT_REGION', 'NOT_SET')}")
            print(f"  - AWS_BUCKET: {bucket}")
            
            # Test S3 connectivity to help debug ComfyUI API upload issues
            print(f"🔧 Testing S3 connectivity for debugging:")
            try:
                import boto3
                s3_client = boto3.client('s3')
                
                print(f"🔍 S3 Client Configuration:")
                print(f"  - Region: {s3_client.meta.region_name}")
                print(f"  - AWS Access Key ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'NOT_SET')[:8]}...")
                
                # Test basic S3 access
                bucket_response = s3_client.head_bucket(Bucket=bucket)
                print(f"  ✅ S3 bucket '{bucket}' is accessible")
                print(f"  📍 Bucket region: {bucket_response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amz-bucket-region', 'unknown')}")
                
                # Test if we can create the path structure by uploading a tiny test file
                test_key = f"user-images/{user_id}/inference/{job_id}/debug_test_{int(time.time())}.txt"
                test_content = f"Modal S3 test at {time.time()}"
                
                put_response = s3_client.put_object(
                    Bucket=bucket,
                    Key=test_key,
                    Body=test_content.encode('utf-8'),
                    ContentType='text/plain'
                )
                print(f"  ✅ PUT response: {put_response.get('ResponseMetadata', {}).get('HTTPStatusCode')}")
                print(f"  📝 Uploaded test file: s3://{bucket}/{test_key}")
                
                # Verify the file actually exists by listing it
                list_response = s3_client.list_objects_v2(
                    Bucket=bucket, 
                    Prefix=test_key,
                    MaxKeys=1
                )
                if 'Contents' in list_response and list_response['Contents']:
                    file_info = list_response['Contents'][0]
                    print(f"  ✅ File verified in S3: {file_info['Key']} ({file_info['Size']} bytes)")
                    print(f"  📅 Last modified: {file_info['LastModified']}")
                else:
                    print(f"  ❌ File NOT found in S3 after upload!")
                    print(f"  🔍 List response: {list_response}")
                
                # Try to read it back
                try:
                    get_response = s3_client.get_object(Bucket=bucket, Key=test_key)
                    content = get_response['Body'].read().decode('utf-8')
                    print(f"  ✅ File content verified: '{content}'")
                except Exception as get_e:
                    print(f"  ❌ Failed to read back file: {get_e}")
                
                # Clean up test file
                try:
                    s3_client.delete_object(Bucket=bucket, Key=test_key)
                    print(f"  🗑️ Cleaned up test file")
                except Exception as del_e:
                    print(f"  ⚠️ Failed to delete test file: {del_e}")
                
            except Exception as s3_e:
                print(f"  ❌ S3 connectivity test failed: {s3_e}")
                print(f"  🔍 This might be why ComfyUI API S3 upload is failing")
                import traceback
                print(f"  📊 Full traceback: {traceback.format_exc()}")
            
            # Also check if ComfyUI API can access these vars
            print(f"🔧 ComfyUI API Environment Check (these vars need to be available to the subprocess):")
            for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_BUCKET']:
                val = os.environ.get(key)
                print(f"  - {key}: {'✅ Set' if val else '❌ Missing'}{' (first 8 chars: ' + val[:8] + '...)' if val and 'KEY' in key else ''}")
        else:
            print("⚠️ No AWS_BUCKET environment variable found - S3 uploads disabled")
        
        # 🔍 FINAL BODY LOGGING - after all modifications
        print(f"📊 === FINAL REQUEST BODY DETAILS ===")
        print(f"📊 Body keys: {list(body.keys())}")
        print(f"📏 Prompt size: {len(str(body.get('prompt', {})))} chars")
        print(f"🆔 Job ID: {body.get('id')}")
        print(f"👤 User ID in body: {body.get('user_id')}")
        print(f"🔗 Webhook in body: {'✅ Set' if body.get('webhook') else '❌ Missing'}")
        print(f"🗄️ S3 config in body: {'✅ Set' if body.get('s3') else '❌ Missing'}")
        if body.get('s3'):
            print(f"📁 S3 bucket: {body['s3'].get('bucket')}")
            print(f"📂 S3 prefix: {body['s3'].get('prefix')}")
        print(f"📊 === END FINAL BODY DETAILS ===")
        
        # Single readiness/health check before submit
        try:
            try:
                urllib.request.urlopen(api_base.rstrip('/') + "/ready", timeout=1.0)
                print("✅ ComfyUI API ready check passed")
            except Exception:
                try:
                    urllib.request.urlopen(api_base.rstrip('/') + "/health", timeout=1.0)
                    print("✅ ComfyUI API health check passed")
                except Exception:
                    urllib.request.urlopen(api_base, timeout=1.0)
                    print("✅ ComfyUI API responded to root endpoint")
            
            # Small delay to ensure API is fully ready after probe
            time.sleep(1.5)
        
        except Exception as e:
            print(f"⚠️ API readiness check failed: {e}, proceeding with retries")

        # No discovery: comfyui-api expects /prompt by default

        last_err = None
        success = False
        
        try:
            print(f"➡️ comfyui-api submit route: {route}")
        
        except Exception:
            pass
        
        for path in submit_paths:
            if not path:
                continue
        
            url = api_base.rstrip("/") + path
        
            # Aggressive retries for connection refused (server still booting)
            for attempt in range(10):  # ~15s total with backoff
                try:
                    req = urllib.request.Request(
                        url=url,
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {os.environ.get('COMFY_API_KEY')}"} if os.environ.get('COMFY_API_KEY') else {})},
                        method="POST",
                    )
                    response = urllib.request.urlopen(req, timeout=300)  # Longer timeout for S3 upload
                    response_data = response.read().decode('utf-8')
                    response_json = json.loads(response_data)
                    
                    print(f"📨 Submitted job to comfyui-api endpoint: {path}")
                    print(f"📨 ComfyUI API Response: {response_json}")
                    
                    # Check ComfyUI API logs for S3 errors
                    print(f"🔍 Checking ComfyUI API logs for S3 upload errors:")
                    try:
                        recent_logs = comfy_api_logs[-30:]  # Last 30 log lines
                        s3_error_found = False
                        for log_line in recent_logs:
                            if any(keyword in log_line.lower() for keyword in ['error uploading', 's3', 'upload', 'failed', 'error:']):
                                print(f"  ⚠️ {log_line}")
                                s3_error_found = True
                        if not s3_error_found:
                            print(f"  ✅ No S3 errors in recent logs")
                    except Exception as log_e:
                        print(f"  ⚠️ Could not check logs: {log_e}")
        
                    os.environ["COMFY_SUBMIT_PATH"] = path  # cache for subsequent jobs
        
                    # Process S3 URLs from synchronous response
                    if response_json.get("images"):
                        images = response_json["images"]
                        print(f"🔍 Analyzing response images ({len(images)} total):")
                        
                        s3_urls = []
                        base64_items = []
                        
                        for i, img in enumerate(images):
                            if isinstance(img, str) and img.startswith("s3://"):
                                s3_urls.append(img)
                                print(f"  ✅ S3 URL {i+1}: {img}")
                            else:
                                base64_items.append(img)
                                print(f"  ❌ Base64 data {i+1}: {len(str(img))} chars (S3 upload failed)")
                        
                        if s3_urls:
                            print(f"✅ S3 upload successful! {len(s3_urls)} URLs received: {s3_urls}")
                            
                            # Call inference-complete Edge Function directly
                            try:
                                # Create artifacts in the expected format
                                artifacts = {"orig": []}
                                for s3_url in s3_urls:
                                    if s3_url.startswith("s3://"):
                                        # Parse s3://bucket/key
                                        s3_parts = s3_url[5:].split("/", 1)
                                        if len(s3_parts) == 2:
                                            bucket_name, s3_key = s3_parts
                                            artifacts["orig"].append({
                                                "bucket": bucket_name,
                                                "key": s3_key
                                            })
                                
                                # Get environment for correct Supabase instance
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
                                    ef_body = {
                                        'job_id': job_id,
                                        'success': True,
                                        'artifacts': artifacts,
                                    }
                                    ef_resp = requests.post(ef_url, json=ef_body, headers=ef_headers, timeout=20)
                                    if ef_resp.ok:
                                        print(f"✅ Called inference-complete Edge Function successfully")
                                    else:
                                        print(f"⚠️ inference-complete EF error: {ef_resp.status_code} {ef_resp.text}")
                                else:
                                    print(f"⚠️ Missing Supabase credentials for {env_tag} environment")
                                    
                            except Exception as ef_e:
                                print(f"⚠️ Failed to call inference-complete EF: {ef_e}")
                        else:
                            print(f"⚠️ No S3 URLs in response, got: {images}")
                            print(f"🔍 This suggests ComfyUI API S3 upload failed")
                            # Check if response contains error information
                            if "error" in response_json:
                                print(f"❌ ComfyUI API error: {response_json['error']}")
                    else:
                        print(f"⚠️ No images in ComfyUI API response: {response_json}")
        
                    success = True
                    break
        
                except urllib.error.HTTPError as he:  # type: ignore[attr-defined]
                    if he.code == 404:
                        last_err = he
                        break  # try next path
                    else:
                        last_err = he
                        break
        
                except urllib.error.URLError as ue:  # connection refused case
                    last_err = ue
        
                    if getattr(ue.reason, 'errno', None) in (111,):  # Connection refused
                        retry_delay = min(1.0 + (attempt * 0.5), 3.0)  # Backoff: 1s -> 3s
                        print(f"🔄 Connection refused (attempt {attempt + 1}/10), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    break
        
                except Exception as e:
                    last_err = e
                    print(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
                    break
        
            if success:
                break
        
        if not success:
            error_msg = f"ComfyUI API submission failed after all retries. Last error: {last_err}"
            print(f"❌ {error_msg}")
            print(f"🔍 Tried endpoints: {submit_paths}")
            print(f"🔍 API Base: {api_base}")
            print(f"🔍 Request body keys: {list(body.keys())}")
            
            # Try to get recent logs for debugging
            try:
                log_file_path = "/tmp/comfyui_api.log"
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r') as f:
                        recent_logs = f.read().split('\n')[-20:]  # Last 20 lines
                        print(f"🔍 Recent ComfyUI API logs: {recent_logs}")
            except Exception:
                pass
            
            raise RuntimeError(error_msg)

        print(f"✅ Successfully submitted job {job_id} to ComfyUI API")
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
            error_details["suggestion"] = "ComfyUI API server may not be running or ready"
        elif "timeout" in str(e).lower():
            error_details["category"] = "timeout_error"
            error_details["suggestion"] = "Request timed out - server may be overloaded"
        elif "404" in str(e):
            error_details["category"] = "endpoint_error"
            error_details["suggestion"] = "ComfyUI API endpoint not found - check route configuration"
        elif "webhook" in str(e).lower():
            error_details["category"] = "webhook_error"
            error_details["suggestion"] = "Issue with webhook configuration or delivery"
        else:
            error_details["category"] = "general_error"
            error_details["suggestion"] = "Check logs for detailed error information"
        
        print(f"❌ Generation failed: {error_details}")
        
        try:
            tracker.mark_failed(job_id, str(e))
        except Exception as tracker_e:
            print(f"⚠️ Failed to update job tracker: {tracker_e}")
    
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
@modal.concurrent(max_inputs=3)
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


# Webhook endpoint for comfyui-api completion callbacks
@app.function(
    image=cuda_image,  # Use the same image as main functions so lib/ is available
    secrets=[aws_secret, inference_secret, supabase_secret]
)
@modal.fastapi_endpoint(method="POST", label="primeshot-webhook")
def webhook_endpoint(payload: Dict[str, Any], secret: str = Query(None)):
    """Accept comfyui-api webhook callbacks, extract S3 outputs, and return summary.

    Expected to receive user_id, job_id, and one or more output artifacts that
    include S3 keys/URLs for multi-resolution web variants and orig (2K/4K).
    
    Requires 'secret' query parameter matching WEBHOOK_SECRET.
    """
    import os
    
    # Verify webhook authentication via query parameter
    expected_secret = os.environ.get("WEBHOOK_SECRET")
    if not expected_secret:
        return {"status": "error", "error": "Webhook secret not configured"}
    
    if not secret:
        return {"status": "error", "error": "Missing 'secret' query parameter"}
    
    if secret != expected_secret:
        return {"status": "error", "error": "Invalid webhook secret"}
    
    # Import using absolute paths to avoid module resolution issues
    import sys
    import os
    sys.path.insert(0, '/root')
    
    try:
        from lib.webhook_handler import handle_webhook
        from lib.s3_artifacts import find_s3_entries, partition_artifacts
    except ImportError as ie:
        print(f"Import error: {ie}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        print(f"Contents of /root: {os.listdir('/root') if os.path.exists('/root') else 'N/A'}")
        print(f"Contents of /root/lib: {os.listdir('/root/lib') if os.path.exists('/root/lib') else 'N/A'}")
        raise

    try:
        result = handle_webhook(payload)
                
    except Exception as e:
        return {"status": "error", "error": str(e), "received": payload}

    # Broadcast completion (best effort)
    try:
        import asyncio
        import websockets
    
        async def _broadcast():
            base = os.environ.get("PROGRESS_WS_URL")
            if not base:
                return
    
            job_id = result.get("job_id") or "unknown"
    
            # Allow either full URL with {job_id} placeholder or base
            if "{job_id}" in base:
                ws_url = base.replace("{job_id}", job_id)
            else:
                ws_url = base.rstrip("/") + f"/ws/broadcast/{job_id}"
    
            msg = {
                "type": "inference_complete",
                "user_id": result.get("user_id"),
                "job_id": job_id,
                "artifacts": result.get("artifacts"),
            }
    
            try:
                async with websockets.connect(ws_url, ping_interval=None) as ws:
                    await ws.send(json.dumps(msg))
    
            except Exception as _e:
                print(f"⚠️ WS broadcast failed: {_e}")
        asyncio.run(_broadcast())
    
    except Exception as _wse:
        print(f"⚠️ WS broadcast error: {_wse}")

    return result


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

    class ConnectionManager:
        @staticmethod
        async def connect_listener(websocket: WebSocket, job_id: str) -> None:
            await websocket.accept()
            active_connections.setdefault(job_id, []).append(websocket)

        @staticmethod
        def disconnect(websocket: WebSocket) -> None:
            for job_id, sockets in list(active_connections.items()):
                if websocket in sockets:
                    sockets.remove(websocket)
                if not sockets:
                    active_connections.pop(job_id, None)

        @staticmethod
        async def broadcast(job_id: str, message: str) -> None:
            for ws in list(active_connections.get(job_id, [])):
                try:
                    await ws.send_text(message)
                except Exception:
                    ConnectionManager.disconnect(ws)

    manager = ConnectionManager()

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
@modal.concurrent(max_inputs=4, target_inputs=4)
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
    
    # Launch ComfyUI UI server with SageAttention and H100 performance optimizations
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000 --use-sage-attention --gpu-only --bf16-unet --bf16-vae --output-directory /data/outputs",
        shell=True,
        env=env
    )
    print("🌐 ComfyUI UI available at the development server URL with English locale")