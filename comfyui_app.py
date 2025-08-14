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

sys.path.insert(0, "/root/lib")

app = modal.App(name="primeshot-inference")

# Create Modal volumes for persistent storage
models_volume = modal.Volume.from_name("models-vol", create_if_missing=True)
aws_secret = modal.Secret.from_name("aws-secret")
inference_secret = modal.Secret.from_name("inference-secret")

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
    .pip_install("diffusers>=0.30.0", "transformers>=4.42.0", "accelerate>=0.30.0", "safetensors>=0.4.3", "boto3>=1.34.0")
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
    
    # Verify models are accessible
    if os.path.exists("/models"):
        print("🔍 Available models in /models:")
        import subprocess
        
        print("\n📁 Checking /models/unet:")
        subprocess.run("find /models/unet -name '*wan*' -type f 2>/dev/null || echo 'No flux models found in unet'", shell=True)
        
        print("\n📁 Checking /models/clip:")
        subprocess.run("find /models/clip -name '*.safetensors' -type f 2>/dev/null | head -5", shell=True)
        
        print("\n📁 Checking /models/vae:")
        subprocess.run("find /models/vae -name '*.safetensors' -type f 2>/dev/null | head -3", shell=True)
        
        print("\n📁 Checking /models/loras:")
        subprocess.run("find /models/loras -name '*.safetensors' -type f 2>/dev/null | head -3", shell=True)
        
        # Check essential models
        essential_models = [
            ("/models/unet/wan2.1_t2v_14B_fp16.safetensors", "WAN2.1 T2V 14B"),
            ("/models/clip/umt5_xxl_fp16.safetensors", "UMT5 XXL"),
            ("/models/vae/wan_2.1_vae.safetensors", "WAN2.1 VAE"),
        ]
        
        print("\n🔍 Model availability check:")
        for model_path, name in essential_models:
            if os.path.exists(model_path):
                if os.path.isfile(model_path):
                    size_gb = os.path.getsize(model_path) / (1024*1024*1024)
                    print(f"✅ {name}: {size_gb:.1f}GB")
                else:
                    print(f"✅ {name}: Available")
            else:
                print(f"⚠️ {name}: Not found")
        
        print("✅ ComfyUI configured to use /models volume via extra_model_paths.yaml")
        
    else:
        print("⚠️ /models directory not found")
    
    return True


# Shared runtime launcher so both Fast and Slow classes stay DRY
def _launch_inference_runtime(port: int) -> None:
    print("🔄 Initializing ComfyUI production environment...")
    if not setup_model_paths_config():
        raise RuntimeError("Failed to configure model paths")
    os.makedirs("/root/comfy/ComfyUI/models/loras", exist_ok=True)

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

    cmd = (
        f"comfy launch --background -- --port {port} --use-sage-attention --gpu-only "
        f"--bf16-unet --bf16-vae --output-directory /data/outputs"
    )
    subprocess.run(cmd, shell=True, check=True, env=env)
    print("✅ ComfyUI server running with SageAttention and performance optimizations")

    try:
        api_env = os.environ.copy()
        # Explicitly configure comfyui-api per docs
        api_env.setdefault("COMFYUI_BASE_URL", f"http://127.0.0.1:{port}")  # for forks expecting base URL
        api_env.setdefault("COMFYUI_PORT_HOST", str(port))                   # salad tech config expects host port
        api_env.setdefault("DIRECT_ADDRESS", "127.0.0.1")
        api_env.setdefault("HOST", "::")
        api_env.setdefault("PORT", "3000")                                  # wrapper port
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "AWS_BUCKET"]:
            if key not in api_env and key in os.environ:
                api_env[key] = os.environ[key]
        # comfyui-api default wrapper port per docs is 3000
        print("🚀 Starting comfyui-api (expect default on http://127.0.0.1:3000) ...")
        # Do not silence logs so we can see failures
        subprocess.Popen("cd /root/comfyui-api && npm run start", shell=True, env=api_env)

        # Robust warmup: wait for comfyui-api to accept connections (max ~30s)
        import urllib.request as _rq, urllib.error as _err
        # Probe common ports: 3000 (wrapper), 9000 (older), configurable via COMFY_API_BASE
        probe_ports = [
            os.environ.get("COMFY_API_BASE", "http://127.0.0.1:3000").rstrip("/"),
            "http://127.0.0.1:3000",
            "http://127.0.0.1:9000",
        ]
        ready = False
        for _ in range(60):
            for base in probe_ports:
                try:
                    _rq.urlopen(base + "/", timeout=0.5)
                    os.environ["COMFY_API_BASE"] = base
                    print(f"✅ comfyui-api is accepting connections at {base}")
                    ready = True
                    break
                except Exception:
                    continue
            if ready:
                break
            time.sleep(0.5)
        else:
            print("⚠️ comfyui-api did not become ready within timeout; will rely on submit-side retries")
    except Exception as e:
        print(f"❌ Failed to start comfyui-api: {e}")

    try:
        # Warmup
        import urllib.request as _rq
        _rq.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3)
        api_base = os.environ.get("COMFY_API_BASE", "http://127.0.0.1:3000")
        _rq.urlopen(api_base, timeout=3)
    except Exception:
        pass


# Shared base with all inference helpers so multiple GPU classes can reuse logic

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

 

def submit_to_comfyui_api_async(payload: Dict[str, Any]) -> None:
    import threading, json as _json, urllib.request
    def _submit():
        try:
            payload_with_hook = dict(payload)
            webhook_url = os.environ.get("WEBHOOK_URL")
            if webhook_url:
                payload_with_hook["webhook_url"] = webhook_url
            user_id = payload.get("user_id", "unknown"); job_id = payload.get("job_id", "job")
            payload_with_hook.setdefault("s3_prefix", f"user-images/{user_id}/inference/{job_id}/")
            base_url = os.environ.get("COMFY_API_BASE", "http://127.0.0.1:9000")
            route = os.environ.get("COMFY_WORKFLOW_ENDPOINT", "/workflows/image_default/run")
            url = base_url.rstrip("/") + "/" + route.lstrip("/")
            req = urllib.request.Request(url=url, data=_json.dumps(payload_with_hook).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print(f"📨 Submitted job to comfyui-api: {job_id}")
        except Exception as e:
            print(f"❌ comfyui-api submission failed: {e}")
    threading.Thread(target=_submit, daemon=True).start()

def main(input_data: Dict[str, Any]) -> Dict[str, Any]:
    import sys; sys.path.append("/root")
    from job_tracker import get_job_tracker
    tracker = get_job_tracker()
    job_id = input_data.get("job_id") or str(uuid.uuid4())
    user_id = input_data.get("user_id", "unknown")
    try:
        tracker.create_job(input_data, job_id)
        tracker.mark_processing(job_id)
        print(f"🎯 Starting generation job: {job_id}")
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
        # Prefer character LoRA; if absent, fall back to style LoRA (workflows may only have one loader)
        lora_filename = prepared.get("character_lora") or prepared.get("style_lora")
        patched = patch_workflow(
            wf,
            prompt=prepared.get("prompt", ""),
            negative_prompt=prepared.get("negative_prompt", ""),
            width=width,
            height=height,
            seed=p.get("seed"),
            images_count=int(p.get("nb_takes", 1)),
            lora_filename=lora_filename,
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

        # 4) Submit to comfyui-api generic endpoint for async execution
        import urllib.request, urllib.error, time as _time
        # autodetect port: prefer env, else 3000 (wrapper default), else 9000
        api_base = os.environ.get("COMFY_API_BASE") or "http://127.0.0.1:3000"
        url = api_base.rstrip("/") + "/workflows/run"
        webhook_url = os.environ.get("WEBHOOK_URL")
        body = {
            "workflow": patched,
            "client_id": client_id,
            "s3_prefix": f"user-images/{user_id}/inference/{job_id}/",
        }
        if webhook_url:
            body["webhook_url"] = webhook_url
        # Ensure comfyui-api is ready (avoid Connection refused on fast boots)
        for _i in range(60):  # up to ~30s
            try:
                urllib.request.urlopen(api_base, timeout=0.5)
                break
            except Exception:
                _time.sleep(0.5)
        # Retry submit up to ~30s
        last_err = None
        for _i in range(30):
            try:
                req = urllib.request.Request(url=url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=10)
                last_err = None
                break
            except Exception as e:
                last_err = e
                _time.sleep(1.0)
        if last_err:
            raise RuntimeError(f"comfyui-api submit failed after retries: {last_err}")

        # 5) Return accepted; completion goes via webhook -> EF -> DB
        return {"status": "accepted", "job_id": job_id}
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        try:
            tracker.mark_failed(job_id, str(e))
        except Exception:
            pass
        return {"job_id": job_id, "status": "failed", "error": str(e)}
    finally:
        pass


@app.cls(
    gpu="H100",
    image=cuda_image,
    secrets=[aws_secret, inference_secret],
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
    secrets=[aws_secret, inference_secret],
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
    secrets=[aws_secret, inference_secret],
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
    image=modal.Image.debian_slim().pip_install([
        "fastapi==0.115.4",
        "boto3>=1.34.0",
        "pillow>=10.3.0",
        "websockets>=12.0",
        "requests>=2.32.0"
    ]),
    secrets=[aws_secret, inference_secret]
)
@modal.fastapi_endpoint(method="POST", label="primeshot-webhook", requires_proxy_auth=True)
def webhook_endpoint(payload: Dict[str, Any]):
    """Accept comfyui-api webhook callbacks, extract S3 outputs, and return summary.

    Expected to receive user_id, job_id, and one or more output artifacts that
    include S3 keys/URLs for both web (1K) and orig (2K/4K) variants.
    """
    from lib.webhook_handler import handle_webhook
    from lib.s3_artifacts import find_s3_entries, partition_artifacts

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