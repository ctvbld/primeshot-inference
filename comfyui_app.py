# ---
# deploy: true  
# cmd: ["modal", "serve", "comfyui_app.py"]
# ---

# # ComfyUI Photography Platform with Flux + LoRA

# Advanced photography platform using ComfyUI with Flux.1-dev + custom LoRAs + 4K upscaling.
# Built following Modal's best practices with S3 integration and persistent model caching.

import json
import subprocess
import uuid
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import modal
import modal.experimental

# Create Modal volumes for persistent storage
vol = modal.Volume.from_name("models-vol", create_if_missing=True)
aws_secret = modal.Secret.from_name("aws-secret")

# S3 mount for user LoRAs and outputs
s3_mount = modal.CloudBucketMount(
    bucket_name="primeshot-uploads-01",
    key_prefix="user-images/",
    secret=aws_secret,
    read_only=False
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
        
        # Check for Flux models in both checkpoints and unet directories
        print("\n📁 Checking /models/checkpoints:")
        subprocess.run("find /models/checkpoints -name '*flux*' -type f 2>/dev/null || echo 'No flux models found in checkpoints'", shell=True)
        
        print("\n📁 Checking /models/unet:")
        subprocess.run("find /models/unet -name '*flux*' -type f 2>/dev/null || echo 'No flux models found in unet'", shell=True)
        
        print("\n📁 Checking /models/clip:")
        subprocess.run("find /models/clip -name '*.safetensors' -type f 2>/dev/null | head -5", shell=True)
        
        print("\n📁 Checking /models/vae:")
        subprocess.run("find /models/vae -name '*.safetensors' -type f 2>/dev/null | head -3", shell=True)
        
        # Check essential models
        essential_models = [
            ("/models/checkpoints/flux1-dev-fp8.safetensors", "Flux FP8 (checkpoints)"),
            ("/models/unet/flux1-dev.safetensors", "Flux UNET"),
            ("/models/unet/flux1-dev-fp8.safetensors", "Flux FP8 UNET"),
            ("/models/clip/clip_l.safetensors", "CLIP-L"),
            ("/models/clip/t5xxl_fp8_e4m3fn.safetensors", "T5XXL FP8"),
            ("/models/vae/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors", "VAE"),
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

# ## Image Building

# Build ComfyUI image following Modal's clean approach
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "locales")  # git for ComfyUI installation, locales for language support
    .run_commands("locale-gen en_US.UTF-8")  # Generate English locale
    .env({"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "LANGUAGE": "en_US:en"})  # Set default locale
    .pip_install("fastapi[standard]==0.115.4")  # web dependencies
    .pip_install("comfy-cli==1.3.8")  # ComfyUI installer
    .run_commands(  # install ComfyUI with NVIDIA support
        "comfy --skip-prompt install --fast-deps --nvidia"
    )
    # Install custom nodes for LoRA and upscaling
    .run_commands(
        "comfy node install --fast-deps was-node-suite-comfyui@1.0.2"
    )
    # Install SUPIR upscaling nodes (by Kijai) - direct GitHub install
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-SUPIR.git",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-SUPIR && pip install -r requirements.txt"
    )
    # Install SD-Latent-Interposer for Flux→SDXL latent conversion
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/city96/SD-Latent-Interposer.git",
        "cd /root/comfy/ComfyUI/custom_nodes/SD-Latent-Interposer && pip install huggingface-hub"
    )
    # Add workflow templates directory and job tracker
    .add_local_dir("workflows", "/root/workflows")
    .add_local_file("job_tracker.py", "/root/job_tracker.py")
    # Add ComfyUI model paths configuration
    .add_local_file("extra_model_paths.yaml", "/extra_model_paths.yaml")
)

# ## Modal App Definition

app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    gpu="A10G",  # Cost-effective for UI development
    volumes={"/models": vol, "/data": s3_mount},
    timeout=3600
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
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
    
    # Set environment variables to force English locale
    env = os.environ.copy()
    env.update({
        'LANG': 'en_US.UTF-8',
        'LC_ALL': 'en_US.UTF-8',
        'LANGUAGE': 'en_US:en'
    })
    
    # Launch ComfyUI UI server (extra_model_paths.yaml will be loaded automatically)
    subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000 --output-directory /data/outputs",
        shell=True,
        env=env
    )
    print("🌐 ComfyUI UI available at the development server URL")

# ## Production Image Generation

@app.cls(
    scaledown_window=300,  # 5 minute keep-alive
    gpu="L40S",  # High-performance for batch inference
    volumes={"/models": vol, "/data": s3_mount},
)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    """Production ComfyUI class for optimized batch image generation."""
    
    port: int = 8000

    @modal.enter()
    def launch_comfy_background(self):
        """Launch ComfyUI server in background when container starts."""
        print("🔄 Initializing ComfyUI production environment...")
        
        # Configure ComfyUI with extra_model_paths.yaml
        if not setup_model_paths_config():
            raise RuntimeError("Failed to configure model paths")
        
        # Create LoRA directory for job-specific dynamic linking
        os.makedirs("/root/comfy/ComfyUI/models/loras", exist_ok=True)
        
        # Set environment variables to force English locale
        env = os.environ.copy()
        env.update({
            'LANG': 'en_US.UTF-8',
            'LC_ALL': 'en_US.UTF-8',
            'LANGUAGE': 'en_US:en'
        })
        
        # Launch ComfyUI server in background (extra_model_paths.yaml will be loaded automatically)
        cmd = f"comfy launch --background -- --port {self.port} --output-directory /data/outputs"
        subprocess.run(cmd, shell=True, check=True, env=env)
        print("✅ ComfyUI server running in background")

    def setup_style_lora(self, lora_s3_path: str, style_id: str) -> str:
        """Link specific LoRA from S3 path for this style only."""
        if not lora_s3_path or lora_s3_path == "default.safetensors":
            return "default.safetensors"
        
        # Validate S3 path exists
        if not os.path.exists(lora_s3_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_s3_path}")
        
        # Extract filename and create style-specific name
        original_filename = os.path.basename(lora_s3_path)
        style_lora_filename = f"style_{style_id}_{original_filename}"
        
        # Create symlink in ComfyUI models directory
        lora_models_dir = "/root/comfy/ComfyUI/models/loras"
        style_lora_path = os.path.join(lora_models_dir, style_lora_filename)
        
        try:
            os.symlink(lora_s3_path, style_lora_path)
            print(f"🔗 Linked LoRA for style {style_id}: {original_filename}")
            return style_lora_filename
        except Exception as e:
            raise Exception(f"Failed to link LoRA: {str(e)}")
    
    def cleanup_style_lora(self, style_id: str) -> None:
        """Remove style-specific LoRA symlinks."""
        lora_models_dir = "/root/comfy/ComfyUI/models/loras"
        
        try:
            # Find and remove all symlinks for this style
            for filename in os.listdir(lora_models_dir):
                if filename.startswith(f"style_{style_id}_"):
                    lora_path = os.path.join(lora_models_dir, filename)
                    if os.path.islink(lora_path):
                        os.unlink(lora_path)
                        print(f"🧹 Cleaned up LoRA symlink: {filename}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to cleanup LoRA symlinks: {str(e)}")

    @modal.method()
    def generate_images(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate images using Flux + LoRA workflow."""
        import sys
        sys.path.append("/root")
        
        try:
            # Import job tracker
            from job_tracker import get_job_tracker
            tracker = get_job_tracker()
            
            # Create style entry with optional style_id from request
            style_id = request_data.get("style_id", str(uuid.uuid4()))
            tracker.create_job(request_data, style_id)
            tracker.mark_processing(style_id)
            
            print(f"🎯 Starting generation job: {style_id}")
            
            # Health check before processing
            self.poll_server_health()
            
            # Extract workflow name and parameters
            workflow_name = request_data.get("workflow_name", "flux_lora")
            parameters = request_data.get("parameters", {})
            
            # Validate workflow and parameters
            self.validate_workflow_request(workflow_name, parameters)
            
            # Setup style-specific LoRA (for workflows that need it)
            style_lora_filename = None
            if "lora_path" in parameters:
                lora_s3_path = parameters.get("lora_path")
                style_lora_filename = self.setup_style_lora(lora_s3_path, style_id)
            
            # Load and customize workflow
            workflow = self.load_workflow_template(workflow_name)
            user_id = request_data["user_id"]
            workflow = self.inject_parameters(workflow, workflow_name, parameters, style_id, user_id, style_lora_filename)
            
            # Execute workflow
            start_time = time.time()
            result = self.execute_workflow(workflow, style_id, user_id)
            execution_time = time.time() - start_time
            
            # Process results and upload to S3
            output_urls = self.process_results(result, request_data, style_id)
            
            # Mark job as completed
            tracker.mark_completed(style_id, output_urls, execution_time)
            
            # Cleanup style-specific LoRA
            self.cleanup_style_lora(style_id)
            
            return {
                "style_id": style_id,
                "status": "completed",
                "execution_time": execution_time,
                "images_generated": len(output_urls),
                "output_urls": output_urls,
                "cost_estimate": len(output_urls) * 0.10
            }
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            tracker.mark_failed(style_id, str(e))
            
            # Cleanup style-specific LoRA even on failure
            self.cleanup_style_lora(style_id)
            
            return {
                "style_id": style_id,
                "status": "failed", 
                "error": str(e)
            }

    def get_workflow_config(self) -> Dict[str, Any]:
        """Load workflow configuration metadata."""
        config_path = Path("/root/workflows/workflow_config.json")
        if not config_path.exists():
            raise FileNotFoundError("Workflow configuration not found")
        return json.loads(config_path.read_text())

    def load_workflow_template(self, workflow_name: str) -> Dict[str, Any]:
        """Load specified workflow template."""
        config = self.get_workflow_config()
        
        if workflow_name not in config["workflows"]:
            available = list(config["workflows"].keys())
            raise ValueError(f"Unknown workflow: {workflow_name}. Available: {available}")
        
        workflow_file = config["workflows"][workflow_name]["file"]
        workflow_path = Path(f"/root/workflows/{workflow_file}")
        
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_file}")
        
        return json.loads(workflow_path.read_text())

    def validate_workflow_request(self, workflow_name: str, parameters: Dict[str, Any]) -> None:
        """Validate workflow exists and parameters are correct."""
        config = self.get_workflow_config()
        
        if workflow_name not in config["workflows"]:
            available = list(config["workflows"].keys())
            raise ValueError(f"Unknown workflow: {workflow_name}. Available: {available}")
        
        # Validate parameters against workflow config
        workflow_config = config["workflows"][workflow_name]
        param_definitions = workflow_config.get("parameters", {})
        
        # Check required parameters
        for param_name, param_config in param_definitions.items():
            if param_config.get("required", False) and param_name not in parameters:
                raise ValueError(f"Missing required parameter: {param_name}")
        
        # Validate parameter types and ranges
        for param_name, param_value in parameters.items():
            if param_name in param_definitions:
                param_config = param_definitions[param_name]
                self._validate_parameter_value(param_name, param_value, param_config)

    def _validate_parameter_value(self, param_name: str, value: Any, config: Dict[str, Any]) -> None:
        """Validate individual parameter value."""
        param_type = config.get("type", "string")
        
        # Type validation
        if param_type == "integer" and not isinstance(value, int):
            raise ValueError(f"Parameter {param_name} must be an integer")
        elif param_type == "float" and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {param_name} must be a number")
        elif param_type == "string" and not isinstance(value, str):
            raise ValueError(f"Parameter {param_name} must be a string")
        
        # Range validation
        if "min" in config and value < config["min"]:
            raise ValueError(f"Parameter {param_name} must be >= {config['min']}")
        if "max" in config and value > config["max"]:
            raise ValueError(f"Parameter {param_name} must be <= {config['max']}")
        
        # Options validation
        if "options" in config and value not in config["options"]:
            raise ValueError(f"Parameter {param_name} must be one of: {config['options']}")

    def inject_parameters(self, workflow: Dict[str, Any], workflow_name: str, params: Dict[str, Any], style_id: str, user_id: str, style_lora_filename: str = None) -> Dict[str, Any]:
        """Generic parameter injection for any workflow based on configuration."""
        config = self.get_workflow_config()
        workflow_config = config["workflows"][workflow_name]
        param_definitions = workflow_config.get("parameters", {})
        parameter_mappings = workflow_config.get("parameter_mappings", {})
        
        # Apply default values for missing parameters
        processed_params = {}
        for param_name, param_config in param_definitions.items():
            if param_name in params:
                processed_params[param_name] = params[param_name]
            elif "default" in param_config:
                processed_params[param_name] = param_config["default"]
        
        # Add user-specific output prefix automatically
        processed_params["output_prefix"] = f"{user_id}/generated/{style_id}"
        
        # Apply each parameter using its mapping configuration
        for param_name, param_value in processed_params.items():
            mapping = parameter_mappings.get(param_name)
            if mapping:
                self._apply_parameter_mapping(workflow, mapping, param_value, style_id, style_lora_filename, workflow_name)
        
        print(f"✅ Applied {len(processed_params)} parameters to workflow {workflow_name}")
        return workflow

    def _apply_parameter_mapping(self, workflow: Dict[str, Any], mapping: Dict[str, Any], param_value: Any, style_id: str, style_lora_filename: str = None, workflow_name: str = None) -> None:
        """Apply a single parameter mapping to the workflow."""
        special_handler = mapping.get("special_handler")
        
        if special_handler == "resolution":
            # Parse resolution and apply to width/height mappings
            width, height = map(int, str(param_value).split('x'))
            
            width_mapping = mapping.get("width_mapping")
            if width_mapping:
                node_id = width_mapping["node"]
                input_key = width_mapping["input_key"]
                workflow[node_id]["inputs"][input_key] = width
            
            height_mapping = mapping.get("height_mapping")
            if height_mapping:
                node_id = height_mapping["node"]
                input_key = height_mapping["input_key"]
                workflow[node_id]["inputs"][input_key] = height
                
        elif special_handler == "random_seed":
            # Generate random seed if -1
            if param_value == -1:
                import random
                param_value = random.randint(0, 2**32 - 1)
            
            node_id = mapping["node"]
            input_key = mapping["input_key"]
            workflow[node_id]["inputs"][input_key] = param_value
            
        elif special_handler == "lora_file":
            # Handle LoRA file linking (already processed)
            if style_lora_filename and style_lora_filename != "default.safetensors":
                # Find LoRA node by checking workflow config
                config = self.get_workflow_config()
                node_mappings = config["workflows"].get(workflow_name, {}).get("node_mappings", {})
                if "lora_loader" in node_mappings:
                    lora_node = node_mappings["lora_loader"]
                    workflow[lora_node]["inputs"]["lora_name"] = style_lora_filename
                    
        elif special_handler == "style_output":
            # Apply style-specific output prefix
            node_id = mapping["node"]
            input_key = mapping["input_key"]
            workflow[node_id]["inputs"][input_key] = param_value
            
        else:
            # Standard parameter mapping
            node_id = mapping["node"]
            
            # Check if this is a widget mapping or input mapping
            if "widget_index" in mapping:
                # Widget value mapping
                widget_index = mapping["widget_index"]
                if "widgets_values" not in workflow[node_id]:
                    workflow[node_id]["widgets_values"] = []
                
                # Ensure the widgets_values list is long enough
                while len(workflow[node_id]["widgets_values"]) <= widget_index:
                    workflow[node_id]["widgets_values"].append(None)
                
                workflow[node_id]["widgets_values"][widget_index] = param_value
            else:
                # Input mapping
                input_key = mapping["input_key"]
                workflow[node_id]["inputs"][input_key] = param_value
            
            # Apply secondary mappings (e.g., LoRA strength to both model and clip)
            secondary_mappings = mapping.get("secondary_mappings", [])
            for secondary in secondary_mappings:
                sec_node_id = secondary["node"]
                if "widget_index" in secondary:
                    # Widget value mapping
                    widget_index = secondary["widget_index"]
                    if "widgets_values" not in workflow[sec_node_id]:
                        workflow[sec_node_id]["widgets_values"] = []
                    
                    # Ensure the widgets_values list is long enough
                    while len(workflow[sec_node_id]["widgets_values"]) <= widget_index:
                        workflow[sec_node_id]["widgets_values"].append(None)
                    
                    workflow[sec_node_id]["widgets_values"][widget_index] = param_value
                else:
                    # Input mapping
                    sec_input_key = secondary["input_key"]
                    workflow[sec_node_id]["inputs"][sec_input_key] = param_value

    def execute_workflow(self, workflow: Dict[str, Any], style_id: str, user_id: str) -> str:
        """Execute ComfyUI workflow and return output directory."""
        # Save workflow to temporary file
        workflow_file = f"/tmp/workflow_{style_id}.json"
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f)
        
        # Run workflow using comfy CLI
        cmd = f"comfy run --workflow {workflow_file} --wait --timeout 1200 --verbose"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        print(f"✅ Workflow executed successfully for style {style_id}")
        return f"/data/outputs/{user_id}/generated/{style_id}"

    def process_results(self, output_dir: str, request_data: Dict[str, Any], style_id: str) -> List[str]:
        """Process generated images and return S3 URLs."""
        import boto3
        
        output_urls = []
        user_id = request_data.get("user_id", "unknown")
        
        try:
            # Initialize S3 client  
            s3_client = boto3.client('s3')
            bucket_name = "primeshot-uploads-01"
            
            # Find generated images
            output_path = Path(output_dir)
            image_files = list(output_path.glob(f"{user_id}/generated/{style_id}*.png"))
            
            for i, image_file in enumerate(image_files):
                # Upload to S3
                s3_key = f"user-images/{user_id}/generated/{style_id}/image_{i+1:03d}.png"
                s3_client.upload_file(str(image_file), bucket_name, s3_key)
                
                # Generate URL
                s3_url = f"s3://{bucket_name}/{s3_key}"
                output_urls.append(s3_url)
                
                print(f"📤 Uploaded: {s3_url}")
                
        except Exception as e:
            print(f"❌ Error processing results: {e}")
        
        return output_urls

    @modal.fastapi_endpoint(method="POST")
    def api(self, request_data: Dict[str, Any]):
        """Main API endpoint for image generation."""
        from fastapi import HTTPException
        
        # Validate required top-level parameters
        if "user_id" not in request_data:
            raise HTTPException(status_code=400, detail="Missing required parameter: user_id")
        
        # Get workflow and parameters
        workflow_name = request_data.get("workflow_name", "flux_lora")
        parameters = request_data.get("parameters", {})
        user_id = request_data["user_id"]
        
        # Validate LoRA path for security (for any workflow that uses LoRAs)
        if "lora_path" in parameters:
            lora_path = parameters["lora_path"]
            if lora_path != "default.safetensors":
                # Ensure user can only access their own LoRAs
                expected_prefix = f"/data/{user_id}/loras/"
                if not lora_path.startswith(expected_prefix):
                    raise HTTPException(
                        status_code=403,
                        detail=f"LoRA path must start with {expected_prefix} for security"
                    )
                
                # Validate file extension
                if not lora_path.endswith(".safetensors"):
                    raise HTTPException(
                        status_code=400,
                        detail="LoRA file must have .safetensors extension"
                    )
        
        # Execute generation
        result = self.generate_images.local(request_data)
        return result

    def poll_server_health(self) -> None:
        """Check if ComfyUI server is healthy."""
        import socket
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/system_stats")
            urllib.request.urlopen(req, timeout=5)
            print("✅ ComfyUI server is healthy")
        except (socket.timeout, urllib.error.URLError) as e:
            print(f"❌ Server health check failed: {str(e)}")
            modal.experimental.stop_fetching_inputs()
            raise Exception("ComfyUI server is not healthy, stopping container")

# ## API Endpoints

@app.function(
    image=modal.Image.debian_slim().pip_install(["fastapi==0.104.1"]).add_local_dir("workflows", "/root/workflows")
)
@modal.fastapi_endpoint(method="GET", label="workflows")
def list_workflows():
    """List available workflows and their parameters."""
    import json
    from pathlib import Path
    
    try:
        config_path = Path("/root/workflows/workflow_config.json")
        if config_path.exists():
            return json.loads(config_path.read_text())
        else:
            return {"error": "Workflow configuration not found"}
    except Exception as e:
        return {"error": str(e)}

@app.function(
    image=modal.Image.debian_slim().pip_install(["fastapi==0.104.1", "pydantic==2.5.0"]).add_local_file("job_tracker.py", "/root/job_tracker.py"),
    secrets=[aws_secret]
)
@modal.fastapi_endpoint(method="GET", label="job-status") 
def job_status_endpoint(job_id: str):
    """Get status of a generation job."""
    import sys
    sys.path.append("/root")
    
    try:
        from job_tracker import get_job_tracker
        tracker = get_job_tracker()
        
        job_status = tracker.get_job_status(job_id)
        
        if not job_status:
            return {
                "error": f"Job {job_id} not found",
                "status": "error"
            }
        
        return job_status
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.function(
    image=modal.Image.debian_slim().pip_install(["fastapi==0.104.1"])
)
@modal.fastapi_endpoint(method="GET", label="health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ComfyUI",
        "version": "2.0.0",
        "features": ["flux", "lora", "upscaling", "s3"]
    } 