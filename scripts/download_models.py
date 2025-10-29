#!/usr/bin/env python3
"""
Comprehensive model download script for ComfyUI Photography Platform.
Downloads all required models with existence checks to avoid redundant downloads.
"""

import modal
import os
from pathlib import Path
from grpclib import GRPCError

models_volume = modal.Volume.from_name("models-vol")
app = modal.App("download-all-models")

@app.function(
    image=modal.Image.debian_slim().pip_install([
        "huggingface_hub>=0.19.0", 
        "hf_transfer>=0.1.4",
        "requests"
    ]).apt_install("git-lfs"),  # Required for large files
    volumes={"/models": models_volume},
    timeout=10800,  # 3 hours for large files
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def download_models():
    """Download all required models with smart existence checks and partial downloads."""
    from huggingface_hub import hf_hub_download, snapshot_download
    import shutil
    import os
    
    # Enable faster downloads for large files
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    print("🚀 ComfyUI Model Download - Smart checking and downloading missing models...")
    print("⚡ HF_TRANSFER enabled for faster large file downloads")
    
    # Clean any corrupted cache first
    print("🧹 Cleaning corrupted cache files...")
    import subprocess
    subprocess.run(["find", "/models", "-name", "*.incomplete", "-delete"], capture_output=True)
    subprocess.run(["find", "/models", "-name", "*.lock", "-delete"], capture_output=True)
    
    def check_file_integrity(file_path, min_size_mb=0.01):
        """Check if a file exists and has reasonable size."""
        if not os.path.exists(file_path):
            return False
        size_mb = os.path.getsize(file_path) / (1024*1024)
        return size_mb >= min_size_mb
    
    def get_missing_files(base_path, required_files, min_sizes=None):
        """Get list of missing or corrupted files from a directory."""
        if min_sizes is None:
            min_sizes = {}
        
        missing = []
        for file_path in required_files:
            full_path = os.path.join(base_path, file_path)
            min_size = min_sizes.get(file_path, 0.01)  # Default 0.01MB minimum
            if not check_file_integrity(full_path, min_size):
                missing.append(file_path)
        return missing
    
    def smart_snapshot_download(repo_id, local_dir, required_files=None, min_sizes=None):
        """Download only missing files from a snapshot repository."""
        os.makedirs(local_dir, exist_ok=True)
        
        if required_files:
            missing_files = get_missing_files(local_dir, required_files, min_sizes)
            if not missing_files:
                return True  # All files present
            
            print(f"📁 Missing files in {os.path.basename(local_dir)}: {missing_files}")
            
            # Download only missing files using allow_patterns
            patterns_to_download = []
            for missing_file in missing_files:
                # Add the specific file and its directory structure
                patterns_to_download.append(missing_file)
                # Also include parent directories to ensure proper structure
                parent_dir = os.path.dirname(missing_file)
                if parent_dir and parent_dir not in patterns_to_download:
                    patterns_to_download.append(f"{parent_dir}/*")
            
            print(f"📥 Downloading patterns: {patterns_to_download}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=patterns_to_download,
                ignore_patterns=["*.git*", "README.md"]
            )
        else:
            # Download everything if no specific files specified
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                ignore_patterns=["*.git*", "README.md"]
            )
        
        return True
    
    # Create all necessary directories
    directories = [
        "/models/unet",
        "/models/upscale_models", 
        "/models/clip",
        "/models/vae",
        "/models/loras",
        "/models/Wan-AI",
        "/models/ultralytics/bbox",
        "/models/ultralytics/sam"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Define all models to download with smart checking
    models_to_download = [
        {
            "name": "Wan-AI Wan2.2-T2V-A14B",
            "type": "snapshot",
            "check_path": "/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers-bf16",
            "repo_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers-bf16"
        },
        {
            "name": "Wan-AI Wan2.1-T2V-14B",
            "type": "snapshot",
            "check_path": "/models/Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "repo_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        },
        {
            "name": "UMT5 XXL FP16 Text Encoder (Wan 2.2 Repackaged - ~10GB)",
            "type": "single_file",
            "check_path": "/models/clip/umt5_xxl_fp16.safetensors",
            "min_size_gb": 9,
            "download_func": lambda: shutil.move(
                hf_hub_download(
                    repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    filename="split_files/text_encoders/umt5_xxl_fp16.safetensors",
                    local_dir="/models/clip"
                ),
                "/models/clip/umt5_xxl_fp16.safetensors"
            )
        },
        {
            "name": "Wan 2.2 VAE (Text-to-Video VAE - ~335MB)",
            "type": "single_file",
            "check_path": "/models/vae/wan_2.2_vae.safetensors",
            "min_size_mb": 300,
            "download_func": lambda: shutil.move(
                hf_hub_download(
                    repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    filename="split_files/vae/wan2.2_vae.safetensors",
                    local_dir="/models/vae"
                ),
                "/models/vae/wan_2.2_vae.safetensors"
            )
        },
        {
            "name": "Face YOLOv9c Bounding Box Model (Face Detection - 52MB)",
            "type": "single_file",
            "check_path": "/models/ultralytics/bbox/face_yolov9c.pt",
            "min_size_mb": 52,
            "download_func": lambda: hf_hub_download(
                repo_id="Bingsu/adetailer",
                filename="face_yolov9c.pt",
                local_dir="/models/ultralytics/bbox"
            )
        },
        {
            "name": "SAM Vit B (Semantic Segmentation - 375MB)",
            "type": "single_file",
            "check_path": "/models/sams/sam_vit_b_01ec64.pth",
            "min_size_mb": 375,
            "download_func": lambda: hf_hub_download(
                repo_id="scenario-labs/sam_vit",
                filename="sam_vit_b_01ec64.pth",
                local_dir="/models/sams"
            )
        },
        {
            "name": "4x_NMKD-Siax_200k Upscaler (Best for Faces - 67MB)",
            "type": "single_file",
            "check_path": "/models/upscale_models/4x_NMKD-Siax_200k.pth",
            "min_size_mb": 60,
            "download_func": lambda: hf_hub_download(
                repo_id="gemasai/4x_NMKD-Siax_200k",
                filename="4x_NMKD-Siax_200k.pth",
                local_dir="/models/upscale_models"
            )
        },
        {
            "name": "4xNomosUniDAT_otf Upscaler (Best for Faces - 154MB)",
            "type": "single_file",
            "check_path": "/models/upscale_models/4xNomosUniDAT_otf.safetensors",
            "min_size_mb": 154,
            "download_func": lambda: hf_hub_download(
                repo_id="Phips/4xNomosUniDAT_otf",
                filename="4xNomosUniDAT_otf.safetensors",
                local_dir="/models/upscale_models"
            )
        },
        {
            "name": "WanVideo T2V 14B Light x2v CFG Step Distill LoRA (Video Generation)",
            "type": "single_file",
            "check_path": "/models/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
            "min_size_mb": 100,
            "download_func": lambda: hf_hub_download(
                repo_id="Kijai/WanVideo_comfy",
                filename="Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                local_dir="/models/loras"
            )
        },
        {
            "name": "Wan2.1 T2V 14B FusionX LoRA (Video Generation Enhancement - 317MB)",
            "type": "single_file",
            "check_path": "/models/loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
            "min_size_mb": 300,
            "download_func": lambda: hf_hub_download(
                repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
                filename="FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors",
                local_dir="/models/loras"
            )
        },
        {
            "name": "SDXLrender v2.0 (SDXL Render - 181MB)",
            "type": "single_file",
            "check_path": "/models/loras/SDXLrender_v2.0.safetensors",
            "min_size_mb": 181,
            "download_func": lambda: hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                local_dir="/models/loras"
            )
        },
        {
            "name": "more_details (SDXL Render - 10MB)",
            "type": "single_file",
            "check_path": "/models/loras/more_details.safetensors",
            "min_size_mb": 10,
            "download_func": lambda: hf_hub_download(
                repo_id="digiplay/LORA",
                filename="more_details.safetensors",
                local_dir="/models/loras"
            )
        },
        {
            "name": "control_v11f1e_sd15_tile (ControlNet - 723MB)",
            "type": "single_file",
            "check_path": "/models/controlnet/control_v11f1e_sd15_tile.safetensors",
            "min_size_mb": 723,
            "download_func": lambda: hf_hub_download(
                repo_id="copybaiter/ControlNet",
                filename="control_v11f1e_sd15_tile.safetensors",
                local_dir="/models/controlnet"
            )
        },
    ]
    
    # Download each model with smart existence check
    downloaded_count = 0
    skipped_count = 0
    partial_downloads = 0
    
    for model in models_to_download:
        name = model["name"]
        check_path = model["check_path"]
        model_type = model["type"]
        
        # Smart checking based on model type
        skip_download = False
        
        if model_type == "single_file":
            # Single file check with size validation
            if os.path.exists(check_path):
                file_size_gb = os.path.getsize(check_path) / (1024*1024*1024)
                file_size_mb = os.path.getsize(check_path) / (1024*1024)
                
                # Check against expected minimum size
                min_size_gb = model.get("min_size_gb", 0)
                min_size_mb = model.get("min_size_mb", 0.01)
                
                if min_size_gb > 0 and file_size_gb >= min_size_gb:
                    print(f"✅ SKIP: {name} already exists ({file_size_gb:.1f}GB)")
                    skip_download = True
                elif min_size_mb > 0 and file_size_mb >= min_size_mb:
                    print(f"✅ SKIP: {name} already exists ({file_size_mb:.1f}MB)")
                    skip_download = True
                else:
                    expected = f"{min_size_gb}GB" if min_size_gb > 0 else f"{min_size_mb}MB"
                    actual = f"{file_size_gb:.1f}GB" if file_size_gb > 1 else f"{file_size_mb:.1f}MB"
                    print(f"🔄 SIZE MISMATCH: {name} - expected ≥{expected}, got {actual}, will re-download")
                    os.remove(check_path)  # Remove corrupted file
            
        elif model_type == "snapshot":
            # Smart directory check for snapshot downloads
            if os.path.exists(check_path):
                required_files = model.get("required_files", [])
                min_sizes = model.get("min_sizes", {})
                
                missing_files = get_missing_files(check_path, required_files, min_sizes)
                
                if not missing_files:
                    total_files = len([f for f in os.listdir(check_path) 
                                     if f.endswith(('.safetensors', '.bin', '.json'))])
                    print(f"✅ SKIP: {name} complete ({total_files} files)")
                    skip_download = True
                else:
                    print(f"🔄 PARTIAL: {name} - missing {len(missing_files)} files: {missing_files[:3]}{'...' if len(missing_files) > 3 else ''}")
                    # Don't delete directory, just download missing files
                    partial_downloads += 1
        
        if skip_download:
            skipped_count += 1
            continue
        
        # Download the model
        action = "PARTIAL DOWNLOAD" if model_type == "snapshot" and os.path.exists(check_path) else "DOWNLOADING"
        print(f"\n📦 {action}: {name}")
        
        try:
            if model_type == "snapshot":
                # Use smart snapshot download
                smart_snapshot_download(
                    repo_id=model["repo_id"],
                    local_dir=check_path,
                    required_files=model.get("required_files"),
                    min_sizes=model.get("min_sizes")
                )
            else:
                # Use existing download function for single files
                model["download_func"]()
            
            print(f"✅ SUCCESS: {name}")
            
            # Verify download
            if os.path.exists(check_path):
                if os.path.isfile(check_path):
                    size_gb = os.path.getsize(check_path) / (1024*1024*1024)
                    print(f"🔍 Verified: {name} - {size_gb:.1f}GB")
                elif os.path.isdir(check_path):
                    files = os.listdir(check_path)
                    model_files = [f for f in files if f.endswith(('.safetensors', '.bin'))]
                    total_size_mb = sum(
                        os.path.getsize(os.path.join(check_path, f)) / (1024*1024)
                        for f in files if os.path.isfile(os.path.join(check_path, f))
                    )
                    print(f"🔍 Verified: {name} - {len(model_files)} model files, {total_size_mb:.0f}MB total")
            else:
                print(f"❌ VERIFICATION FAILED: {name} - path does not exist after download!")
                
            downloaded_count += 1
        except Exception as e:
            print(f"❌ FAILED: {name} - {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue with other downloads
    
    # Force volume sync to ensure files are actually saved
    if downloaded_count > 0:
        print("🔄 Forcing volume sync to ensure persistence...")
        models_volume.commit()
        print("✅ Volume sync completed")
    
    # Final verification
    print(f"\n📊 DOWNLOAD SUMMARY:")
    print(f"📥 Downloaded: {downloaded_count} models")
    print(f"🔄 Partial downloads: {partial_downloads} models")
    print(f"⏭️ Skipped: {skipped_count} models (already complete)")
    
    print(f"\n🔍 FINAL VERIFICATION:")
    essential_checks = [
        ("/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers-bf16", "Wan-AI 2.2 Text-to-Video Diffusers"),
        ("/models/Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan-AI 2.1 Text-to-Video Diffusers"),
        ("/models/vae/wan_2.2_vae.safetensors", "Wan 2.2 VAE"),
        ("/models/upscale_models/4x_NMKD-Siax_200k.pth", "4x NMKD-Siax Upscaler"),
        ("/models/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", "WanVideo T2V 14B Light x2v CFG Step Distill LoRA"),
        ("/models/loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors", "Wan2.1 T2V 14B FusionX LoRA"),
    ]
    
    all_ready = True
    for path, component in essential_checks:
        if os.path.exists(path):
            if os.path.isfile(path):
                size_gb = os.path.getsize(path) / (1024*1024*1024)
                print(f"✅ {component}: {size_gb:.1f}GB")
            else:
                files = len([f for f in os.listdir(path) 
                           if f.endswith(('.safetensors', '.bin', '.json'))])
                print(f"✅ {component}: {files} files")
        else:
            print(f"❌ {component}: MISSING!")
            all_ready = False
    
    if all_ready:
        print(f"\n🎉 ALL MODELS READY! Platform is ready for image generation!")
    else:
        print(f"\n⚠️ Some essential models are missing. Check the logs above.")
    
    print(f"\n✅ Model download process completed!")

def _volume_dir_exists(volume: modal.Volume, dir_path: str) -> bool:
    """Return True if directory exists in the volume, else False."""
    try:
        volume.listdir(dir_path)
        return True
    except (FileNotFoundError, GRPCError, Exception):
        # Modal can raise GRPCError when directory doesn't exist
        return False

def _volume_file_exists(volume: modal.Volume, file_path: str) -> bool:
    """Return True if file exists in the volume, else False."""
    try:
        # read_file raises FileNotFoundError when not present; avoid loading big files
        # Use listdir on parent and check membership to avoid reading contents
        parent = os.path.dirname(file_path) or "/"
        base = os.path.basename(file_path)
        entries = volume.listdir(parent)
        names = []
        for e in entries:
            # e may be a string path or a mapping with name/path
            if isinstance(e, str):
                names.append(os.path.basename(e))
            elif isinstance(e, dict):
                name = e.get("name") or e.get("path") or ""
                names.append(os.path.basename(str(name)))
            else:
                # Fallback to string representation
                names.append(os.path.basename(str(e)))
        return base in set(names)
    except (FileNotFoundError, GRPCError, Exception):
        # Modal can raise GRPCError when directory doesn't exist
        return False

def _volume_list_names(volume: modal.Volume, dir_path: str) -> set:
    """Return a set of basenames present in a volume directory."""
    try:
        entries = volume.listdir(dir_path)
    except (FileNotFoundError, GRPCError, Exception):
        # Modal can raise GRPCError when directory doesn't exist
        return set()
    names = set()
    for e in entries:
        if isinstance(e, str):
            names.add(os.path.basename(e))
        elif isinstance(e, dict):
            name = e.get("name") or e.get("path") or ""
            names.add(os.path.basename(str(name)))
        else:
            names.add(os.path.basename(str(e)))
    return names

def upload_local_uploads(local_base: str = None, mappings: dict = None):
    """
    Upload files from local uploads/ subfolders into corresponding paths in the models volume.
    - local_base: base directory on local filesystem (default: repo's modal_apps/inference/uploads)
    - mappings: dict of { local_subdir: volume_subdir }, e.g., { "loras": "/loras" }
    Behavior:
      - Logs if a target directory does not exist in the volume
      - Skips files that already exist in the volume
      - Commits the volume if any uploads occurred
    """
    vol = models_volume  # modal.Volume.from_name("models-vol") already defined
    if local_base is None:
        local_base = str(Path(__file__).resolve().parents[1] / "uploads")
    if mappings is None:
        mappings = {
            "loras": "/loras",
            "checkpoints": "/checkpoints",
        }

    print(f"\n📤 Uploading from local uploads base: {local_base}")

    total_uploaded = 0
    for local_subdir, volume_subdir in mappings.items():
        local_dir = os.path.join(local_base, local_subdir)
        target_dir = volume_subdir

        if not os.path.isdir(local_dir):
            print(f"⏭️ Skip: local folder not found: {local_dir}")
            continue

        # Check if directory exists, auto-create if needed
        if not _volume_dir_exists(vol, target_dir):
            print(f"📁 Creating directory: {target_dir} (will be created on first file upload)")

        print(f"➡️  Syncing {local_dir} -> {target_dir}")

        local_files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
        if not local_files:
            print(f"(empty) {local_dir}")
            continue

        # Upload each file in its own batch to avoid aborting the whole group
        for fname in local_files:
            local_path = os.path.join(local_dir, fname)
            remote_path = os.path.join(target_dir, fname)
            # Per-file existence check to avoid ALREADY_EXISTS errors
            if _volume_file_exists(vol, remote_path):
                print(f"✅ Exists, skip: {remote_path}")
                continue
            try:
                with vol.batch_upload() as batch:
                    batch.put_file(local_path, remote_path)
                total_uploaded += 1
                print(f"📥 Uploaded: {remote_path}")
            except FileExistsError:
                # Gracefully handle race where file appeared between listing and upload
                print(f"✅ Exists (race), skip: {remote_path}")

    if total_uploaded > 0:
        # batch_upload() finalizes uploads; explicit commit is only valid inside a container.
        print(f"✅ Uploaded {total_uploaded} file(s) to volume.")
    else:
        print("✅ No uploads needed; volume already up to date.")

@app.local_entrypoint()
def main():
    """Run the comprehensive model download."""
    download_models.remote()

@app.local_entrypoint()
def upload_uploads():
    """Upload local uploads (e.g., loras) to the models volume."""
    upload_local_uploads()

if __name__ == "__main__":
    # When called directly (not as Modal function), just verify models exist
    # and skip download since models are already in the volume
    print("🔍 Checking if models are available...")
    
    essential_checks = [
        ("/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers-bf16", "Wan-AI 2.2 Text-to-Video Diffusers"),
        ("/models/Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan-AI 2.1 Text-to-Video Diffusers"),
        ("/models/vae/wan_2.2_vae.safetensors", "Wan 2.2 VAE"),
        ("/models/upscale_models/4x_NMKD-Siax_200k.pth", "4x NMKD-Siax Upscaler"),
        ("/models/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors", "WanVideo T2V 14B Light x2v CFG Step Distill LoRA"),
        ("/models/loras/Wan2.1_T2V_14B_FusionX_LoRA.safetensors", "Wan2.1 T2V 14B FusionX LoRA"),
    ]
    
    all_ready = True
    for path, component in essential_checks:
        if os.path.exists(path):
            if os.path.isfile(path):
                size_gb = os.path.getsize(path) / (1024*1024*1024)
                print(f"✅ {component}: {size_gb:.1f}GB")
            else:
                files = len([f for f in os.listdir(path) 
                           if f.endswith(('.safetensors', '.bin', '.json'))])
                print(f"✅ {component}: {files} files")
        else:
            print(f"⚠️ {component}: Not found (may download on first use)")
            # Don't mark as not ready since models can be downloaded later
    
    print(f"\n✅ Model check completed!") 