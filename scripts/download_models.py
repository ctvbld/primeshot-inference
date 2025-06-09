#!/usr/bin/env python3
"""
Comprehensive model download script for ComfyUI Photography Platform.
Downloads all required models with existence checks to avoid redundant downloads.
"""

import modal
import os
from pathlib import Path

models_volume = modal.Volume.from_name("models-vol")
app = modal.App("download-all-models")



def download_vae_model():
    """Download VAE model files individually - this actually works."""
    from huggingface_hub import hf_hub_download
    
    # Download the main model file
    hf_hub_download(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="diffusion_pytorch_model.safetensors",
        local_dir="/models/vae/sdxl-vae-fp16-fix"
    )
    
    # Download config
    hf_hub_download(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="config.json",
        local_dir="/models/vae/sdxl-vae-fp16-fix"
    )

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
    """Download all required models with existence checks."""
    from huggingface_hub import hf_hub_download, snapshot_download
    import shutil
    import os
    
    # Enable faster downloads for large files
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    print("🚀 ComfyUI Model Download - Checking and downloading missing models...")
    print("⚡ HF_TRANSFER enabled for faster large file downloads")
    
    # Clean any corrupted cache first
    print("🧹 Cleaning corrupted cache files...")
    import subprocess
    subprocess.run(["find", "/models", "-name", "*.incomplete", "-delete"], capture_output=True)
    subprocess.run(["find", "/models", "-name", "*.lock", "-delete"], capture_output=True)
    
    # Create all necessary directories
    directories = [
        "/models/checkpoints",
        "/models/unet",
        "/models/upscale_models", 
        "/models/clip",
        "/models/vae",
        "/models/loras"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Define all models to download with existence checks
    models_to_download = [
        {
            "name": "Flux.1-dev FP8 (Essential - 12GB)",
            "check_path": "/models/checkpoints/flux1-dev-fp8.safetensors", 
            "download_func": lambda: hf_hub_download(
                repo_id="Kijai/flux-fp8",
                filename="flux1-dev-fp8.safetensors",
                local_dir="/models/checkpoints",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "Flux.1-dev UNET (Essential - 23.8GB)",
            "check_path": "/models/unet/flux1-dev.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                filename="flux1-dev.safetensors",
                local_dir="/models/unet",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "Flux.1-dev FP8 UNET (Optimized - 12GB)",
            "check_path": "/models/unet/flux1-dev-fp8.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="Kijai/flux-fp8",
                filename="flux1-dev-fp8.safetensors",
                local_dir="/models/unet",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "SUPIR Upscaler (Essential - 2GB)",
            "check_path": "/models/checkpoints/SUPIR-v0Q_fp16.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="Kijai/SUPIR_pruned", 
                filename="SUPIR-v0Q_fp16.safetensors",
                local_dir="/models/checkpoints",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "JuggernautXL Photorealistic Model (Required for SUPIR - 7.1GB)",
            "check_path": "/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="RunDiffusion/Juggernaut-XL-v9", 
                filename="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
                local_dir="/models/checkpoints",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "RealVisXL V5.0 - Realistic Faces Model (Optional - 6.9GB)",
            "check_path": "/models/checkpoints/RealVisXL_V5.0_fp16.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="SG161222/RealVisXL_V5.0", 
                filename="RealVisXL_V5.0_fp16.safetensors",
                local_dir="/models/checkpoints",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "CyberRealistic XL V5.7 - Hyper-realistic Model (Optional - 6.9GB)",
            "check_path": "/models/checkpoints/CyberRealisticXLPlay_V5.7.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="cyberdelia/CyberRealisticXL", 
                filename="CyberRealisticXLPlay_V5.7.safetensors",
                local_dir="/models/checkpoints",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "CLIP-L Text Encoder (Essential - 246MB)",
            "check_path": "/models/clip/clip_l.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                filename="clip_l.safetensors",
                local_dir="/models/clip",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "T5XXL FP8 Text Encoder (Essential - 4.89GB)",
            "check_path": "/models/clip/t5xxl_fp8_e4m3fn.safetensors",
            "download_func": lambda: hf_hub_download(
                repo_id="comfyanonymous/flux_text_encoders",
                filename="t5xxl_fp8_e4m3fn.safetensors",
                local_dir="/models/clip",
                local_dir_use_symlinks=False
            )
        },
        {
            "name": "VAE (Essential - 335MB)",
            "check_path": "/models/vae/sdxl-vae-fp16-fix",
            "check_files": ["diffusion_pytorch_model.safetensors", "config.json"],
            "download_func": lambda: download_vae_model()
        }
    ]
    
    # Download each model with existence check
    downloaded_count = 0
    skipped_count = 0
    
    for model in models_to_download:
        name = model["name"]
        check_path = model["check_path"]
        check_files = model.get("check_files", [])
        
        # Check if model already exists and is complete
        skip_download = False
        
        if os.path.exists(check_path):
            if os.path.isfile(check_path):
                # Single file check
                file_size = os.path.getsize(check_path) / (1024*1024*1024)
                if file_size > 0.01:  # At least 10MB
                    print(f"✅ SKIP: {name} already exists ({file_size:.1f}GB)")
                    skip_download = True
            elif os.path.isdir(check_path):
                # Directory with multiple files check - be more thorough
                if check_files:
                    # Check that ALL required files exist AND have reasonable sizes
                    all_files_exist = True
                    for required_file in check_files:
                        file_path = os.path.join(check_path, required_file)
                        if not os.path.exists(file_path):
                            print(f"⚠️ Missing required file: {required_file}")
                            all_files_exist = False
                            break
                        # Check file size - model files should be substantial
                        if required_file.endswith(('.bin', '.safetensors')):
                            size_mb = os.path.getsize(file_path) / (1024*1024)
                            if size_mb < 10:  # Model files should be at least 10MB
                                print(f"⚠️ File too small (possibly corrupted): {required_file} ({size_mb:.1f}MB)")
                                all_files_exist = False
                                break
                    
                    if all_files_exist:
                        total_files = len([f for f in os.listdir(check_path) 
                                         if f.endswith(('.safetensors', '.bin', '.json'))])
                        print(f"✅ SKIP: {name} complete ({total_files} files)")
                        skip_download = True
                    else:
                        print(f"🔄 INCOMPLETE: {name} - will re-download")
                        # Clean incomplete download
                        shutil.rmtree(check_path)
                else:
                    # Check if directory has any substantial model files
                    model_files = [f for f in os.listdir(check_path) 
                                 if f.endswith(('.safetensors', '.bin'))]
                    if model_files:
                        # Check if model files have reasonable sizes
                        total_size = sum(os.path.getsize(os.path.join(check_path, f)) 
                                       for f in model_files) / (1024*1024)
                        if total_size > 10:  # At least 10MB total
                            print(f"✅ SKIP: {name} already exists ({len(model_files)} model files, {total_size:.0f}MB)")
                            skip_download = True
                        else:
                            print(f"🔄 CORRUPTED: {name} - files too small, will re-download")
                            shutil.rmtree(check_path)
                    else:
                        print(f"🔄 EMPTY: {name} - no model files found, will re-download")
                        shutil.rmtree(check_path)
        
        if skip_download:
            skipped_count += 1
            continue
        
        # Download the model
        print(f"\n📦 DOWNLOADING: {name}")
        try:
            model["download_func"]()
            print(f"✅ SUCCESS: {name}")
            
            # Verify download actually worked
            if os.path.exists(check_path):
                if os.path.isfile(check_path):
                    size_gb = os.path.getsize(check_path) / (1024*1024*1024)
                    print(f"🔍 Verified: {name} - {size_gb:.1f}GB")
                elif os.path.isdir(check_path):
                    files = os.listdir(check_path)
                    total_size_mb = sum(
                        os.path.getsize(os.path.join(check_path, f)) / (1024*1024)
                        for f in files if os.path.isfile(os.path.join(check_path, f))
                    )
                    print(f"🔍 Verified: {name} - {len(files)} files, {total_size_mb:.1f}MB total")
                    
                    # Check specific required files
                    if check_files:
                        missing_files = []
                        for required_file in check_files:
                            file_path = os.path.join(check_path, required_file)
                            if not os.path.exists(file_path):
                                missing_files.append(required_file)
                            else:
                                size_mb = os.path.getsize(file_path) / (1024*1024)
                                print(f"  ✓ {required_file}: {size_mb:.1f}MB")
                        
                        if missing_files:
                            print(f"  ❌ Still missing: {missing_files}")
                        else:
                            print(f"  ✅ All required files present")
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
    print(f"⏭️ Skipped: {skipped_count} models (already existed)")
    
    print(f"\n🔍 FINAL VERIFICATION:")
    essential_checks = [
        ("/models/checkpoints/flux1-dev-fp8.safetensors", "Flux FP8 Checkpoint"),
        ("/models/unet/flux1-dev.safetensors", "Flux Dev UNET"),
        ("/models/unet/flux1-dev-fp8.safetensors", "Flux FP8 UNET"),
        ("/models/checkpoints/SUPIR-v0Q_fp16.safetensors", "SUPIR"),
        ("/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", "JuggernautXL"),
        ("/models/checkpoints/RealVisXL_V5.0_fp16.safetensors", "RealVisXL V5.0"),
        ("/models/checkpoints/CyberRealisticXLPlay_V5.7.safetensors", "CyberRealistic XL V5.7"),
        ("/models/clip/clip_l.safetensors", "CLIP-L"),
        ("/models/clip/t5xxl_fp8_e4m3fn.safetensors", "T5XXL FP8"),
        ("/models/vae/sdxl-vae-fp16-fix", "VAE")
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

@app.local_entrypoint()
def main():
    """Run the comprehensive model download."""
    download_models.remote()

if __name__ == "__main__":
    # When called directly (not as Modal function), just verify models exist
    # and skip download since models are already in the volume
    print("🔍 Checking if models are available...")
    
    essential_checks = [
        ("/models/checkpoints/flux1-dev-fp8.safetensors", "Flux FP8 Checkpoint"),
        ("/models/unet/flux1-dev.safetensors", "Flux Dev UNET"),
        ("/models/unet/flux1-dev-fp8.safetensors", "Flux FP8 UNET"),
        ("/models/checkpoints/SUPIR-v0Q_fp16.safetensors", "SUPIR"),
        ("/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", "JuggernautXL"),
        ("/models/checkpoints/RealVisXL_V5.0_fp16.safetensors", "RealVisXL V5.0"),
        ("/models/checkpoints/cyberrealistic-xl-v31-sdxl.safetensors", "CyberRealistic XL v3.1"),
        ("/models/clip/clip_l.safetensors", "CLIP-L"),
        ("/models/clip/t5xxl_fp8_e4m3fn.safetensors", "T5XXL FP8"),
        ("/models/vae/sdxl-vae-fp16-fix", "VAE")
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