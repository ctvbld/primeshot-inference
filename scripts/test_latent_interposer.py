#!/usr/bin/env python3
"""
Test script for SD-Latent-Interposer integration with SUPIR workflow.
Validates the experimental Flux→SDXL latent conversion pipeline.
"""

import modal
import json
import time
from typing import Dict, Any

# Import the Modal app from the main module
from comfyui_app import app

def test_latent_interposer_workflow():
    """Test the experimental latent interposer workflow."""
    
    # Test parameters
    test_request = {
        "user_id": "test_user",
        "workflow_name": "flux_lora_supir_interposer",
        "parameters": {
            "lora_path": "/data/test_user/loras/test_style.safetensors",
            "style_prompt": "professional portrait, high quality, sharp focus, studio lighting",
            "negative_prompt": "blurry, low quality, distorted, artifacts",
            "batch_size": 2,  # Small batch for testing
            "resolution": "512x512",  # Lower resolution for faster testing
            "steps": 20,  # Reduced steps for faster testing
            "cfg": 3.5,
            "lora_strength": 1.0,
            "upscale_factor": 2.0,
            "supir_steps": 15,  # Reduced for testing
            "supir_cfg": 4.0,
            "supir_denoise_strength": 40,
            "use_tiled_processing": true
        }
    }
    
    print("🧪 Testing SD-Latent-Interposer Integration")
    print("=" * 50)
    print(f"Workflow: {test_request['workflow_name']}")
    print(f"Resolution: {test_request['parameters']['resolution']}")
    print(f"Batch size: {test_request['parameters']['batch_size']}")
    print(f"Upscale factor: {test_request['parameters']['upscale_factor']}")
    print()
    
    # Test workflow configuration loading
    try:
        comfy_instance = app.cls()
        
        # Test workflow configuration
        print("🔧 Testing workflow configuration...")
        workflow_config = comfy_instance.get_workflow_config()
        
        if "flux_lora_supir_interposer" in workflow_config["workflows"]:
            print("✅ Latent interposer workflow found in configuration")
        else:
            print("❌ Latent interposer workflow not found in configuration")
            return False
        
        # Test workflow template loading
        print("📄 Testing workflow template loading...")
        workflow = comfy_instance.load_workflow_template("flux_lora_supir_interposer")
        
        # Validate critical nodes exist
        required_nodes = [
            "1",   # UNETLoader (Flux)
            "10",  # SD_Latent_Interposer
            "11",  # VAELoader (SDXL)
            "12",  # SUPIR_model_loader_v2
            "17",  # SUPIR_first_stage
            "14",  # SUPIR_sample
            "15",  # SUPIR_decode
        ]
        
        for node_id in required_nodes:
            if node_id in workflow:
                node_type = workflow[node_id]["class_type"]
                print(f"✅ Node {node_id} ({node_type}) found")
            else:
                print(f"❌ Critical node {node_id} missing")
                return False
        
        # Test parameter validation
        print("🔍 Testing parameter validation...")
        comfy_instance.validate_workflow_request("flux_lora_supir_interposer", test_request["parameters"])
        print("✅ Parameter validation passed")
        
        print()
        print("🎯 Integration Test Summary:")
        print("✅ Workflow configuration loaded successfully")
        print("✅ All critical nodes present in workflow")
        print("✅ Parameter validation passed")
        print("✅ SD-Latent-Interposer integration ready for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_performance_comparison():
    """Compare performance metrics between standard and interposer workflows."""
    
    print("\n📊 Performance Comparison Framework")
    print("=" * 50)
    print("To compare standard vs interposer workflows:")
    print()
    print("1. **Quality Metrics**:")
    print("   - Run both workflows with identical parameters")
    print("   - Compare LPIPS, SSIM, and PSNR scores")
    print("   - Visual quality assessment")
    print()
    print("2. **Performance Metrics**:")
    print("   - Execution time comparison")
    print("   - Memory usage analysis")
    print("   - GPU utilization patterns")
    print()
    print("3. **Latent Space Analysis**:")
    print("   - Compare Flux latents vs converted SDXL latents")
    print("   - Analyze conversion artifacts")
    print("   - Monitor color/saturation shifts")
    print()
    print("4. **Test Parameters**:")
    print("   - Standard: flux_lora_supir")
    print("   - Experimental: flux_lora_supir_interposer")
    print("   - Use identical LoRAs, prompts, and settings")

if __name__ == "__main__":
    print("🚀 SD-Latent-Interposer Integration Test")
    print("Testing experimental Flux→SDXL latent conversion")
    print()
    
    # Run integration test
    success = test_latent_interposer_workflow()
    
    if success:
        print("\n🎉 Integration test passed!")
        print("The latent interposer workflow is ready for experimental testing.")
        test_performance_comparison()
    else:
        print("\n💥 Integration test failed!")
        print("Please check the workflow configuration and node mappings.") 