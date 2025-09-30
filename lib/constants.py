"""Constants and configuration for the inference system."""

# GPU Configuration
DEFAULT_GPU_TYPE = "H200"
DEV_SERVER_GPU_TYPE = "H100"
GPU_TYPES = ["H200", "H100"]

# Server Configuration
COMFYUI_PORT = 8000
DEV_SERVER_COMFYUI_PORT = 8001
DEV_SERVER_PROXY_PORT = 8000

# Timeouts (seconds)
COMFYUI_STARTUP_TIMEOUT = 30
WEBSOCKET_PING_INTERVAL = 30
WEBSOCKET_PING_TIMEOUT = 20
WEBSOCKET_CLOSE_TIMEOUT = 10
WEBSOCKET_OPEN_TIMEOUT = 30
HTTP_REQUEST_TIMEOUT = 10
INFERENCE_TIMEOUT = 600
RELAY_MAX_RUNTIME = 600

# WebSocket Configuration
WEBSOCKET_MAX_MESSAGE_SIZE = 2 * 1024 * 1024  # 2MB
WEBSOCKET_RECONNECT_MAX_ATTEMPTS = 3
WEBSOCKET_RECONNECT_DELAY = 2  # seconds

# Image Processing
MAX_PREVIEW_SIZE = 50 * 1024  # 50KB
PREVIEW_THUMBNAIL_SIZE = (320, 320)
PREVIEW_THUMBNAIL_QUALITY = 60
PREVIEW_FALLBACK_SIZE = (256, 256)
PREVIEW_FALLBACK_QUALITY = 45
IMAGE_BINARY_HEADER_SIZE = 8

# File System Paths
MODELS_PATH = "/models"
COMFYUI_PATH = "/root/comfy/ComfyUI"
COMFYUI_OUTPUT_DIR = f"{COMFYUI_PATH}/output"
LORA_MODELS_DIR = f"{COMFYUI_PATH}/models/loras"
WORKFLOWS_MOUNT_PATH = "/workflows"
WORKFLOWS_BAKED_PATH = "/root/workflows"
S3_MOUNT_PATH = "/data"

# Progress Updates
PROGRESS_THROTTLE_SECONDS = 1.5
RELAY_MEMORY_CLEANUP_INTERVAL = 50  # messages
RELAY_CLEANUP_CHECK_INTERVAL = 300  # seconds (5 minutes)
RELAY_MAX_AGE_MINUTES = 15

# Retry Configuration
S3_UPLOAD_MAX_RETRIES = 3
EDGE_FUNCTION_MAX_RETRIES = 3
DIRECTORY_SCAN_MAX_ATTEMPTS = 10
DIRECTORY_SCAN_DELAY = 1.0
DIRECTORY_SCAN_BACKOFF = 1.5
DIRECTORY_SCAN_MAX_DELAY = 3.0

# Image Formats
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.webp', '.jpg', '.jpeg')
IMAGE_FORMAT_HEADERS = {
    'png': b'\x89PNG\r\n\x1a\n',
    'jpeg': b'\xff\xd8\xff',
    'webp_prefix': b'RIFF',
    'webp_marker': b'WEBP'
}

# Workflow Configuration
DEFAULT_SEED_MAX = 2**32 - 1
DEFAULT_NB_TAKES = 1
DEFAULT_QUALITY = "1K"
DEFAULT_ASPECT_RATIO = "1:1"

# ComfyUI Node Types
LATENT_NODE_TYPES = [
    "EmptyHunyuanLatentVideo",
    "EmptyLatentImage", 
    "EmptySD3LatentImage",
    "EmptyLTXVLatentVideo"
]

SAMPLER_NODE_TYPES = [
    "KSampler",
    "KSamplerWithNAG",
    "KSamplerAdvanced"
]

SAVE_NODE_TYPES = [
    "SaveImage",
    "SaveImagePlus"
]

IMAGE_PRODUCING_NODE_TYPES = [
    "Decode", "Preview", "Image", 
    "Upscale", "Scale", "Resize"
]

# Environment Variables
ENV_TAGS = ["dev", "staging", "prod"]
DEFAULT_ENV_TAG = "dev"

# Modal Configuration
MODAL_MAX_CONTAINERS = 40
MODAL_RETRIES = 3
MODAL_SCALEDOWN_WINDOW = 300  # seconds
MODAL_TIMEOUT = 30000  # seconds

# Logging
LOG_LEVEL_DEFAULT = "WARNING"
