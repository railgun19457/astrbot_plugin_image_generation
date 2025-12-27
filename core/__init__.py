"""
Core module for image generation plugin
图像生成插件的核心模块
"""

from .base_adapter import BaseImageAdapter
from .constants import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    DEFAULT_RESOLUTION,
    DEFAULT_TIMEOUT,
    GEMINI_DEFAULT_BASE_URL,
    GEMINI_SAFETY_CATEGORIES,
    GITEE_AI_DEFAULT_BASE_URL,
    JIMENG_DEFAULT_BASE_URL,
    LOG_PREFIX,
    OPENAI_DEFAULT_BASE_URL,
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_RESOLUTIONS,
)
from .generator import ImageGenerator
from .task_manager import TaskManager
from .types import (
    AdapterConfig,
    AdapterMetadata,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageCapability,
    ImageData,
)
from .utils import (
    convert_image_format,
    convert_images_batch,
    detect_mime_type,
    validate_aspect_ratio,
    validate_resolution,
)

__all__ = [
    # 基类和核心组件
    "BaseImageAdapter",
    "ImageGenerator",
    "TaskManager",
    # 数据类型
    "AdapterConfig",
    "AdapterMetadata",
    "AdapterType",
    "GenerationRequest",
    "GenerationResult",
    "ImageCapability",
    "ImageData",
    # 工具函数
    "convert_image_format",
    "convert_images_batch",
    "detect_mime_type",
    "validate_aspect_ratio",
    "validate_resolution",
    # 常量
    "LOG_PREFIX",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_RESOLUTION",
    "GEMINI_DEFAULT_BASE_URL",
    "GEMINI_SAFETY_CATEGORIES",
    "OPENAI_DEFAULT_BASE_URL",
    "GITEE_AI_DEFAULT_BASE_URL",
    "JIMENG_DEFAULT_BASE_URL",
    "RESOLUTION_1K_MAP",
    "RESOLUTION_2K_MAP",
    "SUPPORTED_ASPECT_RATIOS",
    "SUPPORTED_RESOLUTIONS",
]
