"""
Core module for image generation plugin
图像生成插件的核心模块
"""

from .base_adapter import BaseImageAdapter
from .generator import ImageGenerator
from .types import (
    AdapterConfig,
    AdapterMetadata,
    AdapterType,
    GenerationRequest,
    GenerationResult,
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
    "BaseImageAdapter",
    "ImageGenerator",
    "AdapterConfig",
    "AdapterMetadata",
    "AdapterType",
    "GenerationRequest",
    "GenerationResult",
    "ImageData",
    "convert_image_format",
    "convert_images_batch",
    "detect_mime_type",
    "validate_aspect_ratio",
    "validate_resolution",
]
