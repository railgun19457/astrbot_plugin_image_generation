"""
Core type definitions for image generation plugin
定义图像生成插件的核心数据类型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AdapterType(str, Enum):
    """适配器类型枚举"""

    GEMINI = "gemini"
    OPENAI = "openai"
    ZAI = "zai"
    DALLE = "dalle"
    STABLE_DIFFUSION = "stable_diffusion"
    MIDJOURNEY = "midjourney"


@dataclass
class ImageData:
    """图片数据封装"""

    data: bytes
    mime_type: str = "image/jpeg"

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class GenerationRequest:
    """图像生成请求"""

    prompt: str
    reference_images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = "1:1"
    resolution: str | None = "1K"
    task_id: str | None = None

    # 扩展参数，不同适配器可能有不同的参数
    extra_params: dict = field(default_factory=dict)

    @property
    def is_image_to_image(self) -> bool:
        """是否为图生图模式"""
        return len(self.reference_images) > 0


@dataclass
class GenerationResult:
    """图像生成结果"""

    success: bool
    images: list[bytes] = field(default_factory=list)
    error_message: str | None = None

    @property
    def image_count(self) -> int:
        """生成的图片数量"""
        return len(self.images)


@dataclass
class AdapterConfig:
    """适配器配置"""

    adapter_type: AdapterType
    api_keys: list[str] = field(default_factory=list)
    base_url: str = ""
    model: str = ""
    timeout: int = 300
    max_retry_attempts: int = 3
    proxy: str | None = None

    # 适配器特定配置
    extra_config: dict = field(default_factory=dict)

    @property
    def has_api_keys(self) -> bool:
        """是否配置了API密钥"""
        return len(self.api_keys) > 0

    def get_current_key(self, index: int = 0) -> str:
        """获取指定索引的API密钥"""
        if not self.has_api_keys:
            return ""
        return self.api_keys[index % len(self.api_keys)]


@dataclass
class AdapterMetadata:
    """适配器元数据"""

    name: str
    display_name: str
    description: str
    supported_features: list[str] = field(default_factory=list)
    default_models: list[str] = field(default_factory=list)

    # 支持的功能标志
    supports_image_to_image: bool = True
    supports_aspect_ratio: bool = True
    supports_resolution: bool = False
    supports_safety_settings: bool = False
