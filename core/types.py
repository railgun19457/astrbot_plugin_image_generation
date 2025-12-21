from __future__ import annotations

import enum
from dataclasses import dataclass, field


class AdapterType(str, enum.Enum):
    """支持的图像生成适配器类型。"""

    GEMINI = "gemini"
    GEMINI_OPENAI = "gemini(OpenAI)"
    GEMINI_ZAI = "gemini(Zai)"
    OPENAI = "openai"
    Z_IMAGE = "z-image(gitee)"


@dataclass
class AdapterMetadata:
    """关于适配器能力的元数据。"""

    name: str
    supports_aspect_ratio: bool = True
    supports_resolution: bool = True


@dataclass
class AdapterConfig:
    """构造适配器所需的配置。"""

    type: AdapterType = AdapterType.GEMINI
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    available_models: list[str] = field(default_factory=list)
    provider_id: str | None = None
    proxy: str | None = None
    timeout: int = 180
    max_retry_attempts: int = 3
    safety_settings: str | None = None


@dataclass
class ImageData:
    """带有 MIME 类型的图像二进制数据。"""

    data: bytes
    mime_type: str


@dataclass
class GenerationRequest:
    """用户生图请求。"""

    prompt: str
    images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = None
    resolution: str | None = None
    task_id: str | None = None


@dataclass
class GenerationResult:
    """生图尝试的结果。"""

    images: list[bytes] | None = None
    error: str | None = None
