from __future__ import annotations

from astrbot.api import logger

from ..adapter import (
    GeminiAdapter,
    GeminiOpenAIAdapter,
    GeminiZaiAdapter,
    OpenAIAdapter,
    ZImageAdapter,
)
from .types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageData,
)
from .utils import convert_images_batch


class ImageGenerator:
    """适配器编排器，负责分发生图请求。"""

    def __init__(self, adapter_config: AdapterConfig):
        self.adapter_config = adapter_config
        self.adapter = self._create_adapter(adapter_config)

    def _create_adapter(self, config: AdapterConfig):
        """根据配置创建对应的适配器。"""
        adapter_map: dict[AdapterType, type] = {
            AdapterType.GEMINI: GeminiAdapter,
            AdapterType.GEMINI_OPENAI: GeminiOpenAIAdapter,
            AdapterType.GEMINI_ZAI: GeminiZaiAdapter,
            AdapterType.OPENAI: OpenAIAdapter,
            AdapterType.Z_IMAGE: ZImageAdapter,
        }

        adapter_cls = adapter_map.get(config.type)
        if not adapter_cls:
            raise ValueError(f"不支持的适配器类型: {config.type}")
        return adapter_cls(config)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行生图逻辑。"""
        if not self.adapter:
            return GenerationResult(images=None, error="适配器未初始化")

        # 先将参考图批量转换成兼容格式，再调用下游适配器
        converted_images: list[ImageData] = []
        if request.images:
            converted_images = await convert_images_batch(request.images)

        patched_request = GenerationRequest(
            prompt=request.prompt,
            images=converted_images,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            task_id=request.task_id,
        )

        try:
            return await self.adapter.generate(patched_request)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] 生成失败: {exc}", exc_info=True)
            return GenerationResult(images=None, error=str(exc))

    def update_model(self, model: str) -> None:
        """更新适配器使用的模型。"""
        if self.adapter:
            self.adapter.update_model(model)

    async def close(self) -> None:
        """关闭适配器。"""
        if self.adapter:
            await self.adapter.close()
