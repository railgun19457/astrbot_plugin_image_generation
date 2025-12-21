"""
Core generator manager for image generation
图像生成的核心管理器
"""

from __future__ import annotations

from astrbot.api import logger

from .base_adapter import BaseImageAdapter
from .types import AdapterType, GenerationRequest, GenerationResult


class ImageGenerator:
    """图像生成管理器"""

    def __init__(self):
        self._adapters: dict[AdapterType, BaseImageAdapter] = {}
        self._current_adapter: BaseImageAdapter | None = None

    def register_adapter(self, adapter: BaseImageAdapter):
        """
        注册适配器

        Args:
            adapter: 适配器实例
        """
        adapter_type = adapter.metadata().name
        self._adapters[adapter_type] = adapter
        logger.info(f"[ImageGenerator] 注册适配器: {adapter.metadata().display_name}")

        # 如果是第一个适配器，设置为当前适配器
        if self._current_adapter is None:
            self._current_adapter = adapter

    def set_current_adapter(self, adapter_type: str | AdapterType):
        """
        设置当前使用的适配器

        Args:
            adapter_type: 适配器类型
        """
        if isinstance(adapter_type, str):
            # 尝试从字符串转换为 AdapterType
            for adapter_name, adapter in self._adapters.items():
                if (
                    adapter_name == adapter_type
                    or adapter.metadata().name == adapter_type
                ):
                    self._current_adapter = adapter
                    logger.info(
                        f"[ImageGenerator] 切换到适配器: {adapter.metadata().display_name}"
                    )
                    return

            logger.warning(f"[ImageGenerator] 未找到适配器: {adapter_type}")
        else:
            if adapter_type in self._adapters:
                self._current_adapter = self._adapters[adapter_type]
                logger.info(
                    f"[ImageGenerator] 切换到适配器: {self._current_adapter.metadata().display_name}"
                )
            else:
                logger.warning(f"[ImageGenerator] 未找到适配器: {adapter_type}")

    def get_adapter(
        self, adapter_type: str | AdapterType
    ) -> BaseImageAdapter | None:
        """
        获取指定类型的适配器

        Args:
            adapter_type: 适配器类型

        Returns:
            Optional[BaseImageAdapter]: 适配器实例，如果不存在则返回 None
        """
        if isinstance(adapter_type, str):
            for adapter_name, adapter in self._adapters.items():
                if (
                    adapter_name == adapter_type
                    or adapter.metadata().name == adapter_type
                ):
                    return adapter
            return None
        return self._adapters.get(adapter_type)

    @property
    def current_adapter(self) -> BaseImageAdapter | None:
        """获取当前适配器"""
        return self._current_adapter

    @property
    def available_adapters(self) -> list[str]:
        """获取所有可用适配器的名称列表"""
        return [adapter.metadata().display_name for adapter in self._adapters.values()]

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        生成图像

        Args:
            request: 生成请求

        Returns:
            GenerationResult: 生成结果
        """
        if self._current_adapter is None:
            return GenerationResult(success=False, error_message="未配置任何适配器")

        prefix = f"[{request.task_id}] " if request.task_id else ""
        logger.info(
            f"[ImageGenerator] {prefix}使用适配器 {self._current_adapter.metadata().display_name} 生成图像"
        )

        return await self._current_adapter.generate_with_retry(request)

    async def close_all_sessions(self):
        """关闭所有适配器的会话"""
        for adapter in self._adapters.values():
            try:
                await adapter.close_session()
            except Exception as e:
                logger.error(f"[ImageGenerator] 关闭适配器会话失败: {e}")
