from __future__ import annotations

import abc

import aiohttp

from astrbot.api import logger

from .types import AdapterConfig, GenerationRequest, GenerationResult, ImageCapability


class BaseImageAdapter(abc.ABC):
    """图像生成适配器基类。"""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.api_keys = config.api_keys or []
        self.current_key_index = 0
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.proxy = config.proxy
        self.timeout = config.timeout
        self.max_retry_attempts = max(1, config.max_retry_attempts)
        self.safety_settings = config.safety_settings
        self._session: aiohttp.ClientSession | None = None

    @abc.abstractmethod
    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""

    async def close(self) -> None:
        """关闭底层的 HTTP 会话。"""

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话。"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key。"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _get_log_prefix(self, task_id: str | None = None) -> str:
        """获取统一的日志前缀。"""
        adapter_name = self.__class__.__name__.replace("Adapter", "")
        prefix = f"[ImageGen] [{adapter_name}]"
        if task_id:
            prefix += f" [{task_id}]"
        return prefix

    def _rotate_api_key(self) -> None:
        """轮换 API Key。"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(f"{self._get_log_prefix()} 轮换 API Key -> 索引 {self.current_key_index}")

    def update_model(self, model: str) -> None:
        """更新使用的模型。"""
        self.model = model

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """为给定的请求生成图像。"""
