from __future__ import annotations

import abc
import asyncio

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

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """带重试逻辑的图像生成模板方法。
        
        子类应重写 `_generate_once()` 方法来实现具体的生成逻辑。
        如需在生成前进行预处理验证，可重写 `_pre_generate()` 方法。
        """
        if not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

        # 预处理检查（子类可重写）
        pre_result = self._pre_generate(request)
        if pre_result is not None:
            return pre_result

        last_error = "未配置 API Key"
        for attempt in range(self.max_retry_attempts):
            if attempt:
                logger.info(
                    f"{self._get_log_prefix(request.task_id)} 重试 ({attempt + 1}/{self.max_retry_attempts})"
                )

            images, err = await self._generate_once(request)
            if images is not None:
                return GenerationResult(images=images, error=None)

            last_error = err or "生成失败"
            if attempt < self.max_retry_attempts - 1:
                self._rotate_api_key()
                # 轮换 Key 时进行指数退避
                if (attempt + 1) % max(1, len(self.api_keys)) == 0:
                    await asyncio.sleep(
                        min(2 ** ((attempt + 1) // len(self.api_keys)), 10)
                    )

        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """生成前的预处理检查。
        
        子类可重写此方法进行参数验证。
        返回 None 表示通过检查，返回 GenerationResult 表示提前返回错误。
        """
        return None

    @abc.abstractmethod
    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生成请求。
        
        子类必须实现此方法。
        返回 (images, error) 元组，成功时 images 非空，失败时 error 非空。
        """
