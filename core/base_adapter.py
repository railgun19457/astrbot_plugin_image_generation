"""
Base adapter abstract class for image generation
图像生成适配器的抽象基类
"""

from __future__ import annotations

import abc
import asyncio

import aiohttp

from astrbot.api import logger

from .types import (
    AdapterConfig,
    AdapterMetadata,
    GenerationRequest,
    GenerationResult,
)


class BaseImageAdapter(abc.ABC):
    """图像生成适配器基类"""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.current_key_index = 0
        self._session: aiohttp.ClientSession | None = None

    @abc.abstractmethod
    def metadata(self) -> AdapterMetadata:
        """返回适配器元数据"""
        pass

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        生成图像的核心方法

        Args:
            request: 生成请求参数

        Returns:
            GenerationResult: 生成结果
        """
        pass

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        """关闭 aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key"""
        return self.config.get_current_key(self.current_key_index)

    def _rotate_api_key(self):
        """切换到下一个 API Key"""
        if len(self.config.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(
                self.config.api_keys
            )
            logger.info(
                f"[{self.metadata().name}] 切换到下一个 API Key (索引: {self.current_key_index})"
            )

    def _mask_api_key(self, api_key: str) -> str:
        """遮蔽 API Key 用于日志输出"""
        if len(api_key) > 8:
            return api_key[:4] + "****" + api_key[-4:]
        return "****"

    async def generate_with_retry(self, request: GenerationRequest) -> GenerationResult:
        """
        带重试机制的图像生成

        Args:
            request: 生成请求参数

        Returns:
            GenerationResult: 生成结果
        """
        if not self.config.has_api_keys:
            return GenerationResult(success=False, error_message="未配置 API Key")

        prefix = f"[{request.task_id}] " if request.task_id else ""
        last_error = "未知错误"

        for attempt in range(self.config.max_retry_attempts):
            if attempt > 0:
                logger.info(
                    f"[{self.metadata().name}] {prefix}重试生成 "
                    f"(第 {attempt + 1}/{self.config.max_retry_attempts} 次)"
                )

            try:
                result = await self.generate(request)
                if result.success:
                    return result
                last_error = result.error_message or "生成失败"
            except asyncio.TimeoutError:
                last_error = "生成超时"
                logger.warning(f"[{self.metadata().name}] {prefix}生成超时")
            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"[{self.metadata().name}] {prefix}生成异常: {e}",
                    exc_info=True,
                )

            # 如果还有重试机会，进行 API Key 轮换和退避等待
            if attempt < self.config.max_retry_attempts - 1:
                if len(self.config.api_keys) > 1:
                    self._rotate_api_key()

                # 指数退避：每轮完整的 Key 轮换后等待
                if (attempt + 1) % len(self.config.api_keys) == 0:
                    round_index = (attempt + 1) // len(self.config.api_keys) - 1
                    wait_time = min(2**round_index, 10)
                    await asyncio.sleep(wait_time)

        return GenerationResult(success=False, error_message=f"重试失败: {last_error}")

    def _log_request(self, endpoint: str, payload: dict):
        """记录请求信息（隐藏敏感数据）"""
        try:

            def truncate(obj):
                if isinstance(obj, dict):
                    return {k: truncate(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [truncate(i) for i in obj]
                elif isinstance(obj, str):
                    if len(obj) > 200:
                        return obj[:50] + f"...({len(obj)} chars)..." + obj[-20:]
                    return obj
                else:
                    return obj

            log_payload = truncate(payload)
            logger.debug(
                f"[{self.metadata().name}] Request to {endpoint}: {log_payload}"
            )
        except Exception:
            pass
