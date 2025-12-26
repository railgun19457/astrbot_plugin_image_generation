from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, GenerationResult, ImageCapability


class OpenAIAdapter(BaseImageAdapter):
    """标准 OpenAI 图像生成适配器 (DALL-E)。"""

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return ImageCapability.TEXT_TO_IMAGE | ImageCapability.ASPECT_RATIO

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行生图逻辑。"""
        if not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

        last_error = "未配置 API Key"
        for attempt in range(self.max_retry_attempts):
            if attempt:
                logger.info(
                    f"{self._get_log_prefix(request.task_id)} 重试适配器 ({attempt + 1}/{self.max_retry_attempts})"
                )

            images, err = await self._generate_once(request)
            if images is not None:
                return GenerationResult(images=images, error=None)

            last_error = err or "生成失败"
            if attempt < self.max_retry_attempts - 1:
                self._rotate_api_key()
                if (attempt + 1) % max(1, len(self.api_keys)) == 0:
                    await asyncio.sleep(
                        min(2 ** ((attempt + 1) // len(self.api_keys)), 10)
                    )

        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)
        if request.images:
            logger.warning(
                f"{prefix} 适配器目前不支持参考图 (Image-to-Image)，将仅使用提示词生成"
            )

        payload = self._build_payload(request)
        session = self._get_session()

        if not self.base_url:
            url = "https://api.openai.com/v1/images/generations"
        else:
            # 考虑到 main.py 会清理掉 /v1，这里统一加上
            url = f"{self.base_url.rstrip('/')}/v1/images/generations"

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self.timeout,
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): {error_text}"
                    )
                    return None, f"API 错误 ({resp.status})"

                data = await resp.json()
                logger.info(
                    f"{prefix} 生成成功 (耗时: {duration:.2f}s)"
                )
                return await self._extract_images(data)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}"
            )
            return None, str(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""
        # OpenAI DALL-E 3 常用参数
        # 映射分辨率
        size = "1024x1024"
        if request.aspect_ratio == "9:16":
            size = "1024x1792"
        elif request.aspect_ratio == "16:9":
            size = "1792x1024"

        # 注意：DALL-E 3 仅支持 1024x1024, 1024x1792, 1792x1024
        # 如果用户指定了其他比例，我们尽量匹配或保持默认

        payload: dict[str, Any] = {
            "model": self.model or "dall-e-3",
            "prompt": request.prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }

        return payload

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """从响应中提取图片数据。"""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # 如果返回的是 URL，需要下载（虽然我们请求的是 b64_json）
                async with self._get_session().get(
                    item["url"], proxy=self.proxy
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的图片数据"

        return images, None
