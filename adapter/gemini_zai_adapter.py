"""
Zai API adapter for image generation
Zai API 图像生成适配器（OpenAI 格式变体）
"""

from __future__ import annotations

import base64

import aiohttp

from astrbot.api import logger

from ..core import (
    AdapterMetadata,
    GenerationRequest,
)
from .gemini_openai_adapter import GeminiOpenAIAdapter


class GeminiZaiAdapter(GeminiOpenAIAdapter):
    """Gemini Zai API 图像生成适配器（继承自 Gemini OpenAI 适配器）"""

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="zai",
            display_name="Zai",
            description="Zai 图像生成 API（OpenAI 兼容格式）",
            supported_features=[
                "text_to_image",
                "image_to_image",
                "aspect_ratio",
                "resolution",
            ],
            default_models=[
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.5-flash-image",
                "gemini-2.5-flash-image-preview",
            ],
            supports_image_to_image=True,
            supports_aspect_ratio=True,
            supports_resolution=True,
            supports_safety_settings=False,
        )

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建 Zai 请求负载（OpenAI 格式 + params）"""
        # Zai 不需要 "Generate an image: " 前缀，直接使用 prompt
        message_content = [{"type": "text", "text": request.prompt}]

        # 处理参考图（OpenAI Vision 格式）
        if request.reference_images:
            for image_data in request.reference_images:
                b64_data = base64.b64encode(image_data.data).decode("utf-8")
                image_url = f"data:{image_data.mime_type};base64,{b64_data}"
                message_content.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": message_content}],
            "stream": False,
        }

        # Zai 特有参数
        params = {}
        if request.aspect_ratio and not request.is_image_to_image:
            params["image_aspect_ratio"] = request.aspect_ratio
        if request.resolution:
            params["image_resolution"] = request.resolution

        if params:
            payload["params"] = params

        self._log_request("Zai", payload)
        return payload

    async def _make_request(
        self, session: aiohttp.ClientSession, payload: dict, task_id: str | None = None
    ) -> dict | None:
        """发送 Zai API 请求（使用 OpenAI 兼容端点）"""
        prefix = f"[{task_id}] " if task_id else ""
        url = f"{self.config.base_url}/v1/chat/completions"

        api_key = self._get_current_api_key()
        masked_key = self._mask_api_key(api_key)
        logger.debug(f"[Zai] {prefix}Request URL: {url}, Key: {masked_key}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                proxy=self.config.proxy,
            ) as response:
                logger.debug(f"[Zai] {prefix}API Response Status: {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    error_preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"[Zai] {prefix}API 错误: {response.status} - {error_preview}"
                    )
                    return None

                return await response.json()

        except Exception as e:
            logger.error(f"[Zai] {prefix}请求异常: {e}")
            return None
