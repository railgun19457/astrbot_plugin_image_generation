"""
Gemini API adapter for image generation
Gemini API 图像生成适配器
"""

from __future__ import annotations

import base64

import aiohttp

from astrbot.api import logger

from ..core import (
    AdapterConfig,
    AdapterMetadata,
    BaseImageAdapter,
    GenerationRequest,
    GenerationResult,
)


class GeminiAdapter(BaseImageAdapter):
    """Gemini 图像生成适配器"""

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.safety_settings = config.extra_config.get("safety_settings", "BLOCK_NONE")

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="gemini",
            display_name="Gemini",
            description="Google Gemini 图像生成模型",
            supported_features=[
                "text_to_image",
                "image_to_image",
                "aspect_ratio",
                "resolution",
                "safety_settings",
            ],
            default_models=[
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.5-flash-image",
                "gemini-2.5-flash-image-preview",
                "gemini-3-pro-image-preview",
            ],
            supports_image_to_image=True,
            supports_aspect_ratio=True,
            supports_resolution=True,
            supports_safety_settings=True,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        使用 Gemini API 生成图像

        Args:
            request: 生成请求

        Returns:
            GenerationResult: 生成结果
        """
        prefix = f"[{request.task_id}] " if request.task_id else ""

        try:
            payload = self._build_payload(request)
            session = self._get_session()
            response_data = await self._make_request(session, payload, request.task_id)

            if response_data is None:
                return GenerationResult(success=False, error_message="API 请求失败")

            images = self._extract_images(response_data, request.task_id)
            if images:
                return GenerationResult(success=True, images=images)

            return GenerationResult(success=False, error_message="响应中未找到图片数据")

        except Exception as e:
            logger.error(f"[Gemini] {prefix}生成失败: {e}", exc_info=True)
            return GenerationResult(success=False, error_message=f"生成失败: {str(e)}")

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建 Gemini API 请求负载"""
        generation_config = {"responseModalities": ["IMAGE"]}
        image_config = {}

        # 如果有参考图，则不传 aspect_ratio，使用参考图的比例
        if request.aspect_ratio and not request.is_image_to_image:
            image_config["aspectRatio"] = request.aspect_ratio

        # imageSize 仅 gemini-3-pro-image-preview 支持
        if request.resolution and "gemini-3" in self.config.model.lower():
            image_config["imageSize"] = request.resolution

        if image_config:
            generation_config["imageConfig"] = image_config

        # 安全设置
        safety_settings = []
        if self.safety_settings:
            for category in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
                "HARM_CATEGORY_CIVIC_INTEGRITY",
            ]:
                safety_settings.append(
                    {"category": category, "threshold": self.safety_settings}
                )

        # 构建内容部分
        parts = [{"text": request.prompt}]

        # 添加参考图片
        if request.reference_images:
            for image_data in request.reference_images:
                encoded_data = base64.b64encode(image_data.data).decode("utf-8")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": image_data.mime_type,
                            "data": encoded_data,
                        }
                    }
                )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": safety_settings,
        }

        self._log_request("Gemini API", payload)
        return payload

    async def _make_request(
        self, session: aiohttp.ClientSession, payload: dict, task_id: str | None = None
    ) -> dict | None:
        """发送 Gemini API 请求"""
        prefix = f"[{task_id}] " if task_id else ""
        url = (
            f"{self.config.base_url}/v1beta/models/{self.config.model}:generateContent"
        )

        api_key = self._get_current_api_key()
        masked_key = self._mask_api_key(api_key)
        logger.debug(f"[Gemini] {prefix}Request URL: {url}, Key: {masked_key}")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                proxy=self.config.proxy,
            ) as response:
                logger.debug(f"[Gemini] {prefix}API Response Status: {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    error_preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"[Gemini] {prefix}API 错误: {response.status} - {error_preview}"
                    )
                    return None

                return await response.json()

        except Exception as e:
            logger.error(f"[Gemini] {prefix}请求异常: {e}")
            return None

    def _extract_images(
        self, response: dict, task_id: str | None = None
    ) -> list[bytes] | None:
        """从 Gemini API 响应中提取图片数据"""
        prefix = f"[{task_id}] " if task_id else ""

        try:
            candidates = response.get("candidates", [])
            logger.debug(f"[Gemini] {prefix}Candidates count: {len(candidates)}")

            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            images = []

            for part in parts:
                inline_data = part.get("inline_data") or part.get("inlineData")
                if inline_data:
                    data = inline_data.get("data")
                    if data:
                        images.append(base64.b64decode(data))

            return images if images else None

        except Exception as e:
            logger.error(f"[Gemini] {prefix}解析响应失败: {e}")
            return None
