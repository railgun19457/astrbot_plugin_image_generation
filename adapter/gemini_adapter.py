from __future__ import annotations

import base64
import time

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import GEMINI_DEFAULT_BASE_URL, GEMINI_SAFETY_CATEGORIES
from ..core.types import GenerationRequest, ImageCapability


class GeminiAdapter(BaseImageAdapter):
    """Gemini 原生图像生成适配器。"""

    DEFAULT_BASE_URL = GEMINI_DEFAULT_BASE_URL

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return (
            ImageCapability.TEXT_TO_IMAGE
            | ImageCapability.IMAGE_TO_IMAGE
            | ImageCapability.ASPECT_RATIO
            | ImageCapability.RESOLUTION
        )

    # generate() 方法由基类提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        payload = self._build_payload(request)
        session = self._get_session()
        response = await self._make_request(session, payload, request.task_id)
        if response is None:
            return None, "API 请求失败"

        images = self._extract_images(response, request.task_id)
        if images:
            return images, None
        return None, "响应中未找到图片数据"

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""
        generation_config: dict = {"responseModalities": ["IMAGE"]}
        image_config: dict = {}

        if request.aspect_ratio and not request.images:
            image_config["aspectRatio"] = request.aspect_ratio

        if request.resolution and "gemini-3" in self.model.lower():
            image_config["imageSize"] = request.resolution

        if image_config:
            generation_config["imageConfig"] = image_config

        safety_settings = []
        if self.safety_settings:
            for category in GEMINI_SAFETY_CATEGORIES:
                safety_settings.append(
                    {"category": category, "threshold": self.safety_settings}
                )

        parts = [{"text": request.prompt}]
        for image in request.images:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": image.mime_type,
                        "data": base64.b64encode(image.data).decode("utf-8"),
                    }
                }
            )

        payload: dict = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        }

        if safety_settings:
            payload["safetySettings"] = safety_settings

        return payload

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        task_id: str | None,
    ) -> dict | None:
        """发送 API 请求。"""
        start_time = time.time()
        url = f"{self.base_url or self.DEFAULT_BASE_URL}/v1beta/models/{self.model}:generateContent"
        api_key = self._get_current_api_key()
        masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
        prefix = self._get_log_prefix(task_id)
        logger.debug(
            f"{prefix} 请求 -> {url}, key={masked_key}"
        )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                proxy=self.proxy,
            ) as response:
                duration = time.time() - start_time
                logger.debug(
                    f"{prefix} 状态 -> {response.status} (耗时: {duration:.2f}s)"
                )
                if response.status != 200:
                    error_text = await response.text()
                    preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"{prefix} 错误 {response.status} (耗时: {duration:.2f}s): {preview}"
                    )
                    return None
                return await response.json()
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}"
            )
            return None

    def _extract_images(
        self, response: dict, task_id: str | None
    ) -> list[bytes] | None:
        """从响应中提取图像数据。"""
        prefix = self._get_log_prefix(task_id)
        try:
            candidates = response.get("candidates", [])
            logger.debug(f"{prefix} 候选结果: {len(candidates)}")
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            images: list[bytes] = []
            for part in parts:
                inline_data = part.get("inline_data") or part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    images.append(base64.b64decode(inline_data["data"]))

            return images if images else None
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 解析失败: {exc}")
            return None
