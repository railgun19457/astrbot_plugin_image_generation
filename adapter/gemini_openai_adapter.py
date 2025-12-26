from __future__ import annotations

import asyncio
import base64
import re
import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, GenerationResult, ImageCapability


class GeminiOpenAIAdapter(BaseImageAdapter):
    """通过 OpenAI 兼容的聊天补全接口进行 Gemini 图像生成。"""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return ImageCapability.TEXT_TO_IMAGE | ImageCapability.IMAGE_TO_IMAGE

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行生图逻辑。"""
        if not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

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
                if (attempt + 1) % max(1, len(self.api_keys)) == 0:
                    await asyncio.sleep(
                        min(2 ** ((attempt + 1) // len(self.api_keys)), 10)
                    )

        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        payload = self._build_payload(request)
        session = self._get_session()
        response = await self._make_request(session, payload, request.task_id)
        if response is None:
            return None, "API 请求失败"

        images = await self._extract_images(response, request.task_id)
        if images:
            return images, None

        # 尝试提取文本错误信息
        if "choices" in response and response["choices"]:
            content = response["choices"][0].get("message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return None, f"未生成图片，API 返回文本: {content[:100]}"
        return None, "响应中未找到图片 data"

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""
        message_content: list[dict] = [
            {"type": "text", "text": f"Generate an image: {request.prompt}"}
        ]

        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{b64_data}"},
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "modalities": ["image", "text"],
            "stream": False,
        }

        image_config: dict[str, Any] = {}
        generation_config: dict[str, Any] = {}

        if request.aspect_ratio and not request.images:
            image_config["aspectRatio"] = request.aspect_ratio
        if request.resolution:
            image_config["imageSize"] = request.resolution
        if image_config:
            generation_config["imageConfig"] = image_config
        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        task_id: str | None,
    ) -> dict | None:
        """发送 API 请求。"""
        start_time = time.time()
        url = f"{self.base_url or self.DEFAULT_BASE_URL}/v1/chat/completions"
        api_key = self._get_current_api_key()
        masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
        prefix = self._get_log_prefix(task_id)
        logger.debug(
            f"{prefix} 请求 -> {url}, key={masked_key}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
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

    async def _download_image_from_url(self, url: str, task_id: str | None = None) -> bytes | None:
        """从 URL 下载图像。"""
        prefix = self._get_log_prefix(task_id)
        try:
            session = self._get_session()
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.read()
                logger.error(f"{prefix} 下载图像失败: {response.status} - {url}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像出错: {exc}")
        return None

    async def _extract_images(
        self, response_data: dict[str, Any], task_id: str | None = None
    ) -> list[bytes] | None:
        """从响应数据中提取图像。"""
        images: list[bytes] = []

        # DALL-E 风格
        if isinstance(response_data.get("data"), list):
            for item in response_data["data"]:
                if not isinstance(item, dict):
                    continue
                if b64 := item.get("b64_json"):
                    try:
                        images.append(base64.b64decode(b64))
                    except Exception:
                        pass
                elif url := item.get("url"):
                    if url.startswith("http"):
                        if content := await self._download_image_from_url(url, task_id):
                            images.append(content)
                    else:
                        decoded = self._decode_image_url(url, task_id)
                        if decoded:
                            images.append(decoded)

        # 聊天补全风格
        if choices := response_data.get("choices"):
            message = (
                choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            )
            content = message.get("content")

            if isinstance(content, str):
                markdown_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
                for url in markdown_matches:
                    if url.startswith("http"):
                        if data := await self._download_image_from_url(url, task_id):
                            images.append(data)
                    else:
                        decoded = self._decode_image_url(url, task_id)
                        if decoded:
                            images.append(decoded)

                content_without_md = re.sub(r"!\[.*?\]\(.*?\)", "", content)
                pattern = re.compile(
                    r"data\s*:\s*image/([a-zA-Z0-9.+-]+)\s*;\s*base64\s*,\s*([-A-Za-z0-9+/=_\s]+)",
                    flags=re.IGNORECASE,
                )
                for _, b64_str in pattern.findall(content_without_md):
                    try:
                        images.append(base64.b64decode(b64_str))
                    except Exception:
                        pass

            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if not image_url:
                            continue
                        if image_url.startswith("http"):
                            if data := await self._download_image_from_url(image_url, task_id):
                                images.append(data)
                        else:
                            decoded = self._decode_image_url(image_url, task_id)
                            if decoded:
                                images.append(decoded)

            if message.get("images"):
                for img_item in message["images"]:
                    url = None
                    if isinstance(img_item, dict):
                        url = img_item.get("url") or img_item.get("image_url", {}).get(
                            "url"
                        )
                    elif isinstance(img_item, str):
                        url = img_item
                    if not url:
                        continue
                    if url.startswith("http"):
                        if data := await self._download_image_from_url(url, task_id):
                            images.append(data)
                    else:
                        decoded = self._decode_image_url(url, task_id)
                        if decoded:
                            images.append(decoded)

        return images or None

    def _decode_image_url(self, url: str, task_id: str | None = None) -> bytes | None:
        """解码 Data URL 形式的图像。"""
        if url.startswith("data:image/") and ";base64," in url:
            try:
                _, _, data_part = url.partition(";base64,")
                return base64.b64decode(data_part)
            except Exception as exc:  # noqa: BLE001
                prefix = self._get_log_prefix(task_id)
                logger.error(f"{prefix} Base64 解码失败: {exc}")
        return None
