"""
OpenAI compatible API adapter for image generation
OpenAI 兼容 API 图像生成适配器
"""

from __future__ import annotations

import base64
import re

import aiohttp

from astrbot.api import logger

from ..core import (
    AdapterMetadata,
    BaseImageAdapter,
    GenerationRequest,
    GenerationResult,
)


class GeminiOpenAIAdapter(BaseImageAdapter):
    """Gemini OpenAI 兼容 API 图像生成适配器"""

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="openai",
            display_name="OpenAI Compatible",
            description="OpenAI 兼容的图像生成 API",
            supported_features=["text_to_image", "image_to_image", "aspect_ratio"],
            default_models=[
                "gpt-4o",
                "gpt-4-turbo",
                "dall-e-3",
                "dall-e-2",
            ],
            supports_image_to_image=True,
            supports_aspect_ratio=True,
            supports_resolution=False,
            supports_safety_settings=False,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        使用 OpenAI 兼容 API 生成图像

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

            images = await self._extract_images(response_data)
            if images:
                return GenerationResult(success=True, images=images)

            # 尝试提取文本错误信息
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0].get("message", {}).get("content")
                if content and isinstance(content, str):
                    return GenerationResult(
                        success=False,
                        error_message=f"未生成图片，API返回文本: {content[:100]}",
                    )

            return GenerationResult(success=False, error_message="响应中未找到图片数据")

        except Exception as e:
            logger.error(f"[OpenAI] {prefix}生成失败: {e}", exc_info=True)
            return GenerationResult(success=False, error_message=f"生成失败: {str(e)}")

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建 OpenAI Chat Completions 请求负载"""
        message_content = [
            {"type": "text", "text": f"Generate an image: {request.prompt}"}
        ]

        # 处理参考图
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
            "modalities": ["image", "text"],
            "stream": False,
        }

        # image_config
        image_config = {}
        generation_config = {}

        if request.aspect_ratio and not request.is_image_to_image:
            image_config["aspectRatio"] = request.aspect_ratio

        if request.resolution:
            image_config["imageSize"] = request.resolution

        if image_config:
            generation_config["imageConfig"] = image_config

        if generation_config:
            payload["generationConfig"] = generation_config

        self._log_request("OpenAI Chat", payload)
        return payload

    async def _make_request(
        self, session: aiohttp.ClientSession, payload: dict, task_id: str | None = None
    ) -> dict | None:
        """发送 OpenAI API 请求"""
        prefix = f"[{task_id}] " if task_id else ""
        url = f"{self.config.base_url}/v1/chat/completions"

        api_key = self._get_current_api_key()
        masked_key = self._mask_api_key(api_key)
        logger.debug(f"[OpenAI] {prefix}Request URL: {url}, Key: {masked_key}")

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
                logger.debug(f"[OpenAI] {prefix}API Response Status: {response.status}")

                if response.status != 200:
                    error_text = await response.text()
                    error_preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"[OpenAI] {prefix}API 错误: {response.status} - {error_preview}"
                    )
                    return None

                return await response.json()

        except Exception as e:
            logger.error(f"[OpenAI] {prefix}请求异常: {e}")
            return None

    async def _extract_images(self, response_data: dict) -> list[bytes] | None:
        """从 OpenAI Chat 响应中提取图片"""
        images = []

        # 0. 检查标准 OpenAI DALL-E 格式 (data 字段)
        if "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if isinstance(item, dict):
                    b64_json = item.get("b64_json")
                    url = item.get("url")
                    if b64_json:
                        try:
                            images.append(base64.b64decode(b64_json))
                        except Exception:
                            pass
                    elif url:
                        if url.startswith("http"):
                            img_data = await self._download_image_from_url(url)
                            if img_data:
                                images.append(img_data)
                        else:
                            img_data = self._decode_image_url(url)
                            if img_data:
                                images.append(img_data)

        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")

            # 1. 检查 message.content 为字符串的情况 (Markdown 图片 / Data URI)
            if isinstance(content, str):
                # 匹配 markdown 图片语法 ![...](url)
                markdown_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
                for url in markdown_matches:
                    if url.startswith("http"):
                        img_data = await self._download_image_from_url(url)
                        if img_data:
                            images.append(img_data)
                    else:
                        img_data = self._decode_image_url(url)
                        if img_data:
                            images.append(img_data)

                # 匹配纯文本中的 Data URI
                content_without_markdown = re.sub(r"!\[.*?\]\(.*?\)", "", content)
                pattern = re.compile(
                    r"data\s*:\s*image/([a-zA-Z0-9.+-]+)\s*;\s*base64\s*,\s*([-A-Za-z0-9+/=_\s]+)",
                    flags=re.IGNORECASE,
                )
                data_uri_matches = pattern.findall(content_without_markdown)
                for _, b64_str in data_uri_matches:
                    try:
                        images.append(base64.b64decode(b64_str))
                    except Exception:
                        pass

            # 2. 检查 message.content 中的 image_url (Gemini 风格)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if image_url:
                            if image_url.startswith("http"):
                                img_data = await self._download_image_from_url(
                                    image_url
                                )
                                if img_data:
                                    images.append(img_data)
                            else:
                                img_data = self._decode_image_url(image_url)
                                if img_data:
                                    images.append(img_data)

            # 3. 检查 message.images 字段
            if message.get("images"):
                for img_item in message["images"]:
                    url = None
                    if isinstance(img_item, dict):
                        url = img_item.get("url") or img_item.get("image_url", {}).get(
                            "url"
                        )
                    elif isinstance(img_item, str):
                        url = img_item

                    if url:
                        if url.startswith("http"):
                            img_data = await self._download_image_from_url(url)
                            if img_data:
                                images.append(img_data)
                        else:
                            img_data = self._decode_image_url(url)
                            if img_data:
                                images.append(img_data)

        return images if images else None

    async def _download_image_from_url(self, url: str) -> bytes | None:
        """从 URL 下载图片"""
        try:
            session = self._get_session()
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"[OpenAI] 下载图片失败: {response.status} - {url}")
                    return None
        except Exception as e:
            logger.error(f"[OpenAI] 下载图片异常: {e}")
            return None

    def _decode_image_url(self, url: str) -> bytes | None:
        """解码 data:image/...;base64,... URL"""
        if url.startswith("data:image/") and ";base64," in url:
            try:
                _, _, data_part = url.partition(";base64,")
                return base64.b64decode(data_part)
            except Exception as e:
                logger.error(f"[OpenAI] Base64 解码失败: {e}")
                return None
        return None
