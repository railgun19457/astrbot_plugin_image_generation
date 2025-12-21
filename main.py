from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Coroutine
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.utils.io import download_image_by_url, save_temp_img

from .core.generator import ImageGenerator
from .core.types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    ImageCapability,
    ImageData,
)
from .core.utils import validate_aspect_ratio, validate_resolution


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """LLM 可调用的图像生成工具。"""

    name: str = "generate_image"
    description: str = "使用生图模型生成或修改图片"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的提示词(直接将用户发送的内容原样传递给模型)",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比",
                    "enum": [],  # 占位符，稍后会被替换
                },
                "resolution": {
                    "type": "string",
                    "description": "图片分辨率",
                    "enum": ["1K", "2K", "4K"],
                },
                "avatar_references": {
                    "type": "array",
                    "description": "需要作为参考的用户头像列表。支持: 'self'(机器人头像)、'sender'(发送者头像)、或具体的QQ号",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    def __post_init_post_parse__(self):
        """初始化后处理，动态补齐宽高比枚举。"""
        # 初始化时动态补齐宽高比枚举，避免写死在默认 schema 中
        self.parameters["properties"]["aspect_ratio"]["enum"] = [
            "自动",
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ]

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行工具调用。"""
        # 获取提示词
        if not (prompt := kwargs.get("prompt", "")):
            return "请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        # 获取事件上下文
        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning(
                f"[ImageGen] 工具调用上下文缺少事件。上下文类型: {type(context)}"
            )
            return "❌ 无法获取当前消息上下文"

        if not plugin.adapter_config.api_keys:
            return "❌ 未配置 API Key，无法生成图片"

        # 工具调用同样支持获取上下文参考图（消息/引用/头像）
        images_data = []
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )

        if capabilities & ImageCapability.IMAGE_TO_IMAGE:
            images_data = await plugin._get_reference_images_for_tool(event)

            # 处理头像引用参数
            avatar_refs = kwargs.get("avatar_references", [])
            if avatar_refs and isinstance(avatar_refs, list):
                for ref in avatar_refs:
                    if not isinstance(ref, str):
                        continue
                    ref = ref.strip().lower()
                    user_id = None
                    if ref == "self":
                        user_id = str(event.get_self_id())
                    elif ref == "sender":
                        user_id = str(event.get_sender_id() or event.unified_msg_origin)
                    else:
                        user_id = ref
                    if user_id:
                        avatar_data = await plugin.get_avatar(user_id)
                        if avatar_data:
                            images_data.append((avatar_data, "image/jpeg"))
                            logger.info(f"[ImageGen] 已添加 {user_id} 的头像作为参考图")

        # 生成任务 ID
        task_id = hashlib.md5(
            f"{time.time()}{event.unified_msg_origin}".encode()
        ).hexdigest()[:8]

        # 创建后台任务进行生图
        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=kwargs.get("aspect_ratio") or plugin.default_aspect_ratio,
                resolution=kwargs.get("resolution") or plugin.default_resolution,
                task_id=task_id,
            )
        )

        mode = "图生图" if images_data else "文生图"
        return f"已启动{mode}任务"


class ImageGenerationPlugin(Star):
    """Gemini 图像生成插件"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()

        self.adapter_config: AdapterConfig | None = None
        self.generator: ImageGenerator | None = None

        # 用于频率限制
        self.user_request_timestamps: dict[str, float] = {}
        # 后台任务集合
        self.background_tasks: set[asyncio.Task] = set()
        # 并发控制信号量
        self.semaphore: asyncio.Semaphore | None = None

        self.enable_llm_tool = True
        self.default_aspect_ratio = "自动"
        self.default_resolution = "1K"
        self.max_image_size_mb = 10
        self.presets: dict[str, Any] = {}
        self.rate_limit_seconds = 0

        self._load_config()

        if self.adapter_config:
            self.generator = ImageGenerator(self.adapter_config)
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        else:
            logger.error("[ImageGen] 适配器配置加载失败，插件未初始化")

        if self.enable_llm_tool and self.generator:
            tool = ImageGenerationTool(plugin=self)
            self._adjust_tool_parameters(tool)
            self.context.add_llm_tools(tool)
            logger.info("[ImageGen] 已注册图像生成工具")

        logger.info(
            f"[ImageGen] 插件加载完成，模型: {self.adapter_config.model if self.adapter_config else '未知'}"
        )

    def _adjust_tool_parameters(self, tool: ImageGenerationTool):
        """根据适配器能力动态调整工具参数。"""
        if not self.generator or not self.generator.adapter:
            return

        capabilities = self.generator.adapter.get_capabilities()
        props = tool.parameters["properties"]

        if not (capabilities & ImageCapability.ASPECT_RATIO):
            if "aspect_ratio" in props:
                del props["aspect_ratio"]
                logger.debug("[ImageGen] 适配器不支持宽高比，已从工具参数中移除")

        if not (capabilities & ImageCapability.RESOLUTION):
            if "resolution" in props:
                del props["resolution"]
                logger.debug("[ImageGen] 适配器不支持分辨率，已从工具参数中移除")

        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
            if "avatar_references" in props:
                del props["avatar_references"]
                logger.debug("[ImageGen] 适配器不支持参考图，已从工具参数中移除头像引用")

    # ---------------------------- 配置加载 -----------------------------
    def _load_config(self) -> None:
        """加载插件配置。"""
        adapter_cfg = self.config.get("adapter", {})
        gen_cfg = self.config.get("generation", {})

        self.enable_llm_tool = self.config.get("enable_llm_tool", True)

        adapter_type_raw = adapter_cfg.get("type", "gemini")
        try:
            adapter_type = AdapterType(adapter_type_raw)
        except Exception:
            adapter_type = AdapterType.GEMINI

        base_url = (adapter_cfg.get("base_url") or "").strip()
        api_keys: list[str] = [k for k in adapter_cfg.get("api_keys", []) if k]
        provider_id = (adapter_cfg.get("provider_id") or "").strip()

        # 如果配置了 provider_id，尝试从系统提供商加载配置
        if provider_id:
            loaded_keys, loaded_base = self._load_provider_config(provider_id)
            if loaded_keys:
                api_keys = loaded_keys
            if loaded_base:
                base_url = loaded_base

        available_models = adapter_cfg.get("available_models") or []

        model = adapter_cfg.get("model") or (
            available_models[0] if available_models else ""
        )

        self.adapter_config = AdapterConfig(
            type=adapter_type,
            base_url=self._clean_base_url(base_url),
            api_keys=api_keys,
            model=model,
            available_models=available_models,
            provider_id=provider_id,
            proxy=(adapter_cfg.get("proxy") or "").strip() or None,
            timeout=gen_cfg.get("timeout", 180),
            max_retry_attempts=gen_cfg.get("max_retry_attempts", 3),
        )

        self.rate_limit_seconds = max(0, gen_cfg.get("rate_limit_seconds", 0))
        self.max_concurrent_tasks = max(1, gen_cfg.get("max_concurrent_tasks", 3))
        self.default_aspect_ratio = gen_cfg.get("default_aspect_ratio", "自动")
        self.default_resolution = gen_cfg.get("default_resolution", "1K")

        self.presets = self._load_presets(self.config.get("presets", []))

    def _clean_base_url(self, url: str) -> str:
        """清理 Base URL，移除末尾的 /v1*"""
        if not url:
            return ""
        url = url.rstrip("/")
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_provider_config(self, provider_id: str) -> tuple[list[str], str]:
        """从 AstrBot 系统提供商加载配置。"""
        provider = self.context.get_provider_by_id(provider_id)
        if not provider:
            logger.warning(f"[ImageGen] 未找到提供商 {provider_id}，使用插件配置")
            return [], ""

        provider_config = getattr(provider, "provider_config", {}) or {}
        api_keys: list[str] = []
        for key_field in ["key", "keys", "api_key", "access_token"]:
            if keys := provider_config.get(key_field):
                api_keys = [keys] if isinstance(keys, str) else [k for k in keys if k]
                break

        api_base = (
            getattr(provider, "api_base", None)
            or provider_config.get("api_base")
            or provider_config.get("api_base_url")
        )

        if not api_keys:
            logger.warning(f"[ImageGen] 提供商 {provider_id} 未提供可用的 API Key")
            return [], ""

        base_url = self._clean_base_url(api_base or "")
        logger.info(f"[ImageGen] 使用系统提供商: {provider_id}")
        return api_keys, base_url

    def _load_presets(self, presets_config: list[Any]) -> dict[str, Any]:
        """加载预设配置。"""
        presets: dict[str, Any] = {}
        if not isinstance(presets_config, list):
            return presets

        for preset_str in presets_config:
            if isinstance(preset_str, str) and ":" in preset_str:
                name, prompt = preset_str.split(":", 1)
                if name.strip() and prompt.strip():
                    presets[name.strip()] = prompt.strip()
        return presets

    # --------------------------- 指令处理 ----------------------------
    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """处理生图指令。"""
        user_id = event.unified_msg_origin

        # 检查频率限制
        if not self._check_rate_limit(user_id):
            if self.rate_limit_seconds > 0:
                yield event.plain_result(
                    f"❌ 请求过于频繁，请间隔 {self.rate_limit_seconds} 秒再试"
                )
            return

        masked_uid = (
            user_id[:4] + "****" + user_id[-4:] if len(user_id) > 8 else user_id
        )

        user_input = (event.message_str or "").strip()
        logger.info(f"[ImageGen] 收到生图指令 - 用户: {masked_uid}, 输入: {user_input}")

        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
        aspect_ratio = self.default_aspect_ratio
        resolution = self.default_resolution

        # 检查是否命中预设
        matched_preset = None
        extra_content = ""
        if prompt:
            parts = prompt.split(maxsplit=1)
            first_token = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if first_token in self.presets:
                matched_preset = first_token
                extra_content = rest
            else:
                for name in self.presets:
                    if name.lower() == first_token.lower():
                        matched_preset = name
                        extra_content = rest
                        break

        if matched_preset:
            logger.info(f"[ImageGen] 命中预设: {matched_preset}")
            preset_content = self.presets[matched_preset]
            try:
                # 预设支持 JSON 格式配置高级参数
                if isinstance(
                    preset_content, str
                ) and preset_content.strip().startswith("{"):
                    preset_data = json.loads(preset_content)
                    if isinstance(preset_data, dict):
                        prompt = preset_data.get("prompt", "")
                        aspect_ratio = preset_data.get("aspect_ratio", aspect_ratio)
                        resolution = preset_data.get("resolution", resolution)
                    else:
                        prompt = preset_content
                else:
                    prompt = preset_content
            except json.JSONDecodeError:
                prompt = preset_content

            if extra_content:
                prompt = f"{prompt} {extra_content}"

        if not prompt:
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        # 获取参考图
        images_data = None
        if (
            self.generator
            and self.generator.adapter
            and (
                self.generator.adapter.get_capabilities()
                & ImageCapability.IMAGE_TO_IMAGE
            )
        ):
            images_data = await self._get_reference_images_for_command(event)

        msg = "已开始生图任务"
        if images_data:
            msg += f"[{len(images_data)}张参考图]"
        if matched_preset:
            msg += f"[预设: {matched_preset}]"
        yield event.plain_result(msg)

        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]

        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """切换生图模型。"""
        if not self.adapter_config:
            yield event.plain_result("❌ 适配器未初始化")
            return

        models = self.adapter_config.available_models or []

        if not model_index:
            lines = ["📋 可用模型列表:"]
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == self.adapter_config.model else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n当前使用: {self.adapter_config.model}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                new_model = models[index]
                self.adapter_config.model = new_model
                if self.generator:
                    self.generator.update_model(new_model)
                self.config.setdefault("adapter", {})["model"] = new_model
                self.config.save_config()
                yield event.plain_result(f"✅ 模型已切换: {new_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """管理生图预设。"""
        user_id = event.unified_msg_origin
        masked_uid = (
            user_id[:4] + "****" + user_id[-4:] if len(user_id) > 8 else user_id
        )
        message_str = (event.message_str or "").strip()
        logger.info(
            f"[ImageGen] 收到预设指令 - 用户: {masked_uid}, 内容: {message_str}"
        )

        parts = message_str.split(maxsplit=1)
        cmd_text = parts[1].strip() if len(parts) > 1 else ""

        if not cmd_text:
            if not self.presets:
                yield event.plain_result("📋 当前没有预设")
                return
            preset_list = ["📋 预设列表:"]
            for idx, (name, prompt) in enumerate(self.presets.items(), 1):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            parts = cmd_text[3:].split(":", 1)
            if len(parts) == 2:
                name, prompt = parts
                self.presets[name.strip()] = prompt.strip()
                self.config["presets"] = [f"{k}:{v}" for k, v in self.presets.items()]
                self.config.save_config()
                yield event.plain_result(f"✅ 预设已添加: {name.strip()}")
            else:
                yield event.plain_result("❌ 格式错误: /预设 添加 名称:内容")
        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if name in self.presets:
                del self.presets[name]
                self.config["presets"] = [f"{k}:{v}" for k, v in self.presets.items()]
                self.config.save_config()
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 预设不存在: {name}")

    # ----------------------------- 辅助方法 ---------------------------
    def _check_rate_limit(self, user_id: str) -> bool:
        """检查用户请求频率限制。"""
        if self.rate_limit_seconds <= 0:
            return True
        now = time.time()
        last_ts = self.user_request_timestamps.get(user_id, 0)
        if now - last_ts < self.rate_limit_seconds:
            return False
        self.user_request_timestamps[user_id] = now
        return True

    async def _fetch_images_from_event(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """从消息事件中提取图片（包括直接发送的图片、引用消息中的图片、被@用户的头像）。"""
        images_data: list[tuple[bytes, str]] = []

        if not event.message_obj.message:
            return images_data

        # 预扫描：记录引用消息的发送者以及各个 @ 出现次数，用于过滤自动 @
        reply_sender_id = None
        at_counts: dict[str, int] = {}

        for component in event.message_obj.message:
            if isinstance(component, Comp.Reply):
                if hasattr(component, "sender_id") and component.sender_id:
                    reply_sender_id = str(component.sender_id)
            elif isinstance(component, Comp.At):
                if component.qq != "all":
                    uid = str(component.qq)
                    at_counts[uid] = at_counts.get(uid, 0) + 1

        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                # 处理直接发送的图片
                url = component.url or component.file
                if url and (data := await self._download_image(url)):
                    images_data.append(data)
            elif isinstance(component, Comp.Reply):
                # 处理引用消息中的图片
                if component.chain:
                    for sub_comp in component.chain:
                        if isinstance(sub_comp, Comp.Image):
                            url = sub_comp.url or sub_comp.file
                            if url and (data := await self._download_image(url)):
                                images_data.append(data)
            elif isinstance(component, Comp.At):
                # 处理 @ 用户的头像
                if component.qq != "all":
                    uid = str(component.qq)
                    # 引用消息带来的单次自动 @ 默认忽略头像，除非用户再次显式 @
                    if reply_sender_id and uid == reply_sender_id:
                        if at_counts.get(uid, 0) == 1:
                            continue
                    self_id = str(event.get_self_id()).strip()
                    # 机器人单次被 @ 多为触发前缀，默认不取机器人头像
                    if self_id and uid == self_id and at_counts.get(uid, 0) == 1:
                        continue
                    if avatar_data := await self.get_avatar(uid):
                        images_data.append((avatar_data, "image/jpeg"))
        return images_data

    async def _get_reference_images_for_command(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """为指令获取参考图。"""
        return await self._fetch_images_from_event(event)

    async def _get_reference_images_for_tool(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """为工具调用获取参考图。"""
        return await self._fetch_images_from_event(event)

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """创建后台任务并添加到集合中，防止被 GC。"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    @staticmethod
    async def get_avatar(user_id: str) -> bytes | None:
        """获取用户头像。"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            path = await download_image_by_url(url)
            if path:
                with open(path, "rb") as f:
                    return f.read()
        except Exception:
            pass
        return None

    async def _download_image(self, url: str) -> tuple[bytes, str] | None:
        """下载图片并返回二进制数据和 MIME 类型。"""
        try:
            data: bytes | None = None
            if os.path.exists(url) and os.path.isfile(url):
                with open(url, "rb") as f:
                    data = f.read()
            else:
                path = await download_image_by_url(url)
                if path:
                    with open(path, "rb") as f:
                        data = f.read()

            if not data:
                return None

            if len(data) > self.max_image_size_mb * 1024 * 1024:
                logger.warning(
                    f"[ImageGen] 图片超过大小限制 ({self.max_image_size_mb}MB)"
                )
                return None

            mime = "image/png"
            if data.startswith(b"\xff\xd8"):
                mime = "image/jpeg"
            elif data.startswith(b"GIF"):
                mime = "image/gif"
            elif data.startswith(b"RIFF") and b"WEBP" in data[:16]:
                mime = "image/webp"
            return data, mime
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] 获取图片失败 (URL/Path: {url}): {exc}")
        return None

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
    ) -> None:
        """异步生成图片并发送。"""
        if not self.generator or not self.generator.adapter:
            return

        capabilities = self.generator.adapter.get_capabilities()

        # 检查并清理不支持的参数
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
            logger.warning(
                f"[ImageGen] 当前适配器不支持参考图，已忽略 {len(images_data)} 张图片"
            )
            images_data = None

        if not (capabilities & ImageCapability.ASPECT_RATIO) and aspect_ratio != "自动":
            logger.info(f"[ImageGen] 当前适配器不支持指定比例，已忽略参数: {aspect_ratio}")
            aspect_ratio = "自动"

        if not (capabilities & ImageCapability.RESOLUTION) and resolution != "1K":
            logger.info(f"[ImageGen] 当前适配器不支持指定分辨率，已忽略参数: {resolution}")
            resolution = "1K"

        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        final_ar = validate_aspect_ratio(aspect_ratio) or None
        if final_ar == "自动":
            final_ar = None
        final_res = validate_resolution(resolution)

        images: list[ImageData] = []
        if images_data:
            for data, mime in images_data:
                images.append(ImageData(data=data, mime_type=mime))

        # 使用信号量控制并发
        if self.semaphore is None:
            await self._do_generate_and_send(
                prompt, unified_msg_origin, images, final_ar, final_res, task_id
            )
            return

        async with self.semaphore:
            await self._do_generate_and_send(
                prompt, unified_msg_origin, images, final_ar, final_res, task_id
            )

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        task_id: str,
    ) -> None:
        """执行生成逻辑并发送结果。"""
        result = await self.generator.generate(
            GenerationRequest(
                prompt=prompt,
                images=images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )

        if result.error:
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 生成失败: {result.error}"),
            )
            return

        if not result.images:
            return

        chain = MessageChain()
        for img_bytes in result.images:
            try:
                file_path = save_temp_img(img_bytes)
                chain.file_image(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[ImageGen] 保存图片失败: {exc}")

        await self.context.send_message(unified_msg_origin, chain)

    async def terminate(self):
        """插件卸载时清理资源。"""
        try:
            if self.generator:
                await self.generator.close()
            for task in list(self.background_tasks):
                if not task.done():
                    task.cancel()
            logger.info("[ImageGen] 插件已卸载")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] 卸载清理出错: {exc}")
