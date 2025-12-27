from __future__ import annotations

import asyncio
import datetime
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
from astrbot.core.utils.io import download_image_by_url

from .core.generator import ImageGenerator
from .core.task_manager import TaskManager
from .core.task_manager import TaskManager
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
                    "description": "生图时使用的提示词(要将用户的意图原样传达给模型)。如果用户提到了画图但没有具体描述，请根据上下文推断或提示用户描述。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比。如果不确定，请使用'自动'。",
                    "enum": [
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
                    ],
                    "default": "自动",
                },
                "resolution": {
                    "type": "string",
                    "description": "图片质量/分辨率。默认使用 '1K'。",
                    "enum": ["1K", "2K", "4K"],
                    "default": "1K",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "当需要使用某人的头像时使用。'self'表示机器人，'sender'表示发送者，也可以直接使用ID做参数。",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行工具调用。"""
        # 获取提示词
        prompt = kwargs.get("prompt", "").strip()
        if not prompt:
            return ToolExecResult(
                summary="未提供提示词",
                success=False,
                error="请提供图片生成的提示词"
            )

        plugin = self.plugin
        if not plugin:
            return ToolExecResult(
                summary="插件实例缺失",
                success=False,
                error="❌ 插件未正确初始化 (Plugin instance missing)"
            )

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
            return ToolExecResult(
                summary="无法获取上下文",
                success=False,
                error="❌ 无法获取当前消息上下文"
            )

        # 检查频率限制和每日限制
        check_result = plugin._check_rate_limit(event.unified_msg_origin)
        if isinstance(check_result, str):
            logger.warning(f"[ImageGen] 工具调用触发限制: {check_result} (用户: {event.unified_msg_origin})")
            return ToolExecResult(
                summary="触发限制",
                success=False,
                error=check_result
            )

        if not plugin.adapter_config or not plugin.adapter_config.api_keys:
            logger.warning(f"[ImageGen] 工具调用失败: 未配置 API Key (用户: {event.unified_msg_origin})")
            return ToolExecResult(
                summary="配置缺失",
                success=False,
                error="❌ 未配置 API Key，无法生成图片"
            )

        # 工具调用同样支持获取上下文参考图（消息/引用/头像）
        images_data = []
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )

        try:
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
                            # 简单的 QQ 号校验（可选）
                            if ref.isdigit():
                                user_id = ref

                        if user_id:
                            avatar_data = await plugin.get_avatar(user_id)
                            if avatar_data:
                                images_data.append((avatar_data, "image/jpeg"))
                                logger.info(f"[ImageGen] 已添加 {user_id} 的头像作为参考图")
        except Exception as e:
            logger.error(f"[ImageGen] 处理参考图失败: {e}", exc_info=True)
            # 参考图处理失败不影响文生图流程，记录日志继续执行

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
        return ToolExecResult(
            summary=f"已启动{mode}任务",
            success=True,
            data={"task_id": task_id, "mode": mode}
        )


class ImageGenerationPlugin(Star):
    """Gemini 图像生成插件"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()

        self.adapter_config: AdapterConfig | None = None
        self.generator: ImageGenerator | None = None
        self.task_manager = TaskManager()
        self.task_manager = TaskManager()

        # 用于频率限制
        self.user_request_timestamps: dict[str, float] = {}
        # 并发控制信号量
        self.semaphore: asyncio.Semaphore | None = None

        self.data_dir = "data/plugin_data/astrbot_plugin_gemini_generation"
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.usage_file = os.path.join(self.data_dir, "usage.json")
        self.usage_data: dict[str, dict[str, int]] = {}  # {date: {user_id: count}}
        self._ensure_dirs()
        self._load_usage_data()

        self.enable_llm_tool = True
        self.default_aspect_ratio = "自动"
        self.default_resolution = "1K"
        self.max_image_size_mb = 10
        self.presets: dict[str, Any] = {}
        self.rate_limit_seconds = 0
        self.enable_daily_limit = False
        self.daily_limit_count = 10
        self.max_cache_count = 100
        self.cleanup_interval_hours = 24
        self.show_model_info = False

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

        # 启动定时任务
        self._setup_tasks()
        # 启动定时任务
        self._setup_tasks()

        logger.info(
            f"[ImageGen] 插件加载完成，模型: {self.adapter_config.model if self.adapter_config else '未知'}"
        )

    def _ensure_dirs(self):
        """确保数据和缓存目录存在。"""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _setup_tasks(self):
        """配置并启动定时任务。"""
        # 1. 缓存清理任务
        self.task_manager.start_loop_task(
            name="cache_cleanup",
            coro_func=self._cleanup_cache,
            interval_seconds=self.cleanup_interval_hours * 3600,
            run_immediately=True
        )

        # 2. Jimeng2API 自动领积分任务
        self._setup_jimeng_token_task()

    def _setup_jimeng_token_task(self):
        """配置即梦自动领积分任务。"""
        from .adapter.jimeng2api_adapter import Jimeng2APIAdapter

        if self.generator and isinstance(self.generator.adapter, Jimeng2APIAdapter):
            # 每 12 小时执行一次
            self.task_manager.start_loop_task(
                name="jimeng_token_receive",
                coro_func=self.generator.adapter.receive_token,
                interval_seconds=12 * 3600,
                run_immediately=True
            )
            logger.info("[ImageGen] 已启动即梦 2 自动领积分任务")

    def _load_usage_data(self):
        """加载用户使用数据。"""
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, encoding="utf-8") as f:
                    self.usage_data = json.load(f)

                # 清理旧数据，只保留最近 7 天
                today = datetime.date.today()
                keys_to_delete = []
                for date_str in self.usage_data:
                    try:
                        date_obj = datetime.date.fromisoformat(date_str)
                        if (today - date_obj).days > 7:
                            keys_to_delete.append(date_str)
                    except ValueError:
                        keys_to_delete.append(date_str)

                if keys_to_delete:
                    for key in keys_to_delete:
                        del self.usage_data[key]
                    self._save_usage_data()
            except Exception as exc:
                logger.error(f"[ImageGen] 加载使用数据失败: {exc}")
                self.usage_data = {}

    def _save_usage_data(self):
        """保存用户使用数据。"""
        try:
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(self.usage_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"[ImageGen] 保存使用数据失败: {exc}")

    async def _cleanup_cache(self):
        """执行缓存清理。"""
        if not os.path.exists(self.cache_dir):
            return

        files = []
        for f in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, f)
            if os.path.isfile(path):
                files.append((path, os.path.getmtime(path)))

        # 按修改时间排序（旧的在前）
        files.sort(key=lambda x: x[1])

        # 按数量清理
        if len(files) > self.max_cache_count:
            to_delete = files[: len(files) - self.max_cache_count]
            for path, _ in to_delete:
                try:
                    os.remove(path)
                except Exception:
                    pass
            logger.info(f"[ImageGen] 已清理 {len(to_delete)} 个旧缓存文件 (按数量)")

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
                logger.debug(
                    "[ImageGen] 适配器不支持参考图，已从工具参数中移除头像引用"
                )

    # ---------------------------- 配置加载 -----------------------------
    def _load_config(self) -> None:
        """加载插件配置。"""
        gen_cfg = self.config.get("generation", {})
        user_limits_cfg = self.config.get("user_limits", {})
        cache_cfg = self.config.get("cache", {})
        api_providers_raw = self.config.get("api_providers", [])

        self.enable_llm_tool = self.config.get("enable_llm_tool", True)

        # 1. 收集所有供应商配置
        all_provider_configs: list[AdapterConfig] = []
        for provider_item in api_providers_raw:
            if not isinstance(provider_item, dict):
                continue

            # 这里的 provider_item 是 template_list 的一个项
            # AstrBot 的 template_list 项结构通常是：
            # {
            #    "__template_key": "gemini",
            #    "name": "...",
            #    "provider_id": "...",
            #    ...其他 items 中的字段
            # }
            adapter_type_str = provider_item.get("__template_key")
            if not adapter_type_str:
                continue

            try:
                adapter_type = AdapterType(adapter_type_str)
            except ValueError:
                logger.warning(f"[ImageGen] 忽略未知适配器类型: {adapter_type_str}")
                continue

            name = provider_item.get("name", "")
            base_url = (provider_item.get("base_url") or "").strip()
            api_keys = [k for k in provider_item.get("api_keys", []) if k]
            provider_id = (provider_item.get("provider_id") or "").strip()
            available_models = provider_item.get("available_models") or []
            proxy = (provider_item.get("proxy") or "").strip() or None

            # 如果配置了 provider_id，从系统提供商加载
            if provider_id:
                loaded_keys, loaded_base = self._load_provider_config(provider_id)
                if loaded_keys:
                    api_keys = loaded_keys
                if loaded_base:
                    base_url = loaded_base

            all_provider_configs.append(
                AdapterConfig(
                    type=adapter_type,
                    name=name,
                    base_url=self._clean_base_url(base_url),
                    api_keys=api_keys,
                    available_models=available_models,
                    provider_id=provider_id,
                    proxy=proxy,
                    timeout=gen_cfg.get("timeout", 180),
                    max_retry_attempts=gen_cfg.get("max_retry_attempts", 3),
                )
            )

        # 2. 获取当前选择的模型
        model_setting = gen_cfg.get("model", "")

        # 3. 匹配当前适配器
        matched_config = None
        current_model = ""

        if "/" in model_setting:
            try:
                target_provider_name, target_model = model_setting.split("/", 1)
                for cfg in all_provider_configs:
                    if cfg.name == target_provider_name:
                        matched_config = cfg
                        current_model = target_model
                        break
            except ValueError:
                logger.warning(f"[ImageGen] 模型设置格式错误: {model_setting}，期望格式为 '供应商/模型'")

        # 如果没匹配到（或者没设置），取第一个可用的
        if not matched_config and all_provider_configs:
            matched_config = all_provider_configs[0]
            current_model = (
                matched_config.available_models[0]
                if matched_config.available_models
                else ""
            )
            logger.info(
                f"[ImageGen] 未匹配到当前模型配置，默认使用: {matched_config.name}/{current_model}"
            )

        if matched_config:
            self.adapter_config = matched_config
            self.adapter_config.model = current_model
            # 将所有可用模型汇总，供切换指令使用，格式为 "供应商名称/模型名称"
            all_available_models = []
            for cfg in all_provider_configs:
                for m in cfg.available_models:
                    all_available_models.append(f"{cfg.name}/{m}")
            self.adapter_config.available_models = all_available_models
        else:
            self.adapter_config = None
            logger.error("[ImageGen] 未找到任何有效的生图模型配置")

        self.rate_limit_seconds = max(0, user_limits_cfg.get("rate_limit_seconds", 0))
        self.max_concurrent_tasks = max(1, gen_cfg.get("max_concurrent_tasks", 3))
        self.max_image_size_mb = max(1, user_limits_cfg.get("max_image_size_mb", 10))
        self.enable_daily_limit = user_limits_cfg.get("enable_daily_limit", False)
        self.daily_limit_count = max(1, user_limits_cfg.get("daily_limit_count", 10))

        self.max_cache_count = max(1, cache_cfg.get("max_cache_count", 100))
        self.cleanup_interval_hours = max(
            1, cache_cfg.get("cleanup_interval_hours", 24)
        )

        self.default_aspect_ratio = gen_cfg.get("default_aspect_ratio", "自动")
        self.default_resolution = gen_cfg.get("default_resolution", "1K")
        self.show_generation_info = gen_cfg.get("show_generation_info", False)
        self.show_model_info = gen_cfg.get("show_model_info", False)

        # 重新初始化信号量以应用新并发数
        if self.max_concurrent_tasks:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

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

        # 检查频率限制和每日限制
        check_result = self._check_rate_limit(user_id)
        if isinstance(check_result, str):
            yield event.plain_result(check_result)
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
            current_model_full = (
                f"{self.adapter_config.name}/{self.adapter_config.model}"
            )
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == current_model_full else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n当前使用: {current_model_full}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                raw_model = models[index]  # "供应商名称/模型名称"

                # 更新配置并重新加载
                self.config.setdefault("generation", {})["model"] = raw_model
                self.config.save_config()
                self._load_config()

                if self.generator:
                    self.generator.update_adapter(self.adapter_config)

                yield event.plain_result(f"✅ 模型已切换: {raw_model}")
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
    def _check_rate_limit(self, user_id: str) -> bool | str:
        """检查用户请求频率限制和每日限制。"""
        # 1. 检查频率限制
        if self.rate_limit_seconds > 0:
            now = time.time()
            last_ts = self.user_request_timestamps.get(user_id, 0)
            if now - last_ts < self.rate_limit_seconds:
                remaining = int(self.rate_limit_seconds - (now - last_ts))
                return f"❌ 请求过于频繁，请在 {remaining} 秒后再试"
            self.user_request_timestamps[user_id] = now

        # 2. 检查每日限制
        if self.enable_daily_limit:
            today = datetime.date.today().isoformat()
            if today not in self.usage_data:
                self.usage_data[today] = {}

            count = self.usage_data[today].get(user_id, 0)
            if count >= self.daily_limit_count:
                return f"❌ 您今日的生图额度已用完 ({self.daily_limit_count}次)，请明天再试"

        return True

    async def _fetch_images_from_event(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """从消息事件中提取图片（包括直接发送的图片、引用消息中的图片、被@用户的头像）。"""
        images_data: list[tuple[bytes, str]] = []

        if not event.message_obj or not event.message_obj.message:
            return images_data

        # 预扫描：记录引用消息的发送者以及各个 @ 出现次数，用于过滤自动 @
        reply_sender_id = None
        at_counts: dict[str, int] = {}

        for component in event.message_obj.message:
            if isinstance(component, Comp.Reply):
                if hasattr(component, "sender_id") and component.sender_id:
                    reply_sender_id = str(component.sender_id)
            elif isinstance(component, Comp.At):
                if hasattr(component, "qq") and component.qq != "all":
                    uid = str(component.qq)
                    at_counts[uid] = at_counts.get(uid, 0) + 1

        for component in event.message_obj.message:
            try:
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
                    if hasattr(component, "qq") and component.qq != "all":
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
            except Exception as e:
                logger.error(f"[ImageGen] 提取消息组件图片失败: {e}")
                continue
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
        """创建后台任务并添加到管理器中。"""
        return self.task_manager.create_task(coro)
        """创建后台任务并添加到管理器中。"""
        return self.task_manager.create_task(coro)

    async def get_avatar(self, user_id: str) -> bytes | None:
        """获取用户头像。"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            # 使用插件缓存目录
            file_name = f"avatar_{user_id}.jpg"
            path = os.path.join(self.cache_dir, file_name)
            path = await download_image_by_url(url, path=path)
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
                # 使用插件缓存目录
                file_name = f"ref_{hashlib.md5(url.encode()).hexdigest()[:10]}"
                path = os.path.join(self.cache_dir, file_name)
                path = await download_image_by_url(url, path=path)
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
        except Exception as exc:
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
            logger.info(
                f"[ImageGen] 当前适配器不支持指定比例，已忽略参数: {aspect_ratio}"
            )
            aspect_ratio = "自动"

        if not (capabilities & ImageCapability.RESOLUTION) and resolution != "1K":
            logger.info(
                f"[ImageGen] 当前适配器不支持指定分辨率，已忽略参数: {resolution}"
            )
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
        start_time = time.time()
        result = await self.generator.generate(
            GenerationRequest(
                prompt=prompt,
                images=images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )
        end_time = time.time()
        duration = end_time - start_time

        if result.error:
            logger.error(
                f"[ImageGen] 任务 {task_id} 生成失败，耗时: {duration:.2f}s, 错误: {result.error}"
            )
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 生成失败: {result.error}"),
            )
            return

        logger.info(
            f"[ImageGen] 任务 {task_id} 生成成功，耗时: {duration:.2f}s, 图片数量: {len(result.images) if result.images else 0}"
        )

        if not result.images:
            return

        # 记录使用次数
        if self.enable_daily_limit:
            today = datetime.date.today().isoformat()
            if today not in self.usage_data:
                self.usage_data[today] = {}
            self.usage_data[today][unified_msg_origin] = (
                self.usage_data[today].get(unified_msg_origin, 0) + 1
            )
            self._save_usage_data()

        chain = MessageChain()
        for img_bytes in result.images:
            try:
                # 保存到插件自定义缓存目录
                file_name = f"gen_{task_id}_{int(time.time())}_{hashlib.md5(img_bytes).hexdigest()[:6]}.png"
                file_path = os.path.join(self.cache_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                chain.file_image(file_path)
            except Exception as exc:
                logger.error(f"[ImageGen] 保存图片失败: {exc}")

        info_parts = []
        if self.show_generation_info:
            info_parts.append(
                f"✨ 生成成功！\n📊 耗时: {duration:.2f}s\n🖼️ 数量: {len(result.images)}张"
            )

        if self.show_model_info and self.adapter_config:
            info_parts.append(
                f"🤖 模型: {self.adapter_config.name}/{self.adapter_config.model}"
            )

        if self.enable_daily_limit:
            today = datetime.date.today().isoformat()
            count = self.usage_data.get(today, {}).get(unified_msg_origin, 0)
            info_parts.append(f"📅 今日用量: {count}/{self.daily_limit_count}")

        if info_parts:
            chain.message("\n" + "\n".join(info_parts))

        await self.context.send_message(unified_msg_origin, chain)

    async def terminate(self):
        """插件卸载时清理资源。"""
        try:
            if self.generator:
                await self.generator.close()
            await self.task_manager.cancel_all()
            await self.task_manager.cancel_all()
            logger.info("[ImageGen] 插件已卸载")
        except Exception as exc:
            logger.error(f"[ImageGen] 卸载清理出错: {exc}")
