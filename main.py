"""
AstrBot Image Generation Plugin
通用图像生成插件
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, CommandResult, MessageChain, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config.astrbot_config import AstrBotConfig

from .adapter import GeminiAdapter, GeminiOpenAIAdapter, GeminiZaiAdapter
from .core import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    ImageData,
    ImageGenerator,
    convert_images_batch,
    detect_mime_type,
)


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """通用图像生成工具，支持多种适配器"""

    name: str = "generate_image"
    description: str = "使用AI模型生成或修改图片"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的提示词（直接将用户发送的内容原样传递给模型）",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比",
                    "enum": [
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
                },
                "resolution": {
                    "type": "string",
                    "description": "图片分辨率",
                    "enum": ["1K", "2K", "4K"],
                },
                "preset": {
                    "type": "string",
                    "description": "使用预设名称（可选）",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "需要作为参考的用户头像列表。支持: 'self'(机器人头像)、'sender'(发送者头像)、或具体的用户ID",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        """执行图像生成"""
        if not (prompt := kwargs.get("prompt", "")):
            return "请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化"

        # 获取事件上下文
        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning("[ImageGen] Tool call context missing event")
            return "❌ 无法获取当前消息上下文"

        # 检查是否有可用的适配器
        if not plugin.generator._adapters:
            return "❌ 未配置任何图像生成适配器"

        # 获取参考图片
        images_data = await plugin._extract_reference_images(event)

        # 处理头像引用参数
        avatar_references = kwargs.get("avatar_references", [])
        if avatar_references and isinstance(avatar_references, list):
            for ref in avatar_references:
                if not isinstance(ref, str):
                    continue

                ref = ref.strip().lower()
                user_id = None

                if ref == "self":
                    # 获取机器人自己的头像
                    user_id = str(event.get_self_id())
                elif ref == "sender":
                    # 获取发送者的头像
                    user_id = str(event.get_sender_id() or event.unified_msg_origin)
                else:
                    # 作为用户ID处理
                    user_id = ref

                if user_id:
                    avatar_data = await plugin._get_user_avatar(user_id)
                    if avatar_data:
                        images_data.append(
                            ImageData(data=avatar_data, mime_type="image/jpeg")
                        )
                        logger.info(
                            f"[ImageGen] 已添加用户 {user_id} 的头像作为参考图片"
                        )

        # 应用预设
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        resolution = kwargs.get("resolution", "1K")
        preset_name = kwargs.get("preset")

        if preset_name and preset_name in plugin.presets:
            preset = plugin.presets[preset_name]
            prompt = preset.get("prompt_template", "{prompt}").format(prompt=prompt)
            aspect_ratio = preset.get("aspect_ratio", aspect_ratio)
            resolution = preset.get("resolution", resolution)

        # 处理自动比例：当比例为 "自动" 时，不传递给模型
        if aspect_ratio == "自动":
            aspect_ratio = None

        # 创建生成请求
        request = GenerationRequest(
            prompt=prompt,
            reference_images=images_data,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=f"{event.get_sender_id()}_{int(time.time())}",
        )

        # 创建后台任务
        task = asyncio.create_task(
            plugin._generate_and_send(request, event, plugin.context)
        )
        plugin.background_tasks.add(task)
        task.add_done_callback(plugin.background_tasks.discard)

        mode = "图生图" if images_data else "文生图"
        return f"已启动{mode}任务"


class ImageGenerationPlugin(Star):
    """通用图像生成插件"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.base_config = config
        self.generator = ImageGenerator()

        # 配置
        self.config = {}
        self.current_adapter_type = None

        # 预设管理
        self.presets = {}
        self.preset_file = (
            Path(self.base_config.get("data_path", "data"))
            / "image_gen_presets.json"
        )

        # 速率限制
        self.user_last_request = {}
        self.rate_limit_seconds = 0

        # 并发控制
        self.semaphore = None
        self.max_concurrent_tasks = 3

        # 后台任务
        self.background_tasks = set()

    async def on_load(self):
        """插件加载时初始化"""
        try:
            # 加载配置
            self.config = self.context.get_config()

            # 初始化适配器
            await self._initialize_adapters()

            # 加载预设
            self._load_presets()

            # 初始化速率限制和并发控制（从 generation 配置组读取）
            generation_config = self.config.get("generation", {})
            self.rate_limit_seconds = generation_config.get("rate_limit_seconds", 0)
            self.max_concurrent_tasks = generation_config.get("max_concurrent_tasks", 3)
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

            # 注册 LLM 工具
            enable_llm_tool = self.config.get("enable_llm_tool", True)
            if enable_llm_tool:
                self.context.add_llm_tools(ImageGenerationTool(plugin=self))
                logger.info("[ImageGen] 已注册图像生成工具（支持头像引用）")

            logger.info("[ImageGen] 插件加载成功")

        except Exception as e:
            logger.error(f"[ImageGen] 插件加载失败: {e}", exc_info=True)
            raise

    def _map_adapter_type(self, adapter_type_str: str) -> AdapterType:
        """将配置中的适配器类型字符串映射到 AdapterType 枚举"""
        adapter_type_map = {
            "gemini": AdapterType.GEMINI,
            "gemini(OpenAI)": AdapterType.OPENAI,
            "gemini(Zai)": AdapterType.ZAI,
        }
        return adapter_type_map.get(adapter_type_str, AdapterType.GEMINI)

    async def _initialize_adapters(self):
        """初始化适配器"""
        try:
            # 获取适配器配置组
            adapter_config = self.config.get("adapter", {})
            generation_config = self.config.get("generation", {})

            # 获取适配器类型
            adapter_type_str = adapter_config.get("type", "gemini")
            adapter_type = self._map_adapter_type(adapter_type_str)

            # 尝试从系统供应商加载配置
            provider_id = (adapter_config.get("provider_id", "") or "").strip()
            api_keys = []
            base_url = ""

            if provider_id and self._load_provider_config(provider_id, adapter_type):
                # 从系统供应商成功加载配置
                api_keys = self._provider_api_keys
                base_url = self._provider_base_url
            else:
                # 使用插件配置
                api_keys = adapter_config.get("api_keys", [])
                base_url = adapter_config.get("base_url", "")

            # 获取其他适配器连接配置
            model = adapter_config.get("model", "")
            proxy = adapter_config.get("proxy")

            # 获取生图配置
            timeout = generation_config.get("timeout", 300)
            max_retry_attempts = generation_config.get("max_retry_attempts", 3)

            # 如果 base_url 为空，根据适配器类型设置默认值
            if not base_url:
                base_url = self._get_default_base_url(adapter_type)

            # 如果 model 为空，根据适配器类型设置默认值
            if not model:
                model = self._get_default_model(adapter_type)

            # 构建适配器配置
            config = AdapterConfig(
                adapter_type=adapter_type,
                api_keys=api_keys,
                base_url=base_url,
                model=model,
                timeout=timeout,
                max_retry_attempts=max_retry_attempts,
                proxy=proxy,
                extra_config={},
            )

            # 创建适配器实例
            adapter = self._create_adapter(adapter_type, config)
            if adapter:
                self.generator.register_adapter(adapter)
                self.generator.set_current_adapter(adapter_type_str)
                self.current_adapter_type = adapter_type_str
                logger.info(f"[ImageGen] 已初始化适配器: {adapter_type_str}")
            else:
                logger.error(f"[ImageGen] 创建适配器失败: {adapter_type_str}")

        except Exception as e:
            logger.error(f"[ImageGen] 初始化适配器失败: {e}", exc_info=True)

    def _get_default_base_url(self, adapter_type: AdapterType) -> str:
        """根据适配器类型获取默认 base_url"""
        default_urls = {
            AdapterType.GEMINI: "https://generativelanguage.googleapis.com",
            AdapterType.OPENAI: "https://api.openai.com",
            AdapterType.ZAI: "https://api.zai.one",
        }
        return default_urls.get(adapter_type, "")

    def _clean_base_url(self, url: str) -> str:
        """清洗 Base URL"""
        if not url:
            return ""
        url = url.rstrip("/")
        # 移除 /v1 及其后的所有内容 (包括 /v1beta, /v1/chat 等)
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_provider_config(self, provider_id: str, adapter_type: AdapterType) -> bool:
        """从系统供应商加载配置

        Returns:
            True 如果成功加载
            False 如果加载失败
        """
        try:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.warning(f"[ImageGen] 未找到提供商 {provider_id}，将使用插件配置")
                return False

            provider_config = getattr(provider, "provider_config", {}) or {}

            # 提取 keys
            api_keys = []
            for key_field in ["key", "keys", "api_key", "access_token"]:
                if keys := provider_config.get(key_field):
                    api_keys = [keys] if isinstance(keys, str) else [k for k in keys if k]
                    break

            # 提取 base_url
            api_base = (
                getattr(provider, "api_base", None)
                or provider_config.get("api_base")
                or provider_config.get("api_base_url")
            )

            if not api_keys:
                logger.warning(f"[ImageGen] 提供商 {provider_id} 未提供可用的 API Key")
                return False

            # 直接设置实例变量
            self._provider_api_keys = api_keys
            self._provider_base_url = self._clean_base_url(
                api_base or self._get_default_base_url(adapter_type)
            )

            logger.info(f"[ImageGen] 使用系统提供商: {provider_id}")
            return True

        except Exception as e:
            logger.error(f"[ImageGen] 从提供商 {provider_id} 加载配置失败: {e}")
            return False

    def _get_default_model(self, adapter_type: AdapterType) -> str:
        """根据适配器类型获取默认 model"""
        default_models = {
            AdapterType.GEMINI: "gemini-2.5-flash-image-preview",
            AdapterType.OPENAI: "gpt-4o",
            AdapterType.ZAI: "gemini-2.5-flash-image",
        }
        return default_models.get(adapter_type, "")

    def _get_available_models(self, adapter_type: AdapterType) -> list[str]:
        """根据适配器类型获取可用模型列表"""
        # 先从配置中读取
        adapter_config = self.config.get("adapter", {})
        available_models = adapter_config.get("available_models", [])

        if available_models:
            return available_models

        # 如果配置为空，返回默认列表
        default_available_models = {
            AdapterType.GEMINI: [
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.5-flash-image",
                "gemini-2.5-flash-image-preview",
                "gemini-3-pro-image-preview"
            ],
            AdapterType.OPENAI: [
                "gpt-4o",
                "gpt-4-turbo",
                "dall-e-3",
                "dall-e-2"
            ],
            AdapterType.ZAI: [
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.5-flash-image",
                "gemini-2.5-flash-image-preview"
            ],
        }
        return default_available_models.get(adapter_type, [])

    def _create_adapter(self, adapter_type: AdapterType, config: AdapterConfig):
        """创建适配器实例"""
        adapter_map = {
            AdapterType.GEMINI: GeminiAdapter,
            AdapterType.OPENAI: GeminiOpenAIAdapter,
            AdapterType.ZAI: GeminiZaiAdapter,
        }

        adapter_class = adapter_map.get(adapter_type)
        if adapter_class:
            return adapter_class(config)
        return None

    async def _switch_model(self, model: str):
        """切换模型（重新创建适配器实例）"""
        # 获取适配器配置组
        adapter_config = self.config.get("adapter", {})
        generation_config = self.config.get("generation", {})

        # 获取适配器类型
        adapter_type_str = adapter_config.get("type", "gemini")
        adapter_type = self._map_adapter_type(adapter_type_str)

        # 获取适配器连接配置
        api_keys = adapter_config.get("api_keys", [])
        base_url = adapter_config.get("base_url", "")
        proxy = adapter_config.get("proxy")

        # 获取生图配置
        timeout = generation_config.get("timeout", 300)
        max_retry_attempts = generation_config.get("max_retry_attempts", 3)

        # 如果 base_url 为空，根据适配器类型设置默认值
        if not base_url:
            base_url = self._get_default_base_url(adapter_type)

        # 构建适配器配置
        config = AdapterConfig(
            adapter_type=adapter_type,
            api_keys=api_keys,
            base_url=base_url,
            model=model,  # 使用新的模型
            timeout=timeout,
            max_retry_attempts=max_retry_attempts,
            proxy=proxy,
            extra_config={},
        )

        # 创建适配器实例
        adapter = self._create_adapter(adapter_type, config)
        if adapter:
            # 重新注册适配器（会覆盖旧的）
            self.generator.register_adapter(adapter)
            self.generator.set_current_adapter(adapter_type_str)
            logger.info(f"[ImageGen] 已切换到模型: {model}")
        else:
            raise Exception(f"创建适配器失败: {adapter_type_str}")

    def _load_presets(self):
        """加载预设"""
        self.presets = {}

        # 从配置中加载预设
        config_presets = self.config.get("presets", [])
        for preset_str in config_presets:
            try:
                self._parse_preset_string(preset_str)
            except Exception as e:
                logger.error(f"[ImageGen] 解析预设失败: {preset_str[:50]}... - {e}")

        # 从文件中加载用户自定义预设（会覆盖配置中的同名预设）
        if self.preset_file.exists():
            try:
                with open(self.preset_file, encoding="utf-8") as f:
                    file_presets = json.load(f)
                    self.presets.update(file_presets)
            except Exception as e:
                logger.error(f"[ImageGen] 加载预设文件失败: {e}")

        logger.info(f"[ImageGen] 已加载 {len(self.presets)} 个预设")

    def _parse_preset_string(self, preset_str: str):
        """解析预设字符串（支持简单格式和 JSON 格式）"""
        if ":" not in preset_str:
            return

        name, _, content = preset_str.partition(":")
        name = name.strip()
        content = content.strip()

        if not name or not content:
            return

        # 尝试解析为 JSON 格式
        if content.startswith("{"):
            try:
                preset_data = json.loads(content)
                self.presets[name] = preset_data
                return
            except json.JSONDecodeError:
                pass

        # 简单格式：直接使用 prompt
        self.presets[name] = {
            "prompt_template": content,
            "description": f"预设: {name}",
        }

    def _save_presets(self):
        """保存预设"""
        try:
            self.preset_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.preset_file, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, ensure_ascii=False, indent=2)
            logger.info(f"[ImageGen] 已保存 {len(self.presets)} 个预设")
        except Exception as e:
            logger.error(f"[ImageGen] 保存预设失败: {e}")

    @filter.command("生图")
    async def cmd_generate(self, message: AstrMessageEvent, context: Context):
        """生成图像命令"""
        try:
            # 速率限制检查
            if not self._check_rate_limit(message.get_sender_id()):
                return CommandResult().message("请求过于频繁，请稍后再试")

            # 解析命令参数
            args = message.message_str.strip()
            if not args:
                return CommandResult().message(
                    "请提供生图提示词，例如：/生图 一只可爱的猫"
                )

            # 解析参数
            prompt, aspect_ratio, resolution, preset_name = self._parse_generate_args(
                args
            )

            # 应用预设
            if preset_name and preset_name in self.presets:
                preset = self.presets[preset_name]
                prompt = preset.get("prompt_template", "{prompt}").format(prompt=prompt)
                aspect_ratio = preset.get("aspect_ratio", aspect_ratio)
                resolution = preset.get("resolution", resolution)

            # 提取参考图
            reference_images = await self._extract_reference_images(message)

            # 处理自动比例：当比例为 "自动" 时，不传递给模型
            if aspect_ratio == "自动":
                aspect_ratio = None

            # 创建生成请求
            request = GenerationRequest(
                prompt=prompt,
                reference_images=reference_images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=f"{message.get_sender_id()}_{int(time.time())}",
            )

            # 发送处理中消息
            await context.send_message(
                MessageChain().plain(
                    f"正在生成图像，请稍候...\n提示词: {prompt[:50]}..."
                ),
                message,
            )

            # 异步生成图像
            task = asyncio.create_task(
                self._generate_and_send(request, message, context)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

            return CommandResult()

        except Exception as e:
            logger.error(f"[ImageGen] 生图命令失败: {e}", exc_info=True)
            return CommandResult().message(f"生图失败: {str(e)}")

    def _parse_generate_args(self, args: str) -> tuple[str, str, str, str | None]:
        """解析生图命令参数"""
        aspect_ratio = "1:1"
        resolution = "1K"
        preset_name = None
        prompt = args

        # 解析 --ratio 参数
        ratio_match = re.search(r"--ratio\s+(\S+)", args)
        if ratio_match:
            aspect_ratio = ratio_match.group(1)
            prompt = prompt.replace(ratio_match.group(0), "").strip()

        # 解析 --res 参数
        res_match = re.search(r"--res\s+(\S+)", args)
        if res_match:
            resolution = res_match.group(1)
            prompt = prompt.replace(res_match.group(0), "").strip()

        # 解析 --preset 参数
        preset_match = re.search(r"--preset\s+(\S+)", args)
        if preset_match:
            preset_name = preset_match.group(1)
            prompt = prompt.replace(preset_match.group(0), "").strip()

        return prompt, aspect_ratio, resolution, preset_name

    async def _generate_and_send(
        self, request: GenerationRequest, message: AstrMessageEvent, context: Context
    ):
        """生成图像并发送"""
        async with self.semaphore:
            try:
                # 生成图像
                result = await self.generator.generate(request)

                if not result.success:
                    await context.send_message(
                        MessageChain().plain(f"生成失败: {result.error_message}"),
                        message,
                    )
                    return

                # 转换图像格式
                converted_images = await convert_images_batch(
                    [ImageData(data=img) for img in result.images]
                )

                # 发送图像
                for img_data in converted_images:
                    await context.send_message(
                        MessageChain().image(img_data.data),
                        message,
                    )

                logger.info(f"[ImageGen] 成功生成 {len(result.images)} 张图像")

            except Exception as e:
                logger.error(f"[ImageGen] 生成图像失败: {e}", exc_info=True)
                await context.send_message(
                    MessageChain().plain(f"生成失败: {str(e)}"),
                    message,
                )

    @filter.command("生图模型")
    async def cmd_switch_model(self, message: AstrMessageEvent, context: Context):
        """切换生图模型"""
        try:
            args = message.message_str.strip()

            # 获取当前适配器类型
            adapter_config = self.config.get("adapter", {})
            adapter_type_str = adapter_config.get("type", "gemini")
            adapter_type = self._map_adapter_type(adapter_type_str)

            # 获取当前模型
            current_model = adapter_config.get("model", "")
            if not current_model:
                current_model = self._get_default_model(adapter_type)

            # 获取可用模型列表
            available_models = self._get_available_models(adapter_type)

            if not args:
                # 显示当前模型和可用模型
                msg = f"当前适配器: {adapter_type_str}\n当前模型: {current_model}\n可用模型: {', '.join(available_models)}"
                return CommandResult().message(msg)

            # 切换模型
            if args not in available_models:
                return CommandResult().message(f"模型 {args} 不在可用模型列表中\n可用模型: {', '.join(available_models)}")

            try:
                await self._switch_model(args)
                return CommandResult().message(f"已切换到模型: {args}")
            except Exception as e:
                return CommandResult().message(f"切换失败: {str(e)}")

        except Exception as e:
            logger.error(f"[ImageGen] 切换模型失败: {e}", exc_info=True)
            return CommandResult().message(f"切换失败: {str(e)}")

    @filter.command("预设")
    async def cmd_preset(self, message: AstrMessageEvent, context: Context):
        """预设管理命令"""
        try:
            args = message.message_str.strip()

            if not args:
                # 列出所有预设
                if not self.presets:
                    return CommandResult().message("暂无预设")

                msg = "可用预设:\n"
                for name, preset in self.presets.items():
                    desc = preset.get("description", "无描述")
                    msg += f"- {name}: {desc}\n"
                return CommandResult().message(msg.strip())

            # 解析子命令
            parts = args.split(maxsplit=1)
            subcmd = parts[0]

            if subcmd == "add" and len(parts) > 1:
                # 添加预设: /预设 add <name> <json>
                return await self._add_preset(parts[1])
            elif subcmd == "del" and len(parts) > 1:
                # 删除预设: /预设 del <name>
                name = parts[1].strip()
                if name in self.presets:
                    del self.presets[name]
                    self._save_presets()
                    return CommandResult().message(f"已删除预设: {name}")
                return CommandResult().message(f"预设不存在: {name}")

            elif subcmd == "show" and len(parts) > 1:
                # 显示预设详情: /预设 show <name>
                name = parts[1].strip()
                if name in self.presets:
                    preset_json = json.dumps(
                        self.presets[name], ensure_ascii=False, indent=2
                    )
                    return CommandResult().message(f"预设 {name}:\n{preset_json}")
                return CommandResult().message(f"预设不存在: {name}")

            else:
                return CommandResult().message(
                    "用法:\n"
                    "/预设 - 列出所有预设\n"
                    "/预设 add <name> <json> - 添加预设\n"
                    "/预设 del <name> - 删除预设\n"
                    "/预设 show <name> - 显示预设详情"
                )

        except Exception as e:
            logger.error(f"[ImageGen] 预设命令失败: {e}", exc_info=True)
            return CommandResult().message(f"操作失败: {str(e)}")

    async def _add_preset(self, args: str) -> CommandResult:
        """添加预设"""
        try:
            # 解析 name 和 json
            parts = args.split(maxsplit=1)
            if len(parts) < 2:
                return CommandResult().message("用法: /预设 add <name> <json>")

            name = parts[0].strip()
            json_str = parts[1].strip()

            # 解析 JSON
            preset = json.loads(json_str)

            # 验证预设格式
            if "prompt_template" not in preset:
                return CommandResult().message("预设必须包含 prompt_template 字段")

            # 保存预设
            self.presets[name] = preset
            self._save_presets()

            return CommandResult().message(f"已添加预设: {name}")

        except json.JSONDecodeError as e:
            return CommandResult().message(f"JSON 解析失败: {str(e)}")
        except Exception as e:
            return CommandResult().message(f"添加失败: {str(e)}")

    async def _extract_reference_images(
        self, message: AstrMessageEvent
    ) -> list[ImageData]:
        """从消息中提取参考图（包括直接图片、回复消息图片、@用户头像）"""
        images = []

        try:
            if not message.message:
                return images

            # 预扫描：获取回复发送者ID和统计At次数
            reply_sender_id = None
            at_counts = {}

            for component in message.message:
                # 检测回复消息
                if (
                    hasattr(component, "__class__")
                    and component.__class__.__name__ == "Reply"
                ):
                    if hasattr(component, "sender_id") and component.sender_id:
                        reply_sender_id = str(component.sender_id)
                # 统计At次数
                elif (
                    hasattr(component, "__class__")
                    and component.__class__.__name__ == "At"
                ):
                    if hasattr(component, "qq") and component.qq != "all":
                        uid = str(component.qq)
                        at_counts[uid] = at_counts.get(uid, 0) + 1

            # 遍历消息组件提取图片
            for component in message.message:
                # 1. 处理直接发送的图片
                if isinstance(component, Image):
                    try:
                        image_data = await component.get_bytes()
                        mime_type = detect_mime_type(image_data)
                        images.append(ImageData(data=image_data, mime_type=mime_type))
                        logger.debug("[ImageGen] 提取到直接图片")
                    except Exception as e:
                        logger.error(f"[ImageGen] 提取图片失败: {e}")

                # 2. 从回复消息中提取图片
                elif (
                    hasattr(component, "__class__")
                    and component.__class__.__name__ == "Reply"
                ):
                    if hasattr(component, "chain") and component.chain:
                        for sub_comp in component.chain:
                            if isinstance(sub_comp, Image):
                                try:
                                    image_data = await sub_comp.get_bytes()
                                    mime_type = detect_mime_type(image_data)
                                    images.append(
                                        ImageData(data=image_data, mime_type=mime_type)
                                    )
                                    logger.debug("[ImageGen] 提取到回复消息中的图片")
                                except Exception as e:
                                    logger.error(f"[ImageGen] 提取回复图片失败: {e}")

                # 3. 从@用户头像中提取图片
                elif (
                    hasattr(component, "__class__")
                    and component.__class__.__name__ == "At"
                ):
                    if hasattr(component, "qq") and component.qq != "all":
                        uid = str(component.qq)

                        # 智能过滤：忽略回复消息带来的自动@
                        if reply_sender_id and uid == reply_sender_id:
                            if at_counts.get(uid, 0) == 1:
                                logger.debug(f"[ImageGen] 忽略回复消息的自动@ {uid}")
                                continue

                        # 智能过滤：忽略触发机器人的单次@
                        self_id = str(message.get_self_id()).strip()
                        if self_id and uid == self_id:
                            if at_counts.get(uid, 0) == 1:
                                logger.debug(f"[ImageGen] 忽略机器人触发@ {uid}")
                                continue

                        # 获取用户头像
                        avatar_data = await self._get_user_avatar(uid)
                        if avatar_data:
                            images.append(
                                ImageData(data=avatar_data, mime_type="image/jpeg")
                            )
                            logger.debug(f"[ImageGen] 提取到用户 {uid} 的头像")

        except Exception as e:
            logger.error(f"[ImageGen] 提取参考图失败: {e}", exc_info=True)

        return images

    async def _get_user_avatar(self, user_id: str) -> bytes | None:
        """获取用户头像（QQ头像）"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.read()
        except Exception as e:
            logger.error(f"[ImageGen] 获取用户 {user_id} 头像失败: {e}")
        return None

    def _check_rate_limit(self, user_id: str) -> bool:
        """检查速率限制"""
        if self.rate_limit_seconds <= 0:
            return True

        now = time.time()
        last_request = self.user_last_request.get(user_id, 0)

        if now - last_request < self.rate_limit_seconds:
            return False

        self.user_last_request[user_id] = now
        return True
