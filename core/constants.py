"""常量定义模块。

集中管理项目中使用的常量，避免魔法字符串分散在代码中。
"""

from __future__ import annotations

# ========================== 日志常量 ==========================

LOG_PREFIX = "[ImageGen]"
"""统一的日志前缀。"""


# ========================== 安全设置 ==========================

GEMINI_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)
"""Gemini API 支持的安全类别列表。"""


# ========================== 默认配置值 ==========================

DEFAULT_TIMEOUT = 180
"""默认请求超时时间（秒）。"""

DEFAULT_MAX_RETRY_ATTEMPTS = 3
"""默认最大重试次数。"""

DEFAULT_ASPECT_RATIO = "自动"
"""默认宽高比。"""

DEFAULT_RESOLUTION = "1K"
"""默认分辨率。"""

DEFAULT_MAX_CONCURRENT_TASKS = 3
"""默认最大并发任务数。"""

DEFAULT_MAX_IMAGE_SIZE_MB = 10
"""默认最大图片大小（MB）。"""

DEFAULT_MAX_CACHE_COUNT = 100
"""默认最大缓存文件数量。"""

DEFAULT_CLEANUP_INTERVAL_HOURS = 24
"""默认缓存清理间隔（小时）。"""

DEFAULT_DAILY_LIMIT_COUNT = 10
"""默认每日生成限制次数。"""


# ========================== 分辨率映射 ==========================

# 1K 分辨率映射（适用于多种适配器）
RESOLUTION_1K_MAP = {
    "1:1": "1024x1024",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "16:9": "1024x576",
    "9:16": "576x1024",
    "3:2": "1024x640",
    "2:3": "640x1024",
}

# 2K 分辨率映射
RESOLUTION_2K_MAP = {
    "1:1": "2048x2048",
    "4:3": "2048x1536",
    "3:4": "1536x2048",
    "3:2": "2048x1360",
    "2:3": "1360x2048",
    "16:9": "2048x1152",
    "9:16": "1152x2048",
}


# ========================== 支持的宽高比 ==========================

SUPPORTED_ASPECT_RATIOS = (
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
)
"""工具参数中支持的宽高比列表。"""


# ========================== 支持的分辨率 ==========================

SUPPORTED_RESOLUTIONS = ("1K", "2K", "4K")
"""工具参数中支持的分辨率列表。"""


# ========================== API 端点 ==========================

GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
"""Gemini API 默认 Base URL。"""

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com"
"""OpenAI API 默认 Base URL。"""

GITEE_AI_DEFAULT_BASE_URL = "https://ai.gitee.com"
"""Gitee AI 默认 Base URL。"""

JIMENG_DEFAULT_BASE_URL = "http://localhost:5100"
"""Jimeng2API 默认 Base URL。"""


# ========================== 文件路径 ==========================

DATA_DIR = "data/plugin_data/astrbot_plugin_gemini_generation"
"""插件数据目录。"""

CACHE_SUBDIR = "cache"
"""缓存子目录名称。"""

USAGE_FILE_NAME = "usage.json"
"""使用统计文件名。"""
