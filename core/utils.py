"""
Utility functions for image generation plugin
图像生成插件的工具函数
"""

from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image

from astrbot.api import logger

from .types import ImageData


def detect_mime_type(data: bytes) -> str:
    """
    检测图片 MIME 类型

    Args:
        data: 图片二进制数据

    Returns:
        str: MIME 类型
    """
    mime = "application/octet-stream"
    if data.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif len(data) > 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"heim", b"heis"):
            mime = "image/heic"
        elif brand in (b"mif1", b"msf1", b"heif"):
            mime = "image/heif"

    logger.debug(f"[Image Utils] Detected MIME type: {mime}")
    return mime


def _sync_convert_image_format(image_data: bytes, mime_type: str) -> tuple[bytes, str]:
    """
    同步的图片格式转换逻辑（在线程池中执行）

    Args:
        image_data: 原始图片数据
        mime_type: 原始 MIME 类型

    Returns:
        tuple[bytes, str]: 转换后的图片数据和 MIME 类型
    """
    try:
        img = Image.open(BytesIO(image_data))

        # 处理透明图片
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode == "LA":
                img = img.convert("RGBA")

            # 此时 img.mode 一定是 RGBA，使用第4个通道作为 alpha mask
            background.paste(img, mask=img.split()[3])
            img = background

        # 转换为 JPEG
        output = BytesIO()
        img.save(output, format="JPEG", quality=95)

        logger.debug("[Image Utils] 图片格式转换成功")
        return output.getvalue(), "image/jpeg"

    except Exception as e:
        logger.error(f"[Image Utils] 图片格式转换失败: {e}")
        return image_data, mime_type


async def convert_image_format(image_data: ImageData) -> ImageData:
    """
    转换不支持的图片格式为 JPEG

    Args:
        image_data: 原始图片数据

    Returns:
        ImageData: 转换后的图片数据
    """
    real_mime = detect_mime_type(image_data.data)

    supported_formats = [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    ]

    if real_mime in supported_formats:
        return ImageData(data=image_data.data, mime_type=real_mime)

    logger.info(f"[Image Utils] 转换图片格式: {image_data.mime_type} -> image/jpeg")
    converted_data, converted_mime = await asyncio.to_thread(
        _sync_convert_image_format, image_data.data, image_data.mime_type
    )
    return ImageData(data=converted_data, mime_type=converted_mime)


async def convert_images_batch(images: list[ImageData]) -> list[ImageData]:
    """
    批量转换图片格式

    Args:
        images: 图片列表

    Returns:
        list[ImageData]: 转换后的图片列表
    """
    if not images:
        return []

    tasks = [convert_image_format(img) for img in images]
    return await asyncio.gather(*tasks)


def validate_aspect_ratio(aspect_ratio: str | None) -> bool:
    """
    验证宽高比格式是否有效

    Args:
        aspect_ratio: 宽高比字符串

    Returns:
        bool: 是否有效
    """
    if aspect_ratio is None:
        return True

    valid_ratios = [
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
    return aspect_ratio in valid_ratios


def validate_resolution(resolution: str | None) -> bool:
    """
    验证分辨率格式是否有效

    Args:
        resolution: 分辨率字符串

    Returns:
        bool: 是否有效
    """
    if resolution is None:
        return True

    valid_resolutions = ["1K", "2K", "4K"]
    return resolution in valid_resolutions
