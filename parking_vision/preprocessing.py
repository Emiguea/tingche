"""
图像预处理模块 - 针对夜间低照度与白天强反光的自适应增强
"""

from typing import Optional

import cv2
import numpy as np


class ImagePreprocessor:
    """停车场监控图像自适应预处理器

    支持两种增强模式：
    - CLAHE自适应直方图均衡化：增强局部对比度，适应不均匀光照
    - 宽动态范围(WDR)融合：融合多曝光信息，平衡高光与暗部
    """

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_size: tuple = (8, 8),
        auto_mode: bool = True,
    ):
        """
        Args:
            clip_limit: CLAHE对比度限制阈值
            tile_size: CLAHE分块大小
            auto_mode: 是否根据图像亮度自动选择增强策略
        """
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.auto_mode = auto_mode

        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_size
        )

    def analyze_illumination(self, image: np.ndarray) -> dict:
        """分析图像光照条件

        Returns:
            光照分析结果：平均亮度、亮度标准差、场景类型
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # 计算过曝和欠曝像素比例
        overexposed_ratio = np.sum(gray > 240) / gray.size
        underexposed_ratio = np.sum(gray < 15) / gray.size

        # 判定场景类型
        if mean_brightness < 60:
            scene_type = "night_low_light"
        elif mean_brightness > 180 or overexposed_ratio > 0.15:
            scene_type = "day_glare"
        elif std_brightness > 80:
            scene_type = "high_dynamic_range"
        else:
            scene_type = "normal"

        return {
            "mean_brightness": float(mean_brightness),
            "std_brightness": float(std_brightness),
            "overexposed_ratio": float(overexposed_ratio),
            "underexposed_ratio": float(underexposed_ratio),
            "scene_type": scene_type,
        }

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """对图像应用CLAHE自适应直方图均衡化

        在LAB色彩空间的L通道上进行，保持色彩不失真
        """
        if len(image.shape) == 2:
            return self.clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_enhanced = self.clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def apply_wdr_fusion(self, image: np.ndarray) -> np.ndarray:
        """宽动态范围融合

        通过模拟多曝光并进行Mertens融合，恢复高光和暗部细节。
        适用于停车场入口等同时存在强光和阴影的场景。
        """
        # 基于gamma变换模拟不同曝光
        img_float = image.astype(np.float32) / 255.0

        # 欠曝版本（恢复高光细节）
        gamma_dark = np.clip(np.power(img_float, 2.2), 0, 1)
        exposure_dark = (gamma_dark * 255).astype(np.uint8)

        # 正常曝光
        exposure_normal = image.copy()

        # 过曝版本（恢复暗部细节）
        gamma_bright = np.clip(np.power(img_float, 0.45), 0, 1)
        exposure_bright = (gamma_bright * 255).astype(np.uint8)

        # Mertens曝光融合
        merge_mertens = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=1.0,
        )
        fusion = merge_mertens.process(
            [exposure_dark, exposure_normal, exposure_bright]
        )

        # 映射回8位
        fusion_8bit = np.clip(fusion * 255, 0, 255).astype(np.uint8)

        return fusion_8bit

    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """非局部均值去噪，适用于夜间高ISO噪声"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, 7, 21
            )
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    def process(
        self, image: np.ndarray, scene_type: Optional[str] = None
    ) -> np.ndarray:
        """根据场景类型自动选择并执行预处理流程

        Args:
            image: 输入BGR图像
            scene_type: 场景类型，None时自动检测

        Returns:
            预处理后的图像
        """
        if scene_type is None and self.auto_mode:
            analysis = self.analyze_illumination(image)
            scene_type = analysis["scene_type"]

        if scene_type == "night_low_light":
            # 夜间模式：去噪 -> CLAHE增强
            denoised = self.denoise(image, strength=12)
            result = self.apply_clahe(denoised)

        elif scene_type == "day_glare":
            # 强反光模式：WDR融合 -> 轻度CLAHE
            fused = self.apply_wdr_fusion(image)
            light_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            lab = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = light_clahe.apply(l)
            result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        elif scene_type == "high_dynamic_range":
            # 高动态范围：WDR融合
            result = self.apply_wdr_fusion(image)

        else:
            # 正常场景：仅轻度CLAHE
            result = self.apply_clahe(image)

        return result
