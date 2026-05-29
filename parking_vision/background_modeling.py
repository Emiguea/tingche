"""
背景建模模块 - 基于高斯混合模型的长时背景学习

使用OpenCV BackgroundSubtractorMOG2从固定摄像头视频中
学习稳定背景，分离静态车位区域与动态车辆目标。
"""

from pathlib import Path

import cv2
import numpy as np


class BackgroundModeler:
    """高斯混合模型背景建模器

    封装MOG2算法，支持：
    - 逐帧更新背景模型
    - 提取前景掩码（含阴影标记）
    - 导出/加载学习到的背景图像
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1,
    ):
        """
        Args:
            history: 用于建模的历史帧数
            var_threshold: 像素与模型之间的马氏距离阈值
            detect_shadows: 是否检测阴影（阴影像素标记为127）
            learning_rate: 背景更新速率，-1表示自动
        """
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        self._frame_count = 0
        self._background = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def update(self, frame: np.ndarray) -> np.ndarray:
        """输入一帧图像，更新背景模型并返回原始前景掩码

        Args:
            frame: BGR图像

        Returns:
            前景掩码，255=前景，127=阴影，0=背景
        """
        lr = self.learning_rate if self.learning_rate >= 0 else -1
        fg_mask = self._subtractor.apply(frame, learningRate=lr)
        self._frame_count += 1
        self._background = self._subtractor.getBackgroundImage()
        return fg_mask

    def get_background(self) -> np.ndarray:
        """获取当前学习到的背景图像"""
        if self._background is None:
            bg = self._subtractor.getBackgroundImage()
            if bg is None:
                raise RuntimeError("背景模型尚未初始化，请先调用update()送入足够帧数")
            self._background = bg
        return self._background

    def save_model(self, path: str) -> None:
        """将背景图像保存到文件"""
        bg = self.get_background()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, bg)

    def load_model(self, path: str) -> np.ndarray:
        """加载已保存的背景图像"""
        bg = cv2.imread(path)
        if bg is None:
            raise IOError(f"无法加载背景模型文件: {path}")
        self._background = bg
        return bg
