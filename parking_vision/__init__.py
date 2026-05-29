"""
停车场车位识别系统 - 相机标定、图像预处理与前景分割模块

模块功能：
1. 基于棋盘格的相机标定（内参矩阵、畸变系数）
2. 自适应图像增强（CLAHE + 宽动态融合）
3. 像素坐标到停车场平面坐标的单应性映射
4. 高斯混合模型背景建模与前景分割
5. 形态学处理与候选车辆提取
"""

from .calibration import CameraCalibrator
from .preprocessing import ImagePreprocessor
from .homography import HomographyMapper
from .pipeline import ParkingLotPipeline
from .background_modeling import BackgroundModeler
from .foreground_segmentation import ForegroundSegmenter
from .segmentation_pipeline import SegmentationPipeline

__all__ = [
    "CameraCalibrator",
    "ImagePreprocessor",
    "HomographyMapper",
    "ParkingLotPipeline",
    "BackgroundModeler",
    "ForegroundSegmenter",
    "SegmentationPipeline",
]
