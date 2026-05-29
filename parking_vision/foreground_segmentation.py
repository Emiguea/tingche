"""
前景分割模块 - 形态学处理与连通域车辆提取

对背景建模输出的原始前景掩码进行后处理：
1. 消除阴影伪影
2. 形态学开闭运算去噪
3. 基于距离变换分离粘连目标
4. 连通域分析提取候选车辆轮廓
5. 面积过滤剔除噪声区域
"""

from typing import List, Tuple

import cv2
import numpy as np


class ForegroundSegmenter:
    """前景分割与车辆候选区域提取"""

    def __init__(
        self,
        morph_kernel_size: int = 5,
        min_area: int = 500,
        max_area: int = 0,
        shadow_value: int = 127,
        split_overlap: bool = True,
        split_dist_ratio: float = 0.4,
    ):
        """
        Args:
            morph_kernel_size: 形态学核尺寸
            min_area: 最小连通域面积，低于此值视为噪声
            max_area: 最大连通域面积，0表示不限制
            shadow_value: MOG2中阴影像素的灰度值
            split_overlap: 是否启用粘连目标分离
            split_dist_ratio: 距离变换阈值比例，越小分离越激进
        """
        self.morph_kernel_size = morph_kernel_size
        self.min_area = min_area
        self.max_area = max_area
        self.shadow_value = shadow_value
        self.split_overlap = split_overlap
        self.split_dist_ratio = split_dist_ratio

        self._kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        # 闭运算用较小核，避免将相邻车辆连接
        close_size = max(3, morph_kernel_size - 2)
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )

    def remove_shadows(self, mask: np.ndarray) -> np.ndarray:
        """将阴影像素（127）置零，仅保留确定前景（255）"""
        result = mask.copy()
        result[result == self.shadow_value] = 0
        return result

    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """形态学开运算（去噪）+ 轻度闭运算（填充内部孔洞但不粘连相邻目标）"""
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel_close, iterations=1)
        return closed

    def split_connected(self, mask: np.ndarray) -> np.ndarray:
        """基于距离变换 + 分水岭分离粘连的相邻车辆

        对大面积连通域进行距离变换，找到各自的核心区域后
        用分水岭算法切分边界，返回分离后的标签掩码。
        """
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist, self.split_dist_ratio * dist.max(), 255, cv2.THRESH_BINARY
        )
        sure_fg = sure_fg.astype(np.uint8)

        # 确定背景区域（膨胀后非前景部分）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=2)

        # 未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 连通域标记作为分水岭种子
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # 分水岭需要3通道输入
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_3ch, markers)

        # 将分水岭结果转回二值掩码，边界线设为0
        result = mask.copy()
        result[markers == -1] = 0
        return result

    def extract_candidates(self, mask: np.ndarray) -> List[dict]:
        """通过连通域分析提取候选车辆区域

        对面积异常大的连通域进行分离处理。
        """
        # 如果启用粘连分离，对大区域做距离变换分离
        if self.split_overlap:
            work_mask = self._split_large_blobs(mask)
        else:
            work_mask = mask

        contours, _ = cv2.findContours(
            work_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue
            if self.max_area > 0 and area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            candidates.append({
                "bbox": (x, y, w, h),
                "area": int(area),
                "centroid": (cx, cy),
                "contour": contour,
            })

        candidates.sort(key=lambda c: c["area"], reverse=True)
        return candidates

    def _split_large_blobs(self, mask: np.ndarray) -> np.ndarray:
        """对宽高比异常或面积过大的连通域单独做距离变换分离"""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 估计单车面积上限：取所有连通域面积中位数的3倍
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= self.min_area]
        if len(areas) < 2:
            return mask

        median_area = float(np.median(areas))
        split_threshold = median_area * 3.0

        result = mask.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < split_threshold:
                continue

            # 这个连通域可能是粘连的多辆车
            x, y, w, h = cv2.boundingRect(contour)
            roi = mask[y:y+h, x:x+w].copy()

            # 在ROI上做距离变换分离
            split_roi = self.split_connected(roi)
            result[y:y+h, x:x+w] = split_roi

        return result

    def process(
        self, raw_mask: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        """完整前景分割流程

        Args:
            raw_mask: MOG2输出的原始前景掩码

        Returns:
            (clean_mask, candidates) - 清洁掩码和候选车辆列表
        """
        no_shadow = self.remove_shadows(raw_mask)
        clean_mask = self.apply_morphology(no_shadow)
        candidates = self.extract_candidates(clean_mask)
        return clean_mask, candidates
