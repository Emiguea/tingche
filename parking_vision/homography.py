"""
单应性映射模块 - 建立像素坐标系到停车场平面坐标系的映射关系
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


class HomographyMapper:
    """像素坐标到停车场地面平面坐标的单应性变换

    通过选取停车场地面上已知物理坐标的标志点（如车位角点、地面标线交点），
    计算从图像像素平面到实际地面平面的单应性矩阵H。
    """

    def __init__(self):
        self.H: Optional[np.ndarray] = None  # 像素->世界
        self.H_inv: Optional[np.ndarray] = None  # 世界->像素
        self.reprojection_error: float = 0.0

    def compute_homography(
        self,
        pixel_points: np.ndarray,
        world_points: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_thresh: float = 3.0,
    ) -> np.ndarray:
        """计算单应性矩阵

        Args:
            pixel_points: 图像中的像素坐标，shape=(N,2)，N>=4
            world_points: 对应的停车场平面坐标(米)，shape=(N,2)
            method: 计算方法 (0=常规最小二乘, cv2.RANSAC, cv2.LMEDS)
            ransac_thresh: RANSAC内点阈值(像素)

        Returns:
            3x3单应性矩阵
        """
        pixel_points = np.array(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
        world_points = np.array(world_points, dtype=np.float64).reshape(-1, 1, 2)

        if len(pixel_points) < 4:
            raise ValueError("至少需要4对对应点")

        self.H, mask = cv2.findHomography(
            pixel_points, world_points, method, ransac_thresh
        )

        if self.H is None:
            raise RuntimeError("单应性计算失败，请检查输入点是否共线")

        self.H_inv = np.linalg.inv(self.H)

        # 计算重投影误差
        self.reprojection_error = self._compute_reprojection_error(
            pixel_points, world_points, mask
        )

        inlier_count = int(mask.sum()) if mask is not None else len(pixel_points)
        print(
            f"单应性计算完成 - 内点数: {inlier_count}/{len(pixel_points)}, "
            f"重投影误差: {self.reprojection_error:.4f} m"
        )

        return self.H

    def _compute_reprojection_error(
        self, pixel_pts: np.ndarray, world_pts: np.ndarray, mask: Optional[np.ndarray]
    ) -> float:
        """计算内点的平均重投影误差"""
        pixel_pts_2d = pixel_pts.reshape(-1, 2)
        world_pts_2d = world_pts.reshape(-1, 2)

        projected = self.pixel_to_world(pixel_pts_2d)
        errors = np.linalg.norm(projected - world_pts_2d, axis=1)

        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
            errors = errors[mask_flat]

        return float(np.mean(errors))

    def pixel_to_world(self, pixel_points: np.ndarray) -> np.ndarray:
        """将像素坐标转换为停车场平面坐标

        Args:
            pixel_points: 像素坐标，shape=(N,2) 或 (2,)

        Returns:
            停车场平面坐标(米)，shape=(N,2)
        """
        if self.H is None:
            raise RuntimeError("尚未计算单应性矩阵")

        pts = np.array(pixel_points, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 2)

        # 齐次坐标变换
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])  # (N, 3)
        world_h = (self.H @ pts_h.T).T  # (N, 3)

        # 归一化
        world_2d = world_h[:, :2] / world_h[:, 2:3]

        if single_point:
            return world_2d[0]
        return world_2d

    def world_to_pixel(self, world_points: np.ndarray) -> np.ndarray:
        """将停车场平面坐标转换为像素坐标（反向映射）

        Args:
            world_points: 平面坐标(米)，shape=(N,2) 或 (2,)

        Returns:
            像素坐标，shape=(N,2)
        """
        if self.H_inv is None:
            raise RuntimeError("尚未计算单应性矩阵")

        pts = np.array(world_points, dtype=np.float64)
        single_point = pts.ndim == 1
        if single_point:
            pts = pts.reshape(1, 2)

        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        pixel_h = (self.H_inv @ pts_h.T).T

        pixel_2d = pixel_h[:, :2] / pixel_h[:, 2:3]

        if single_point:
            return pixel_2d[0]
        return pixel_2d

    def warp_to_bev(
        self, image: np.ndarray, output_size: tuple = (800, 800),
        meters_per_pixel: float = 0.05,
    ) -> np.ndarray:
        """将原始图像透视变换为鸟瞰图(BEV)

        Args:
            image: 输入图像
            output_size: 输出鸟瞰图尺寸 (width, height)
            meters_per_pixel: 每像素对应的实际距离(米)

        Returns:
            鸟瞰图图像
        """
        if self.H is None:
            raise RuntimeError("尚未计算单应性矩阵")

        # 构造缩放矩阵：世界坐标(米) -> 鸟瞰图像素
        S = np.array([
            [1.0 / meters_per_pixel, 0, output_size[0] / 2],
            [0, 1.0 / meters_per_pixel, output_size[1] / 2],
            [0, 0, 1],
        ], dtype=np.float64)

        # 组合变换：像素 -> 世界 -> 鸟瞰图像素
        H_bev = S @ self.H

        bev = cv2.warpPerspective(image, H_bev, output_size)
        return bev

    def save(self, output_path: str):
        """保存单应性矩阵到文件"""
        if self.H is None:
            raise RuntimeError("尚未计算单应性矩阵")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "homography_matrix": self.H.tolist(),
            "homography_inverse": self.H_inv.tolist(),
            "reprojection_error_m": float(self.reprojection_error),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"单应性矩阵已保存至: {output_path}")

    def load(self, param_path: str):
        """从文件加载单应性矩阵"""
        with open(param_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.H = np.array(data["homography_matrix"], dtype=np.float64)
        self.H_inv = np.array(data["homography_inverse"], dtype=np.float64)
        self.reprojection_error = data["reprojection_error_m"]

        print(f"已加载单应性矩阵 (重投影误差: {self.reprojection_error:.4f} m)")

    def interactive_select_points(self, image: np.ndarray, num_points: int = 4) -> np.ndarray:
        """交互式选取图像中的对应点

        在图像上点击标志点，返回像素坐标。
        配合已知的世界坐标使用。

        Args:
            image: 输入图像
            num_points: 需要选取的点数

        Returns:
            选取的像素坐标，shape=(num_points, 2)
        """
        points = []
        display = image.copy()

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
                points.append([x, y])
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    display, f"P{len(points)}",
                    (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2,
                )
                cv2.imshow("Select Points", display)

        cv2.namedWindow("Select Points", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Points", on_mouse)
        cv2.imshow("Select Points", display)

        print(f"请在图像上点击 {num_points} 个标志点，按ESC退出")
        while len(points) < num_points:
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                break

        cv2.destroyWindow("Select Points")
        return np.array(points, dtype=np.float64)
