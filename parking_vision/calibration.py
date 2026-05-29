"""
相机标定模块 - 基于棋盘格标定法估计相机内参与畸变系数
"""

import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


class CameraCalibrator:
    """基于棋盘格图案的相机标定器"""

    def __init__(self, board_size: tuple = (9, 6), square_size: float = 25.0):
        """
        Args:
            board_size: 棋盘格内角点数 (列数, 行数)
            square_size: 棋盘格方格边长(mm)
        """
        self.board_size = board_size
        self.square_size = square_size

        # 标定结果
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[list] = None
        self.tvecs: Optional[list] = None
        self.image_size: Optional[tuple] = None
        self.rms_error: float = 0.0

        # 棋盘格三维世界坐标模板
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[
            0:board_size[0], 0:board_size[1]
        ].T.reshape(-1, 2) * square_size

    def collect_from_video(
        self, video_path: str, frame_interval: int = 30, max_frames: int = 50
    ) -> list:
        """从视频流中采集标定图像

        Args:
            video_path: 视频文件路径或摄像头索引
            frame_interval: 采样间隔帧数
            max_frames: 最大采集帧数

        Returns:
            成功检测到角点的帧列表
        """
        if video_path.isdigit():
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"无法打开视频源: {video_path}")

        valid_frames = []
        frame_count = 0

        while len(valid_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                # 亚像素精度角点检测
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001
                )
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                valid_frames.append((frame, corners_refined))

        cap.release()
        self.image_size = (gray.shape[1], gray.shape[0])
        print(f"从视频中采集到 {len(valid_frames)} 帧有效标定图像")
        return valid_frames

    def collect_from_images(self, image_dir: str, pattern: str = "*.jpg") -> list:
        """从图像文件夹采集标定图像

        Args:
            image_dir: 标定图像所在目录
            pattern: 文件匹配模式

        Returns:
            成功检测到角点的图像列表
        """
        image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
        if not image_paths:
            raise FileNotFoundError(f"未找到匹配的图像: {image_dir}/{pattern}")

        valid_frames = []
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, self.board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001
                )
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                valid_frames.append((frame, corners_refined))

        if valid_frames:
            self.image_size = (
                valid_frames[0][0].shape[1],
                valid_frames[0][0].shape[0]
            )

        print(f"从目录中采集到 {len(valid_frames)} 帧有效标定图像")
        return valid_frames

    def calibrate(self, valid_frames: list) -> dict:
        """执行相机标定

        Args:
            valid_frames: collect_from_video/collect_from_images 返回的有效帧列表

        Returns:
            标定结果字典
        """
        if len(valid_frames) < 10:
            print(f"警告: 有效帧数({len(valid_frames)})较少，建议至少15帧以获得可靠标定")

        obj_points = [self.objp for _ in valid_frames]
        img_points = [corners for _, corners in valid_frames]

        self.rms_error, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = (
            cv2.calibrateCamera(
                obj_points, img_points, self.image_size, None, None
            )
        )

        print(f"标定完成 - RMS重投影误差: {self.rms_error:.4f} 像素")
        return self.get_params()

    def get_params(self) -> dict:
        """获取标定参数"""
        if self.camera_matrix is None:
            raise RuntimeError("尚未执行标定，请先调用 calibrate()")

        return {
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "image_size": self.image_size,
            "rms_error": self.rms_error,
        }

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """对图像进行去畸变校正"""
        if self.camera_matrix is None:
            raise RuntimeError("尚未执行标定")

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )

        # 裁剪有效区域
        x, y, w, h = roi
        if all(v > 0 for v in roi):
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def save_params(self, output_path: str):
        """保存标定参数到YAML文件"""
        if self.camera_matrix is None:
            raise RuntimeError("尚未执行标定")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rms_error": float(self.rms_error),
            "board_size": list(self.board_size),
            "square_size": float(self.square_size),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"标定参数已保存至: {output_path}")

    def load_params(self, param_path: str):
        """从YAML文件加载标定参数"""
        with open(param_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        self.image_size = tuple(data["image_size"])
        self.rms_error = data["rms_error"]
        self.board_size = tuple(data["board_size"])
        self.square_size = data["square_size"]

        print(f"已加载标定参数 (RMS误差: {self.rms_error:.4f})")
