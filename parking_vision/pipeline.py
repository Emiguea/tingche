"""
完整处理流水线 - 整合标定、预处理与坐标映射
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraCalibrator
from .preprocessing import ImagePreprocessor
from .homography import HomographyMapper


def _open_video(video_path: str) -> cv2.VideoCapture:
    """打开视频文件，兼容含中文/特殊字符的路径（Windows）"""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap

    if sys.platform == "win32":
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        ret_val = ctypes.windll.kernel32.GetShortPathNameW(
            str(Path(video_path).resolve()), buf, 260
        )
        if ret_val > 0:
            cap = cv2.VideoCapture(buf.value)
            if cap.isOpened():
                return cap

    raise IOError(f"无法打开视频: {video_path}")


class ParkingLotPipeline:
    """停车场视频处理流水线

    整合相机标定、图像预处理和坐标映射三个模块，
    提供从原始视频到标准化输出的完整处理链。
    """

    def __init__(
        self,
        calibration_file: Optional[str] = None,
        homography_file: Optional[str] = None,
        output_dir: str = "./output",
    ):
        """
        Args:
            calibration_file: 已有标定参数文件路径，None则需要重新标定
            homography_file: 已有单应性矩阵文件路径，None则需要重新计算
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.calibrator = CameraCalibrator()
        self.preprocessor = ImagePreprocessor()
        self.mapper = HomographyMapper()

        if calibration_file and os.path.exists(calibration_file):
            self.calibrator.load_params(calibration_file)

        if homography_file and os.path.exists(homography_file):
            self.mapper.load(homography_file)

    def run_calibration(
        self, source: str, from_video: bool = True, **kwargs
    ) -> dict:
        """执行标定流程

        Args:
            source: 视频路径或图像目录
            from_video: True=从视频采集, False=从图像目录采集

        Returns:
            标定参数字典
        """
        if from_video:
            frames = self.calibrator.collect_from_video(source, **kwargs)
        else:
            frames = self.calibrator.collect_from_images(source, **kwargs)

        params = self.calibrator.calibrate(frames)

        calib_path = str(self.output_dir / "calibration_params.yaml")
        self.calibrator.save_params(calib_path)

        return params

    def setup_homography(
        self,
        pixel_points: np.ndarray,
        world_points: np.ndarray,
    ) -> np.ndarray:
        """建立坐标映射关系

        Args:
            pixel_points: 图像像素坐标 (N, 2)
            world_points: 停车场平面坐标 (N, 2)，单位米

        Returns:
            3x3单应性矩阵
        """
        H = self.mapper.compute_homography(pixel_points, world_points)

        homo_path = str(self.output_dir / "homography_matrix.yaml")
        self.mapper.save(homo_path)

        return H

    def process_video(
        self,
        video_path: str,
        save_interval: int = 1,
        max_frames: int = 0,
        show_preview: bool = False,
    ) -> str:
        """处理视频并输出预处理后的图像序列

        Args:
            video_path: 输入视频路径
            save_interval: 每隔N帧保存一张
            max_frames: 最大处理帧数，0=处理全部
            show_preview: 是否显示实时预览

        Returns:
            输出图像目录路径
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        frames_dir = self.output_dir / "preprocessed_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        wait_ms = max(1, int(1000 / fps))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                break

            frame_idx += 1

            # 去畸变
            if self.calibrator.camera_matrix is not None:
                frame = self.calibrator.undistort(frame)

            # 自适应预处理
            processed = self.preprocessor.process(frame)

            # 按间隔保存
            should_save = (frame_idx % save_interval == 0)
            if should_save:
                output_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_path), processed)
                saved_count += 1

            # 预览每帧都更新，保存帧叠加标记
            if show_preview:
                preview = cv2.resize(processed, (960, 540))
                if should_save:
                    cv2.putText(
                        preview, f"[SAVED] frame_{frame_idx:06d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2,
                    )
                if cv2.imshow("Preprocessed", preview) is not None:
                    pass
                if cv2.waitKey(wait_ms) == 27:
                    break

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        print(f"处理完成: 共保存 {saved_count} 帧至 {frames_dir}")
        return str(frames_dir)

    def transform_point(self, pixel_x: float, pixel_y: float) -> tuple:
        """将单个像素坐标转换为停车场平面坐标

        Returns:
            (world_x, world_y) 单位米
        """
        world = self.mapper.pixel_to_world(np.array([pixel_x, pixel_y]))
        return (float(world[0]), float(world[1]))

    def transform_bbox(self, bbox: tuple) -> np.ndarray:
        """将检测框的四个角点映射到停车场平面

        Args:
            bbox: (x1, y1, x2, y2) 检测框像素坐标

        Returns:
            四个角点的平面坐标，shape=(4, 2)
        """
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float64)

        return self.mapper.pixel_to_world(corners)
