"""
分割流水线 - 整合背景建模、前景分割与可视化输出

协调BackgroundModeler和ForegroundSegmenter，实现从视频输入到
背景模型、前景掩码序列、车辆边界框的完整处理流程。
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .background_modeling import BackgroundModeler
from .foreground_segmentation import ForegroundSegmenter
from .preprocessing import ImagePreprocessor


def _open_video(video_path: str) -> cv2.VideoCapture:
    """打开视频文件，兼容含中文/特殊字符的路径（Windows）"""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap

    # Windows下OpenCV不支持非ASCII路径，用numpy中转读取不适用于视频
    # 改用短路径名绕过编码限制
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


class SegmentationPipeline:
    """背景建模与前景分割完整流水线"""

    def __init__(
        self,
        video_path: str,
        output_dir: str = "./output",
        preprocessor: Optional[ImagePreprocessor] = None,
        bg_modeler: Optional[BackgroundModeler] = None,
        fg_segmenter: Optional[ForegroundSegmenter] = None,
    ):
        """
        Args:
            video_path: 输入视频路径
            output_dir: 输出根目录
            preprocessor: 图像预处理器（兼容第一轮模块），None则新建
            bg_modeler: 背景建模器，None则使用默认参数
            fg_segmenter: 前景分割器，None则使用默认参数
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)

        self.preprocessor = preprocessor or ImagePreprocessor()
        self.bg_modeler = bg_modeler or BackgroundModeler()
        self.fg_segmenter = fg_segmenter or ForegroundSegmenter()

        # 输出子目录
        self.masks_dir = self.output_dir / "foreground_masks"
        self.vis_dir = self.output_dir / "visualization"

        self._results: List[dict] = []

    def run(
        self,
        max_frames: int = 0,
        warmup_frames: int = 200,
        save_interval: int = 5,
        show_preview: bool = False,
    ) -> dict:
        """执行完整分割流水线

        Args:
            max_frames: 最大处理帧数，0=全部
            warmup_frames: 预热帧数（仅更新背景，不输出分割结果）
            save_interval: 每N帧保存一次掩码和可视化
            show_preview: 是否显示实时预览窗口

        Returns:
            处理统计信息
        """
        cap = _open_video(self.video_path)

        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wait_ms = max(1, int(1000 / fps))

        frame_idx = 0
        saved_count = 0
        self._results = []

        print(f"[分割流水线] 视频: {self.video_path}")
        print(f"  总帧数: {total_frames}, FPS: {fps:.1f}")
        print(f"  预热帧数: {warmup_frames}, 保存间隔: {save_interval}")
        print(f"  最小面积阈值: {self.fg_segmenter.min_area}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                break

            frame_idx += 1

            # 对原始帧做轻量归一化（高斯模糊去传感器噪声），
            # 不使用自适应预处理以避免帧间亮度跳变干扰背景模型
            stable_frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # 更新背景模型（用稳定帧）
            raw_mask = self.bg_modeler.update(stable_frame)

            # 预热阶段只更新背景
            if frame_idx <= warmup_frames:
                if frame_idx % 50 == 0:
                    print(f"  预热进度: {frame_idx}/{warmup_frames}")
                continue

            # 前景分割
            clean_mask, candidates = self.fg_segmenter.process(raw_mask)

            # 可视化用预处理后的帧（更清晰）
            display_frame = self.preprocessor.process(frame)

            # 记录结果
            frame_result = {
                "frame_idx": frame_idx,
                "num_candidates": len(candidates),
                "bboxes": [c["bbox"] for c in candidates],
                "areas": [c["area"] for c in candidates],
                "centroids": [c["centroid"] for c in candidates],
            }
            self._results.append(frame_result)

            # 按间隔保存
            should_save = ((frame_idx - warmup_frames) % save_interval == 0)
            if should_save:
                mask_path = self.masks_dir / f"mask_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), clean_mask)

                vis = self.visualize_comparison(display_frame, clean_mask, candidates)
                vis_path = self.vis_dir / f"vis_{frame_idx:06d}.jpg"
                cv2.imwrite(str(vis_path), vis)
                saved_count += 1

            # 实时预览
            if show_preview:
                vis = self.visualize_comparison(display_frame, clean_mask, candidates)
                preview = cv2.resize(vis, (1280, 720))
                cv2.imshow("Segmentation Preview", preview)
                if cv2.waitKey(wait_ms) == 27:
                    break

            if frame_idx % 100 == 0:
                print(f"  处理帧: {frame_idx}, 当前检测到 {len(candidates)} 个候选目标")

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        # 保存背景模型和汇总结果
        self.save_results()

        stats = {
            "total_frames_processed": frame_idx,
            "warmup_frames": warmup_frames,
            "active_frames": frame_idx - warmup_frames,
            "saved_outputs": saved_count,
            "total_detections": sum(r["num_candidates"] for r in self._results),
        }
        print(f"\n[分割流水线] 处理完成:")
        print(f"  总帧数: {stats['total_frames_processed']}")
        print(f"  有效帧: {stats['active_frames']}")
        print(f"  保存输出: {stats['saved_outputs']} 组")
        print(f"  总检测数: {stats['total_detections']}")

        return stats

    def save_results(self) -> None:
        """保存背景模型和分割结果汇总"""
        # 背景模型
        bg_path = str(self.output_dir / "background_model.png")
        self.bg_modeler.save_model(bg_path)
        print(f"  背景模型已保存: {bg_path}")

        # JSON汇总（不含contour数据，仅bbox和面积）
        json_path = self.output_dir / "segmentation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video": self.video_path,
                    "bg_model_path": bg_path,
                    "masks_dir": str(self.masks_dir),
                    "frames": self._results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  分割结果已保存: {json_path}")

    def visualize_comparison(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        candidates: List[dict],
    ) -> np.ndarray:
        """生成2x2可视化对比图

        布局：
        - 左上：原始帧（预处理后）
        - 右上：背景模型
        - 左下：前景掩码（彩色化）
        - 右下：原始帧 + 候选车辆边界框

        Returns:
            拼接后的可视化图像
        """
        h, w = frame.shape[:2]

        # 左上：原始帧
        top_left = frame.copy()

        # 右上：背景模型
        try:
            bg = self.bg_modeler.get_background()
            top_right = cv2.resize(bg, (w, h)) if bg.shape[:2] != (h, w) else bg
        except RuntimeError:
            top_right = np.zeros_like(frame)

        # 左下：前景掩码彩色化
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[mask == 255] = [0, 255, 0]  # 前景绿色
        bottom_left = mask_color

        # 右下：边界框叠加
        bottom_right = frame.copy()
        for cand in candidates:
            x, y, bw, bh = cand["bbox"]
            cv2.rectangle(bottom_right, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cx, cy = cand["centroid"]
            cv2.circle(bottom_right, (cx, cy), 4, (255, 0, 0), -1)
            label = f"{cand['area']}"
            cv2.putText(
                bottom_right, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )

        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(top_left, "Original", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(top_right, "Background", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(bottom_left, "FG Mask", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(
            bottom_right, f"Candidates: {len(candidates)}", (10, 30),
            font, 1.0, (0, 0, 255), 2,
        )

        # 拼接
        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        combined = np.vstack([top_row, bottom_row])

        return combined
