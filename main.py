"""
停车场车位识别系统 - 主程序入口

演示完整的标定→预处理→坐标映射流程。
可直接运行或作为模块调用参考。

使用方式：
    python main.py --mode calibrate --source ./calibration_images/
    python main.py --mode process --video ./parking_lot.mp4
    python main.py --mode demo
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from parking_vision import (
    CameraCalibrator,
    ImagePreprocessor,
    HomographyMapper,
    ParkingLotPipeline,
    BackgroundModeler,
    ForegroundSegmenter,
    SegmentationPipeline,
)


def demo_calibration():
    """演示相机标定流程（使用合成棋盘格图像）"""
    print("=" * 60)
    print("  相机标定演示 (合成数据)")
    print("=" * 60)

    calibrator = CameraCalibrator(board_size=(9, 6), square_size=25.0)

    # 合成相机参数用于验证
    image_size = (1920, 1080)
    fx, fy = 1000.0, 1000.0
    cx, cy = image_size[0] / 2, image_size[1] / 2

    true_camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    true_dist_coeffs = np.array([-0.1, 0.05, 0.001, -0.001, 0.0], dtype=np.float64)

    print(f"\n真实相机内参矩阵:\n{true_camera_matrix}")
    print(f"真实畸变系数: {true_dist_coeffs.flatten()}")

    # 生成合成标定图像
    objp = calibrator.objp
    valid_frames = []

    for i in range(20):
        # 随机旋转和平移
        rvec = np.random.uniform(-0.5, 0.5, (3, 1))
        tvec = np.array([[0], [0], [300 + i * 10]], dtype=np.float64)

        # 投影到图像平面
        img_points, _ = cv2.projectPoints(
            objp, rvec, tvec, true_camera_matrix, true_dist_coeffs
        )
        img_points = img_points.reshape(-1, 1, 2).astype(np.float32)

        # 检查所有点是否在图像内
        pts = img_points.reshape(-1, 2)
        if (pts[:, 0].min() < 50 or pts[:, 0].max() > image_size[0] - 50 or
                pts[:, 1].min() < 50 or pts[:, 1].max() > image_size[1] - 50):
            continue

        # 生成合成灰度图
        frame = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 200
        valid_frames.append((frame, img_points))

    calibrator.image_size = image_size
    print(f"\n生成了 {len(valid_frames)} 张合成标定图像")

    # 执行标定
    params = calibrator.calibrate(valid_frames)

    print(f"\n估计的相机内参矩阵:\n{params['camera_matrix']}")
    print(f"估计的畸变系数: {params['dist_coeffs'].flatten()}")

    # 保存标定结果
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrator.save_params(str(output_dir / "calibration_params.yaml"))

    return params


def demo_preprocessing():
    """演示图像预处理流程"""
    print("\n" + "=" * 60)
    print("  图像预处理演示")
    print("=" * 60)

    preprocessor = ImagePreprocessor(clip_limit=3.0, tile_size=(8, 8))

    # 模拟夜间低照度图像
    print("\n[场景1] 夜间低照度")
    night_image = np.random.randint(5, 40, (1080, 1920, 3), dtype=np.uint8)
    analysis = preprocessor.analyze_illumination(night_image)
    print(f"  光照分析: 平均亮度={analysis['mean_brightness']:.1f}, "
          f"场景类型={analysis['scene_type']}")
    night_enhanced = preprocessor.process(night_image)
    print(f"  增强后亮度: {np.mean(cv2.cvtColor(night_enhanced, cv2.COLOR_BGR2GRAY)):.1f}")

    # 模拟白天强反光图像
    print("\n[场景2] 白天强反光")
    glare_image = np.random.randint(150, 255, (1080, 1920, 3), dtype=np.uint8)
    # 添加局部过曝区域
    glare_image[200:400, 800:1200] = 255
    analysis = preprocessor.analyze_illumination(glare_image)
    print(f"  光照分析: 平均亮度={analysis['mean_brightness']:.1f}, "
          f"过曝比例={analysis['overexposed_ratio']:.2%}, 场景类型={analysis['scene_type']}")
    glare_enhanced = preprocessor.process(glare_image)
    print(f"  增强后亮度: {np.mean(cv2.cvtColor(glare_enhanced, cv2.COLOR_BGR2GRAY)):.1f}")

    # 保存预处理结果
    output_dir = Path("./output/preprocessed_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "night_enhanced.jpg"), night_enhanced)
    cv2.imwrite(str(output_dir / "glare_enhanced.jpg"), glare_enhanced)
    print(f"\n预处理结果已保存至: {output_dir}")


def demo_homography():
    """演示坐标映射流程"""
    print("\n" + "=" * 60)
    print("  坐标映射演示")
    print("=" * 60)

    mapper = HomographyMapper()

    # 模拟停车场地面上的已知标志点
    # 像素坐标（从顶视相机图像中获取）
    pixel_points = np.array([
        [400, 300],   # 左上车位角点
        [1500, 300],  # 右上车位角点
        [1600, 800],  # 右下车位角点
        [300, 800],   # 左下车位角点
        [950, 250],   # 中上标线
        [1000, 850],  # 中下标线
    ], dtype=np.float64)

    # 对应的停车场平面坐标（米），以停车场某角为原点
    world_points = np.array([
        [2.5, 1.0],   # 对应左上
        [12.5, 1.0],  # 对应右上
        [13.0, 6.0],  # 对应右下
        [2.0, 6.0],   # 对应左下
        [7.5, 0.5],   # 对应中上
        [7.8, 6.5],   # 对应中下
    ], dtype=np.float64)

    # 计算单应性
    H = mapper.compute_homography(pixel_points, world_points)
    print(f"\n单应性矩阵 H (像素->世界):\n{H}")

    # 验证映射
    print("\n验证坐标转换:")
    test_pixel = np.array([960, 540])  # 图像中心
    test_world = mapper.pixel_to_world(test_pixel)
    back_pixel = mapper.world_to_pixel(test_world)
    print(f"  像素 {test_pixel} -> 世界 {test_world}")
    print(f"  反向验证: 世界 {test_world} -> 像素 {back_pixel}")
    print(f"  往返误差: {np.linalg.norm(test_pixel - back_pixel):.4f} 像素")

    # 批量转换
    print("\n批量坐标转换:")
    parking_spots_pixel = np.array([
        [500, 450],
        [750, 450],
        [1000, 450],
        [1250, 450],
    ], dtype=np.float64)

    parking_spots_world = mapper.pixel_to_world(parking_spots_pixel)
    for i, (px, wd) in enumerate(zip(parking_spots_pixel, parking_spots_world)):
        print(f"  车位{i+1}: 像素({px[0]:.0f}, {px[1]:.0f}) -> "
              f"平面({wd[0]:.2f}m, {wd[1]:.2f}m)")

    # 保存
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapper.save(str(output_dir / "homography_matrix.yaml"))

    return H


def demo_full_pipeline():
    """演示完整流水线"""
    print("\n" + "=" * 60)
    print("  完整流水线演示")
    print("=" * 60)

    pipeline = ParkingLotPipeline(output_dir="./output")

    # 1. 标定（使用合成数据演示）
    print("\n[步骤1] 相机标定")
    demo_calibration()

    # 2. 预处理
    print("\n[步骤2] 图像预处理")
    demo_preprocessing()

    # 3. 坐标映射
    print("\n[步骤3] 建立坐标映射")
    demo_homography()

    # 4. 背景建模与前景分割
    print("\n[步骤4] 背景建模与前景分割")
    demo_segmentation()

    print("\n" + "=" * 60)
    print("  流水线演示完成！")
    print("  输出文件:")
    print("    - output/calibration_params.yaml       (标定参数)")
    print("    - output/homography_matrix.yaml        (坐标映射)")
    print("    - output/preprocessed_frames/          (预处理图像)")
    print("    - output/segmentation_demo/            (分割结果)")
    print("      - background_model.png              (背景模型)")
    print("      - last_frame_detection.jpg          (检测可视化)")
    print("      - last_mask.png                     (前景掩码)")
    print("=" * 60)


def run_calibration(args):
    """运行标定模式"""
    pipeline = ParkingLotPipeline(output_dir=args.output)

    source = str(args.source)
    is_video = source.isdigit() or source.lower().endswith((".mp4", ".avi", ".mkv"))
    pipeline.run_calibration(source, from_video=is_video)


def run_process(args):
    """运行视频处理模式"""
    pipeline = ParkingLotPipeline(
        calibration_file=args.calibration,
        homography_file=args.homography,
        output_dir=args.output,
    )
    pipeline.process_video(
        args.video,
        save_interval=args.interval,
        show_preview=args.preview,
    )


def demo_segmentation():
    """演示背景建模与前景分割流程（使用合成视频）"""
    print("\n" + "=" * 60)
    print("  背景建模与前景分割演示")
    print("=" * 60)

    # 生成合成停车场视频序列
    h, w = 480, 640
    num_frames = 150
    warmup = 80

    print(f"\n生成合成视频序列: {w}x{h}, {num_frames}帧")

    # 创建静态背景（灰色地面 + 车位线）
    background = np.ones((h, w, 3), dtype=np.uint8) * 128
    # 车位标线
    for x in range(100, 600, 120):
        cv2.line(background, (x, 100), (x, 380), (255, 255, 255), 2)
    cv2.line(background, (100, 100), (580, 100), (255, 255, 255), 2)
    cv2.line(background, (100, 380), (580, 380), (255, 255, 255), 2)

    # 初始化模块
    bg_modeler = BackgroundModeler(history=100, var_threshold=16.0)
    fg_segmenter = ForegroundSegmenter(min_area=300, morph_kernel_size=5)

    print(f"预热阶段: {warmup}帧（仅更新背景模型）")

    all_candidates = []

    for i in range(num_frames):
        frame = background.copy()

        # 添加噪声模拟真实场景
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 第80帧后添加移动"车辆"（矩形色块）
        if i >= warmup:
            t = i - warmup
            # 车辆1：从左向右移动
            cx1 = 120 + t * 4
            if cx1 < 550:
                cv2.rectangle(frame, (cx1, 180), (cx1 + 80, 240), (50, 50, 200), -1)
            # 车辆2：静止停放
            cv2.rectangle(frame, (350, 250), (440, 320), (200, 50, 50), -1)

        # 更新背景
        raw_mask = bg_modeler.update(frame)

        # 预热阶段跳过分割
        if i < warmup:
            continue

        # 前景分割
        clean_mask, candidates = fg_segmenter.process(raw_mask)
        all_candidates.append(candidates)

    # 输出结果统计
    bg = bg_modeler.get_background()
    total_detections = sum(len(c) for c in all_candidates)

    print(f"\n分割结果:")
    print(f"  有效处理帧: {num_frames - warmup}")
    print(f"  总检测候选数: {total_detections}")
    print(f"  背景模型尺寸: {bg.shape}")

    # 保存演示结果
    output_dir = Path("./output/segmentation_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    bg_modeler.save_model(str(output_dir / "background_model.png"))

    # 生成最后一帧的可视化对比
    last_frame = background.copy()
    cv2.rectangle(last_frame, (350, 250), (440, 320), (200, 50, 50), -1)
    noise = np.random.randint(-10, 10, last_frame.shape, dtype=np.int16)
    last_frame = np.clip(last_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    raw_mask = bg_modeler.update(last_frame)
    clean_mask, candidates = fg_segmenter.process(raw_mask)

    # 可视化
    vis_frame = last_frame.copy()
    for cand in candidates:
        x, y, bw, bh = cand["bbox"]
        cv2.rectangle(vis_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            vis_frame, f"area={cand['area']}", (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
        )

    cv2.imwrite(str(output_dir / "last_frame_detection.jpg"), vis_frame)
    cv2.imwrite(str(output_dir / "last_mask.png"), clean_mask)

    print(f"\n演示结果已保存至: {output_dir}")
    print(f"  - background_model.png    (学习到的背景)")
    print(f"  - last_frame_detection.jpg (最后一帧检测可视化)")
    print(f"  - last_mask.png           (前景掩码)")


def run_segmentation(args):
    """运行前景分割模式"""
    bg_modeler = BackgroundModeler(
        history=500,
        var_threshold=16.0,
        detect_shadows=True,
    )
    fg_segmenter = ForegroundSegmenter(
        min_area=args.min_area,
        morph_kernel_size=5,
    )

    pipeline = SegmentationPipeline(
        video_path=args.video,
        output_dir=args.output,
        bg_modeler=bg_modeler,
        fg_segmenter=fg_segmenter,
    )

    pipeline.run(
        max_frames=args.max_frames,
        warmup_frames=args.warmup,
        save_interval=args.save_interval,
        show_preview=args.preview,
    )


def main():
    parser = argparse.ArgumentParser(
        description="停车场车位识别系统 - 相机标定与图像预处理"
    )
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # 标定模式
    calib_parser = subparsers.add_parser("calibrate", help="相机标定")
    calib_parser.add_argument("--source", required=True, help="视频路径或标定图像目录")
    calib_parser.add_argument("--output", default="./output", help="输出目录")

    # 处理模式
    proc_parser = subparsers.add_parser("process", help="视频处理")
    proc_parser.add_argument("--video", required=True, help="输入视频路径")
    proc_parser.add_argument("--calibration", help="标定参数文件")
    proc_parser.add_argument("--homography", help="单应性矩阵文件")
    proc_parser.add_argument("--output", default="./output", help="输出目录")
    proc_parser.add_argument("--interval", type=int, default=1, help="保存间隔帧数")
    proc_parser.add_argument("--preview", action="store_true", help="显示实时预览")

    # 前景分割模式
    seg_parser = subparsers.add_parser("segment", help="背景建模与前景分割")
    seg_parser.add_argument("--video", required=True, help="输入视频路径")
    seg_parser.add_argument("--output", default="./output", help="输出目录")
    seg_parser.add_argument("--warmup", type=int, default=200, help="预热帧数")
    seg_parser.add_argument("--min-area", type=int, default=500, help="最小候选面积")
    seg_parser.add_argument("--max-frames", type=int, default=0, help="最大处理帧数，0=全部")
    seg_parser.add_argument("--save-interval", type=int, default=5, help="保存间隔帧数")
    seg_parser.add_argument("--preview", action="store_true", help="显示实时预览")

    # 演示模式
    subparsers.add_parser("demo", help="使用合成数据演示完整流程")

    args = parser.parse_args()

    if args.mode == "calibrate":
        run_calibration(args)
    elif args.mode == "process":
        run_process(args)
    elif args.mode == "segment":
        run_segmentation(args)
    elif args.mode == "demo":
        demo_full_pipeline()
    else:
        parser.print_help()
        print("\n提示: 运行 'python main.py demo' 可快速查看完整流程演示")


if __name__ == "__main__":
    main()
