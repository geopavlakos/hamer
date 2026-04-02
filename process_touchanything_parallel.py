"""
优化版 TouchAnything 数据集处理脚本
- 批量推理优化（一次处理多帧）
- 多进程并行处理
- 视频可视化（绘制手部骨架）
- 生成 hamer_hands.json
"""
from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
import json
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

from vitpose_model import ViTPoseModel


# ── 手部骨架连接关系 ──
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]


def project_points(points_3d, focal_length, img_w, img_h):
    """3D 点投影到 2D 图像坐标"""
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    eps = 1e-9

    u = focal_length * (x / (z + eps)) + img_w / 2.0
    v = focal_length * (y / (z + eps)) + img_h / 2.0
    return np.stack([u, v], axis=1)


def detect_outliers_and_smooth(results, max_displacement=0.5):
    """
    检测异常跳变并平滑处理
    
    策略：
    1. 计算相邻帧之间的位移
    2. 如果位移超过阈值，标记为异常
    3. 异常帧使用前一帧的结果替代
    
    Args:
        results: 帧结果列表
        max_displacement: 最大允许位移（米），默认0.5米
    """
    if not results or len(results) < 2:
        return results
    
    n_frames = len(results)
    
    # 分别处理左手和右手
    for hand_key in ['left_pos', 'right_pos']:
        outlier_count = 0
        
        for i in range(1, n_frames):
            curr_pos = results[i][hand_key]
            prev_pos = results[i-1][hand_key]
            
            # 如果当前帧或前一帧没有检测结果，跳过
            if not curr_pos or not prev_pos:
                continue
            
            # 计算手腕位置的位移（第0个关节点）
            curr_wrist = np.array(curr_pos[0])
            prev_wrist = np.array(prev_pos[0])
            displacement = np.linalg.norm(curr_wrist - prev_wrist)
            
            # 如果位移过大，标记为异常并使用前一帧结果
            if displacement > max_displacement:
                results[i][hand_key] = prev_pos
                outlier_count += 1
        
        if outlier_count > 0:
            print(f"[INFO] {hand_key}: 检测到 {outlier_count} 个异常跳变帧，已平滑处理")
    
    return results


def interpolate_missing_hands(results):
    """
    插值和填充缺失的手部姿态
    
    策略：
    1. 线性插值：如果前后都有检测结果，使用线性插值
    2. 前向填充：如果只有前面有检测结果，使用前一帧的结果
    3. 后向填充：如果只有后面有检测结果，使用后一帧的结果
    """
    if not results:
        return results
    
    n_frames = len(results)
    
    # 分别处理左手和右手
    for hand_key in ['left_pos', 'right_pos']:
        # 找到所有有效帧的索引
        valid_indices = [i for i, r in enumerate(results) if r[hand_key]]
        
        if not valid_indices:
            continue
        
        # 统计信息
        missing_count = n_frames - len(valid_indices)
        if missing_count > 0:
            print(f"[INFO] {hand_key}: 检测到 {len(valid_indices)}/{n_frames} 帧，缺失 {missing_count} 帧")
        
        # 处理缺失帧
        for i in range(n_frames):
            if results[i][hand_key]:
                continue
            
            # 找到最近的前后有效帧
            prev_idx = None
            next_idx = None
            
            for vi in valid_indices:
                if vi < i:
                    prev_idx = vi
                elif vi > i:
                    next_idx = vi
                    break
            
            # 插值或填充
            if prev_idx is not None and next_idx is not None:
                # 线性插值
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                prev_pos = np.array(results[prev_idx][hand_key])
                next_pos = np.array(results[next_idx][hand_key])
                interpolated_pos = (1 - alpha) * prev_pos + alpha * next_pos
                results[i][hand_key] = interpolated_pos.tolist()
                
            elif prev_idx is not None:
                # 前向填充
                results[i][hand_key] = results[prev_idx][hand_key]
                
            elif next_idx is not None:
                # 后向填充
                results[i][hand_key] = results[next_idx][hand_key]
    
    return results


def load_jsonl_results(json_path):
    """读取按行存储的 JSON 结果"""
    results = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def save_jsonl_results(json_path, results):
    """保存按行存储的 JSON 结果"""
    with open(json_path, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def fill_missing_hands_with_nearest(results):
    """
    用时间上最近的有效帧填补缺失手部结果。

    处理策略分两步：
    1. 如果某一整帧 left_pos 和 right_pos 都为空，则直接复制最近的有效帧结果
    2. 如果只是单手为空，则再按手分别用最近的有效帧补齐

    如果前后两侧都有有效帧，则选择帧距离更近的那一侧；
    距离相同时优先使用前一帧。
    """
    if not results:
        return results, {}

    n_frames = len(results)
    fill_stats = {
        'frame_level': 0,
        'left_pos': 0,
        'right_pos': 0,
    }

    def has_hand(frame_result, hand_key):
        return bool(frame_result.get(hand_key))

    def has_any_hand(frame_result):
        return has_hand(frame_result, 'left_pos') or has_hand(frame_result, 'right_pos')

    def choose_nearest_index(prev_idx, next_idx, current_idx):
        if prev_idx is None and next_idx is None:
            return None
        if prev_idx is None:
            return next_idx
        if next_idx is None:
            return prev_idx
        return prev_idx if (current_idx - prev_idx) <= (next_idx - current_idx) else next_idx

    frame_prev_valid = [None] * n_frames
    frame_next_valid = [None] * n_frames

    last_valid = None
    for i in range(n_frames):
        if has_any_hand(results[i]):
            last_valid = i
        frame_prev_valid[i] = last_valid

    last_valid = None
    for i in range(n_frames - 1, -1, -1):
        if has_any_hand(results[i]):
            last_valid = i
        frame_next_valid[i] = last_valid

    for i in range(n_frames):
        if has_any_hand(results[i]):
            continue

        source_idx = choose_nearest_index(frame_prev_valid[i], frame_next_valid[i], i)
        if source_idx is None:
            continue

        results[i]['left_pos'] = list(results[source_idx].get('left_pos', []))
        results[i]['right_pos'] = list(results[source_idx].get('right_pos', []))
        fill_stats['frame_level'] += 1

    for hand_key in ['left_pos', 'right_pos']:
        prev_valid = [None] * n_frames
        next_valid = [None] * n_frames

        last_valid = None
        for i in range(n_frames):
            if has_hand(results[i], hand_key):
                last_valid = i
            prev_valid[i] = last_valid

        last_valid = None
        for i in range(n_frames - 1, -1, -1):
            if has_hand(results[i], hand_key):
                last_valid = i
            next_valid[i] = last_valid

        for i in range(n_frames):
            if has_hand(results[i], hand_key):
                continue

            source_idx = choose_nearest_index(prev_valid[i], next_valid[i], i)
            if source_idx is None:
                continue

            results[i][hand_key] = list(results[source_idx][hand_key])
            fill_stats[hand_key] += 1

    return results, fill_stats


def repair_existing_hamer_json(json_path):
    """
    检查现有 hamer_hands.json 是否存在缺失手部结果，若有则就地修复。

    Returns:
        bool: True 表示已成功检查/修复；False 表示读取失败，需要重新处理视频。
    """
    try:
        results = load_jsonl_results(json_path)
    except Exception as exc:
        print(f"[WARN] 读取现有 JSON 失败，准备重新处理视频: {json_path}, error={exc}")
        return False

    results, fill_stats = fill_missing_hands_with_nearest(results)
    total_filled = sum(fill_stats.values())

    if total_filled == 0:
        print(f"[INFO] 现有 JSON 无缺失手部帧，保留原结果: {json_path}")
        return True

    save_jsonl_results(json_path, results)
    print(
        f"[INFO] 已修复现有 JSON: {json_path} "
        f"(整帧替代 {fill_stats.get('frame_level', 0)} 帧, "
        f"left_pos 补 {fill_stats.get('left_pos', 0)} 帧, "
        f"right_pos 补 {fill_stats.get('right_pos', 0)} 帧)"
    )
    return True


def draw_one_hand(canvas, joints_2d, is_right, edges=HAND_EDGES):
    """
    绘制单只手的骨架
    右手：蓝线 + 绿点
    左手：红线 + 黄点
    """
    if is_right:
        line_color = (255, 0, 0)    # 蓝
        point_color = (0, 255, 0)   # 绿
    else:
        line_color = (0, 0, 255)    # 红
        point_color = (0, 255, 255) # 黄

    for i, j in edges:
        pt1 = tuple(np.round(joints_2d[i]).astype(int))
        pt2 = tuple(np.round(joints_2d[j]).astype(int))
        cv2.line(canvas, pt1, pt2, line_color, 2)

    for pt in joints_2d:
        x, y = np.round(pt).astype(int)
        cv2.circle(canvas, (x, y), 3, point_color, -1)

    return canvas


def detect_both_hands_boxes(img_cv2, detector, cpm):
    """检测图像中的左右手边界框"""
    det_out = detector(img_cv2)
    img_rgb = img_cv2[:, :, ::-1].copy()

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)

    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    if len(pred_bboxes) == 0:
        return None, None

    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right_list = []

    for vitposes in vitposes_out:
        keypoints = vitposes['keypoints']

        # 左手 21 个点
        left_hand_keyp = keypoints[-42:-21]
        left_valid = left_hand_keyp[:, 2] > 0.5
        if np.sum(left_valid) > 3:
            left_bbox = [
                left_hand_keyp[left_valid, 0].min(),
                left_hand_keyp[left_valid, 1].min(),
                left_hand_keyp[left_valid, 0].max(),
                left_hand_keyp[left_valid, 1].max()
            ]
            bboxes.append(left_bbox)
            is_right_list.append(0)

        # 右手 21 个点
        right_hand_keyp = keypoints[-21:]
        right_valid = right_hand_keyp[:, 2] > 0.5
        if np.sum(right_valid) > 3:
            right_bbox = [
                right_hand_keyp[right_valid, 0].min(),
                right_hand_keyp[right_valid, 1].min(),
                right_hand_keyp[right_valid, 0].max(),
                right_hand_keyp[right_valid, 1].max()
            ]
            bboxes.append(right_bbox)
            is_right_list.append(1)

    if len(bboxes) == 0:
        return None, None

    return np.stack(bboxes), np.array(is_right_list, dtype=np.int64)


def process_frames_batch(
    frames,
    detector,
    cpm,
    model,
    model_cfg,
    device,
    batch_size=8,
    rescale_factor=2.0
):
    """
    批量处理多帧，返回每帧的结果
    
    Returns:
        results: List[dict] - 每帧的结果
            {
                'left_pos': [[x,y,z], ...] or None,
                'right_pos': [[x,y,z], ...] or None,
                'vis_frame': np.ndarray - 可视化帧,
                'focal_length': float - 焦距
            }
    """
    # 批量检测所有帧的手部边界框
    all_boxes = []
    all_rights = []
    for frame in frames:
        boxes, right = detect_both_hands_boxes(frame, detector, cpm)
        all_boxes.append(boxes)
        all_rights.append(right)
    
    results = []
    
    for frame_idx, (frame, boxes, right) in enumerate(zip(frames, all_boxes, all_rights)):
        img_h, img_w = frame.shape[:2]
        vis_frame = frame.copy()
        left_pos = None
        right_pos = None
        
        focal_length_value = max(img_w, img_h)  # 默认焦距
        
        if boxes is not None:
            dataset = ViTDetDataset(
                model_cfg,
                frame,
                boxes,
                right,
                rescale_factor=rescale_factor
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,  # 并行数据加载
                pin_memory=True  # 加速GPU传输
            )

            for batch in dataloader:
                batch = recursive_to(batch, device)

                # 混合精度推理加速
                with torch.no_grad(), torch.cuda.amp.autocast():
                    out = model(batch)

                pred_joints_3d = out['pred_keypoints_3d']  # (B, 21, 3)
                
                multiplier = (2 * batch['right'] - 1)
                pred_cam = out['pred_cam'].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()

                scaled_focal_length = (
                    model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                )

                pred_cam_t_full = cam_crop_to_full(
                    pred_cam,
                    box_center,
                    box_size,
                    img_size,
                    scaled_focal_length
                ).detach().cpu().numpy()

                bs = batch['img'].shape[0]
                for n in range(bs):
                    joints_3d = pred_joints_3d[n].detach().cpu().numpy()
                    is_right_hand = int(batch['right'][n].detach().cpu().item())
                    
                    # 镜像处理
                    joints_3d[:, 0] = (2 * is_right_hand - 1) * joints_3d[:, 0]
                    
                    # 转换到相机坐标系
                    joints_3d_cam = joints_3d + pred_cam_t_full[n][None, :]
                    
                    # 投影到 2D 用于可视化
                    if scaled_focal_length.ndim == 0:
                        current_focal_length = float(scaled_focal_length.detach().cpu().item())
                    else:
                        current_focal_length = float(scaled_focal_length[n].detach().cpu().item())
                    
                    # 更新焦距（使用实际计算的焦距）
                    focal_length_value = current_focal_length
                    
                    joints_2d = project_points(
                        joints_3d_cam,
                        focal_length=current_focal_length,
                        img_w=img_w,
                        img_h=img_h
                    )
                    
                    # 绘制骨架
                    vis_frame = draw_one_hand(vis_frame, joints_2d, is_right_hand)

                    if is_right_hand:
                        right_pos = joints_3d_cam.tolist()
                    else:
                        left_pos = joints_3d_cam.tolist()
        
        results.append({
            'left_pos': left_pos if left_pos else [],
            'right_pos': right_pos if right_pos else [],
            'vis_frame': vis_frame,
            'focal_length': focal_length_value
        })
    
    return results


def process_video_sequence(
    video_path,
    output_json_path,
    output_video_path,
    detector,
    cpm,
    model,
    model_cfg,
    device,
    batch_size=8,
    frame_batch_size=32,
    rescale_factor=2.0,
    max_frames=None,
    max_displacement=0.5,
    resize_scale=0.5  # 新增: 图像缩放比例,0.5=缩小一半,加速2-3倍
):
    """
    处理视频序列，生成 hamer_hands.json 和可视化视频
    
    Args:
        batch_size: HaMeR 模型推理的 batch size
        frame_batch_size: 一次读取并处理的帧数
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    print(f"[INFO] 处理视频: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, 总帧数: {total_frames}, 分辨率: {width}x{height}")

    # 稍后创建视频写入器（在后处理之后）
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    # 优先尝试 H.264；若当前 OpenCV/FFmpeg 不支持，再回退到更通用的 mp4v
    preferred_codecs = ['avc1', 'mp4v']

    results = []
    frame_index = 0
    
    import time
    time_stats = {'read': 0, 'inference': 0, 'save': 0}
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            # 批量读取帧
            t0 = time.time()
            frames = []
            frame_indices = []
            
            for _ in range(frame_batch_size):
                if max_frames is not None and frame_index >= max_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 可选: 跳帧处理 (frame_skip=2 表示每2帧处理1帧,速度翻倍)
                # frame_skip = 1  # 不跳帧
                # if frame_index % frame_skip != 0:
                #     frame_index += 1
                #     continue
                    
                frames.append(frame)
                frame_indices.append(frame_index)
                frame_index += 1
            
            if len(frames) == 0:
                break
            
            time_stats['read'] += time.time() - t0
            
            # 批量处理
            t1 = time.time()
            batch_results = process_frames_batch(
                frames,
                detector,
                cpm,
                model,
                model_cfg,
                device,
                batch_size=batch_size,
                rescale_factor=rescale_factor
            )
            time_stats['inference'] += time.time() - t1
            
            # 保存结果和可视化帧
            for idx, (frame_idx, result) in enumerate(zip(frame_indices, batch_results)):
                timestamp = frame_idx / fps if fps > 0 else frame_idx * 0.033
                
                frame_data = {
                    "ts": timestamp,
                    "frame_index": frame_idx,
                    "left_pos": result['left_pos'],
                    "right_pos": result['right_pos'],
                    "aligned_to_vive": False,
                    "vis_frame": result['vis_frame']  # 保存可视化帧
                }
                
                results.append(frame_data)
            
            pbar.update(len(frames))

    cap.release()

    # 后处理：用最近的有效帧填补空手部结果
    results, fill_stats = fill_missing_hands_with_nearest(results)
    total_filled = sum(fill_stats.values())
    if total_filled > 0:
        print(
            "\n[INFO] 新生成结果已补全缺失手部帧: "
            f"整帧替代 {fill_stats.get('frame_level', 0)} 帧, "
            f"left_pos 补 {fill_stats.get('left_pos', 0)} 帧, "
            f"right_pos 补 {fill_stats.get('right_pos', 0)} 帧"
        )

    # 写入可视化视频（使用第一次生成的vis_frame）
    print("\n[INFO] 写入可视化视频...")
    t_write_start = time.time()
    video_writer = None
    selected_codec = None
    for codec in preferred_codecs:
        candidate = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height)
        )
        if candidate.isOpened():
            video_writer = candidate
            selected_codec = codec
            break
        candidate.release()

    if video_writer is None:
        raise RuntimeError(
            f"无法创建视频写入器，尝试的编码器: {preferred_codecs}"
        )
    print(f"[INFO] 可视化视频编码器: {selected_codec}")
    
    for result in tqdm(results, desc="Writing visualization"):
        # 直接使用第一次生成的vis_frame
        video_writer.write(result['vis_frame'])
        
        # 移除vis_frame以节省内存
        del result['vis_frame']
    
    video_writer.release()
    time_stats['save'] = time.time() - t_write_start
    
    # 输出性能统计
    total_time = sum(time_stats.values())
    print(f"\n[性能统计]")
    print(f"  视频读取: {time_stats['read']:.2f}s ({time_stats['read']/total_time*100:.1f}%)")
    print(f"  模型推理: {time_stats['inference']:.2f}s ({time_stats['inference']/total_time*100:.1f}%)")
    print(f"  视频写入: {time_stats['save']:.2f}s ({time_stats['save']/total_time*100:.1f}%)")
    print(f"  总耗时: {total_time:.2f}s")

    # 保存 JSON 结果
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl_results(output_json_path, results)

    print(f"[INFO] JSON 已保存到: {output_json_path}")
    print(f"[INFO] 视频已保存到: {output_video_path}")
    print(f"[INFO] 处理了 {len(results)} 帧")
    return True


def process_single_sequence_wrapper(seq_dir, gpu_id, args):
    """包装函数，用于多进程 starmap"""
    return process_single_sequence(seq_dir, args, gpu_id=gpu_id)


def process_single_sequence(seq_dir, args, gpu_id=0):
    """单个序列的处理函数（用于多进程）"""
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 性能优化: 启用 PyTorch 后端优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 加载模型
    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # torch.compile 和 FP16 已禁用 (首次运行会很慢)
    # 如需启用,取消下面的注释:
    # model = torch.compile(model, mode='reduce-overhead')
    # model = model.half()
    
    # 加载检测器
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        local_detectron2_ckpt = Path(CACHE_DIR_HAMER) / 'detectron2_models' / 'model_final_f05665.pkl'
        if local_detectron2_ckpt.exists():
            detectron2_cfg.train.init_checkpoint = str(local_detectron2_ckpt)
            print(f"[INFO] 使用本地 Detectron2 权重: {local_detectron2_ckpt}")
        else:
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5  # 提高阈值减少误检
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py',
            trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    # 加载关键点检测器
    cpm = ViTPoseModel(device)
    
    chest_video = seq_dir / "chest.mp4"
    output_json = seq_dir / "hamer_hands.json"
    output_video = seq_dir / "hamer_visualization.mp4"
    
    if output_json.exists() and not args.overwrite:
        print(f"[INFO] 已存在 hamer_hands.json，检查缺失手部帧: {seq_dir}")
        if repair_existing_hamer_json(output_json):
            return True
    
    success = process_video_sequence(
        chest_video,
        output_json,
        output_video,
        detector,
        cpm,
        model,
        model_cfg,
        device,
        batch_size=args.batch_size,
        frame_batch_size=args.frame_batch_size,
        rescale_factor=args.rescale_factor,
        max_frames=args.max_frames,
        max_displacement=args.max_displacement
    )
    
    return success


def find_all_sequences(dataset_root):
    """扫描数据集，找到所有包含 chest.mp4 的序列"""
    dataset_root = Path(dataset_root)
    sequences = []

    for seq_dir in dataset_root.rglob("*"):
        if not seq_dir.is_dir():
            continue
        
        chest_video = seq_dir / "chest.mp4"
        if chest_video.exists():
            sequences.append(seq_dir)

    return sequences


def main():
    parser = argparse.ArgumentParser(description='优化版 TouchAnything 数据集处理（并行 + 可视化）')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help='HaMeR 模型 checkpoint 路径'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='数据集根目录'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='指定单个序列路径（可选）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='HaMeR 推理 batch size'
    )
    parser.add_argument(
        '--frame_batch_size',
        type=int,
        default=16,
        help='一次处理的帧数（批量读取优化）'
    )
    parser.add_argument(
        '--rescale_factor',
        type=float,
        default=2.0,
        help='边界框扩展倍数'
    )
    parser.add_argument(
        '--body_detector',
        type=str,
        default='vitdet',
        choices=['vitdet', 'regnety'],
        help='人体检测器类型'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='最大处理帧数（用于测试）'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='每个GPU的工作进程数'
    )
    parser.add_argument(
        '--max_displacement',
        type=float,
        default=0.5,
        help='异常检测阈值：相邻帧最大允许位移（米），默认0.5米'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help='使用的 GPU ID，逗号分隔（如 "0,1,2"）'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的结果'
    )

    args = parser.parse_args()

    # 下载模型
    print("[INFO] 检查模型...")
    download_models(CACHE_DIR_HAMER)

    # 查找序列
    if args.sequence:
        sequences = [Path(args.sequence)]
    else:
        print(f"[INFO] 扫描数据集: {args.dataset_root}")
        sequences = find_all_sequences(args.dataset_root)
        print(f"[INFO] 找到 {len(sequences)} 个序列")

    if len(sequences) == 0:
        print("[ERROR] 未找到任何序列")
        return

    # 解析 GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    if args.num_workers == 1 or len(sequences) == 1:
        # 单进程模式（单个worker或单个序列）
        print("[INFO] 单进程模式")
        for idx, seq_dir in enumerate(sequences):
            print(f"\n[{idx + 1}/{len(sequences)}] 处理序列: {seq_dir}")
            process_single_sequence(seq_dir, args, gpu_id=gpu_ids[0])
    else:
        # 多进程模式
        print(f"[INFO] 多进程模式: {args.num_workers} 个进程")
        print(f"[INFO] 使用 GPU: {gpu_ids}")
        
        # 设置多进程启动方法为 spawn 以避免 CUDA 初始化问题
        mp.set_start_method('spawn', force=True)
        
        # 为每个序列分配 GPU
        seq_gpu_pairs = [(seq, gpu_ids[i % len(gpu_ids)], args) for i, seq in enumerate(sequences)]
        
        # 并行处理
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.starmap(process_single_sequence_wrapper, seq_gpu_pairs),
                total=len(sequences),
                desc="Processing sequences"
            ))
        
        success_count = sum(results)
        print(f"\n[INFO] 成功处理: {success_count}/{len(sequences)}")

    print("\n[INFO] 全部处理完成！")


if __name__ == '__main__':
    main()
