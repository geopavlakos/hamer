"""
Optimized TouchAnything dataset processing script.
- Batched inference optimization (process multiple frames at once)
- Multiprocess parallel processing
- Video visualization (draw hand skeleton)
- Generate hamer_hands.json
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


# Hand skeleton connectivity
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]


def project_points(points_3d, focal_length, img_w, img_h):
    """Project 3D points to 2D image coordinates."""
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    eps = 1e-9

    u = focal_length * (x / (z + eps)) + img_w / 2.0
    v = focal_length * (y / (z + eps)) + img_h / 2.0
    return np.stack([u, v], axis=1)


def detect_outliers_and_smooth(results, max_displacement=0.5):
    """
    Detect abrupt outliers and smooth them.
    
    Strategy:
    1. Compute displacement between adjacent frames
    2. Mark as outlier if displacement exceeds threshold
    3. Replace outlier frame with previous frame result
    
    Args:
        results: List of per-frame results
        max_displacement: Maximum allowed displacement (meters), default 0.5m
    """
    if not results or len(results) < 2:
        return results
    
    n_frames = len(results)
    
    # Process left and right hands separately
    for hand_key in ['left_pos', 'right_pos']:
        outlier_count = 0
        
        for i in range(1, n_frames):
            curr_pos = results[i][hand_key]
            prev_pos = results[i-1][hand_key]
            
            # Skip if current or previous frame has no detection
            if not curr_pos or not prev_pos:
                continue
            
            # Compute wrist displacement (joint index 0)
            curr_wrist = np.array(curr_pos[0])
            prev_wrist = np.array(prev_pos[0])
            displacement = np.linalg.norm(curr_wrist - prev_wrist)
            
            # If displacement is too large, mark as outlier and use previous frame
            if displacement > max_displacement:
                results[i][hand_key] = prev_pos
                outlier_count += 1
        
        if outlier_count > 0:
            print(f"[INFO] {hand_key}: detected {outlier_count} outlier jump frames, smoothing applied")
    
    return results


def interpolate_missing_hands(results):
    """
    Interpolate and fill missing hand poses.
    
    Strategy:
    1. Linear interpolation if both neighboring detections exist
    2. Forward fill if only previous detection exists
    3. Backward fill if only next detection exists
    """
    if not results:
        return results
    
    n_frames = len(results)
    
    # Process left and right hands separately
    for hand_key in ['left_pos', 'right_pos']:
        # Find indices of valid frames
        valid_indices = [i for i, r in enumerate(results) if r[hand_key]]
        
        if not valid_indices:
            continue
        
        # Stats
        missing_count = n_frames - len(valid_indices)
        if missing_count > 0:
            print(f"[INFO] {hand_key}: detected {len(valid_indices)}/{n_frames} frames, missing {missing_count} frames")
        
        # Process missing frames
        for i in range(n_frames):
            if results[i][hand_key]:
                continue
            
            # Find nearest previous/next valid frames
            prev_idx = None
            next_idx = None
            
            for vi in valid_indices:
                if vi < i:
                    prev_idx = vi
                elif vi > i:
                    next_idx = vi
                    break
            
            # Interpolate or fill
            if prev_idx is not None and next_idx is not None:
                # Linear interpolation
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                prev_pos = np.array(results[prev_idx][hand_key])
                next_pos = np.array(results[next_idx][hand_key])
                interpolated_pos = (1 - alpha) * prev_pos + alpha * next_pos
                results[i][hand_key] = interpolated_pos.tolist()
                
            elif prev_idx is not None:
                # Forward fill
                results[i][hand_key] = results[prev_idx][hand_key]
                
            elif next_idx is not None:
                # Backward fill
                results[i][hand_key] = results[next_idx][hand_key]
    
    return results


def load_jsonl_results(json_path):
    """Load line-delimited JSON results."""
    results = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def save_jsonl_results(json_path, results):
    """Save line-delimited JSON results."""
    with open(json_path, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def fill_missing_hands_with_nearest(results):
    """
    Fill missing hand results using temporally nearest valid frames.

    Two-step strategy:
    1. If both left_pos and right_pos are empty in a frame, copy from nearest valid frame
    2. If only one hand is missing, fill per-hand from nearest valid frame

    If both previous and next valid frames exist, choose the closer one;
    if equally distant, prefer the previous frame.
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
    Check existing hamer_hands.json for missing hand results and repair in place if needed.

    Returns:
        bool: True means check/repair succeeded; False means read failed and video should be reprocessed.
    """
    try:
        results = load_jsonl_results(json_path)
    except Exception as exc:
        print(f"[WARN] Failed to read existing JSON, will reprocess video: {json_path}, error={exc}")
        return False

    results, fill_stats = fill_missing_hands_with_nearest(results)
    total_filled = sum(fill_stats.values())

    if total_filled == 0:
        print(f"[INFO] Existing JSON has no missing hand frames, keeping original result: {json_path}")
        return True

    save_jsonl_results(json_path, results)
    print(
        f"[INFO] Repaired existing JSON: {json_path} "
        f"(full-frame replacement: {fill_stats.get('frame_level', 0)} frames, "
        f"left_pos filled: {fill_stats.get('left_pos', 0)} frames, "
        f"right_pos filled: {fill_stats.get('right_pos', 0)} frames)"
    )
    return True


def draw_one_hand(canvas, joints_2d, is_right, edges=HAND_EDGES):
    """
    Draw skeleton for one hand.
    Right hand: blue lines + green points
    Left hand: red lines + yellow points
    """
    if is_right:
        line_color = (255, 0, 0)    # Blue
        point_color = (0, 255, 0)   # Green
    else:
        line_color = (0, 0, 255)    # Red
        point_color = (0, 255, 255) # Yellow

    for i, j in edges:
        pt1 = tuple(np.round(joints_2d[i]).astype(int))
        pt2 = tuple(np.round(joints_2d[j]).astype(int))
        cv2.line(canvas, pt1, pt2, line_color, 2)

    for pt in joints_2d:
        x, y = np.round(pt).astype(int)
        cv2.circle(canvas, (x, y), 3, point_color, -1)

    return canvas


def detect_both_hands_boxes(img_cv2, detector, cpm):
    """Detect left/right hand bounding boxes in an image."""
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

        # Left hand 21 keypoints
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

        # Right hand 21 keypoints
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
    Process multiple frames in batch and return per-frame results.
    
    Returns:
        results: List[dict] - result for each frame
            {
                'left_pos': [[x,y,z], ...] or None,
                'right_pos': [[x,y,z], ...] or None,
                'vis_frame': np.ndarray - visualization frame,
                'focal_length': float - focal length
            }
    """
    # Detect hand boxes for all frames in batch
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
        
        focal_length_value = max(img_w, img_h)  # Default focal length
        
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
                num_workers=4,  # Parallel data loading
                pin_memory=True  # Speed up GPU transfer
            )

            for batch in dataloader:
                batch = recursive_to(batch, device)

                # Mixed precision inference for speed
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
                    
                    # Mirror correction
                    joints_3d[:, 0] = (2 * is_right_hand - 1) * joints_3d[:, 0]
                    
                    # Convert to camera coordinates
                    joints_3d_cam = joints_3d + pred_cam_t_full[n][None, :]
                    
                    # Project to 2D for visualization
                    if scaled_focal_length.ndim == 0:
                        current_focal_length = float(scaled_focal_length.detach().cpu().item())
                    else:
                        current_focal_length = float(scaled_focal_length[n].detach().cpu().item())
                    
                    # Update focal length (use computed value)
                    focal_length_value = current_focal_length
                    
                    joints_2d = project_points(
                        joints_3d_cam,
                        focal_length=current_focal_length,
                        img_w=img_w,
                        img_h=img_h
                    )
                    
                    # Draw skeleton
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
    resize_scale=0.5  # Added: image resize ratio, 0.5 = half size, ~2-3x speedup
):
    """
    Process a video sequence and generate hamer_hands.json plus a visualization video.
    
    Args:
        batch_size: HaMeR model inference batch size
        frame_batch_size: Number of frames read/processed per iteration
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, Total frames: {total_frames}, Resolution: {width}x{height}")

    # Create video writer later (after post-processing)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    # Try H.264 first; fall back to more generic mp4v if unsupported
    preferred_codecs = ['avc1', 'mp4v']

    results = []
    frame_index = 0
    
    import time
    time_stats = {'read': 0, 'inference': 0, 'save': 0}
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            # Read frames in batch
            t0 = time.time()
            frames = []
            frame_indices = []
            
            for _ in range(frame_batch_size):
                if max_frames is not None and frame_index >= max_frames:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Optional frame skipping (frame_skip=2 means process 1 out of every 2 frames)
                # frame_skip = 1  # No frame skipping
                # if frame_index % frame_skip != 0:
                #     frame_index += 1
                #     continue
                    
                frames.append(frame)
                frame_indices.append(frame_index)
                frame_index += 1
            
            if len(frames) == 0:
                break
            
            time_stats['read'] += time.time() - t0
            
            # Batch processing
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
            
            # Save results and visualization frames
            for idx, (frame_idx, result) in enumerate(zip(frame_indices, batch_results)):
                timestamp = frame_idx / fps if fps > 0 else frame_idx * 0.033
                
                frame_data = {
                    "ts": timestamp,
                    "frame_index": frame_idx,
                    "left_pos": result['left_pos'],
                    "right_pos": result['right_pos'],
                    "aligned_to_vive": False,
                    "vis_frame": result['vis_frame']  # Save visualization frame
                }
                
                results.append(frame_data)
            
            pbar.update(len(frames))

    cap.release()

    # Post-processing: fill missing hand results with nearest valid frames
    results, fill_stats = fill_missing_hands_with_nearest(results)
    total_filled = sum(fill_stats.values())
    if total_filled > 0:
        print(
            "\n[INFO] Filled missing hand frames in newly generated results: "
            f"full-frame replacement: {fill_stats.get('frame_level', 0)} frames, "
            f"left_pos filled: {fill_stats.get('left_pos', 0)} frames, "
            f"right_pos filled: {fill_stats.get('right_pos', 0)} frames"
        )

    # Write visualization video (using the first generated vis_frame)
    print("\n[INFO] Writing visualization video...")
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
            f"Failed to create video writer. Tried codecs: {preferred_codecs}"
        )
    print(f"[INFO] Visualization video codec: {selected_codec}")
    
    for result in tqdm(results, desc="Writing visualization"):
        # Directly use the first generated vis_frame
        video_writer.write(result['vis_frame'])
        
        # Remove vis_frame to save memory
        del result['vis_frame']
    
    video_writer.release()
    time_stats['save'] = time.time() - t_write_start
    
    # Print performance stats
    total_time = sum(time_stats.values())
    print(f"\n[Performance Stats]")
    print(f"  Video read: {time_stats['read']:.2f}s ({time_stats['read']/total_time*100:.1f}%)")
    print(f"  Model inference: {time_stats['inference']:.2f}s ({time_stats['inference']/total_time*100:.1f}%)")
    print(f"  Video write: {time_stats['save']:.2f}s ({time_stats['save']/total_time*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")

    # Save JSON results
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl_results(output_json_path, results)

    print(f"[INFO] JSON saved to: {output_json_path}")
    print(f"[INFO] Video saved to: {output_video_path}")
    print(f"[INFO] Processed {len(results)} frames")
    return True


def process_single_sequence_wrapper(seq_dir, gpu_id, args):
    """Wrapper function for multiprocess starmap."""
    return process_single_sequence(seq_dir, args, gpu_id=gpu_id)


def process_single_sequence(seq_dir, args, gpu_id=0):
    """Process a single sequence (for multiprocessing)."""
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Performance optimization: enable PyTorch backend optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load model
    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # torch.compile and FP16 are disabled (first run can be slow)
    # To enable, uncomment below:
    # model = torch.compile(model, mode='reduce-overhead')
    # model = model.half()
    
    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        local_detectron2_ckpt = Path(CACHE_DIR_HAMER) / 'detectron2_models' / 'model_final_f05665.pkl'
        if local_detectron2_ckpt.exists():
            detectron2_cfg.train.init_checkpoint = str(local_detectron2_ckpt)
            print(f"[INFO] Using local Detectron2 weights: {local_detectron2_ckpt}")
        else:
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5  # Raise threshold to reduce false positives
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
    
    # Load keypoint detector
    cpm = ViTPoseModel(device)
    
    chest_video = seq_dir / "chest.mp4"
    output_json = seq_dir / "hamer_hands.json"
    output_video = seq_dir / "hamer_visualization.mp4"
    
    if output_json.exists() and not args.overwrite:
        print(f"[INFO] Existing hamer_hands.json found, checking missing hand frames: {seq_dir}")
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
    """Scan dataset and find all sequences containing chest.mp4."""
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
    parser = argparse.ArgumentParser(description='Optimized TouchAnything dataset processing (parallel + visualization)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help='Path to HaMeR model checkpoint'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Dataset root directory'
    )
    parser.add_argument(
        '--sequence',
        type=str,
        default=None,
        help='Path to a single sequence (optional)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='HaMeR inference batch size'
    )
    parser.add_argument(
        '--frame_batch_size',
        type=int,
        default=16,
        help='Number of frames to process at once (batched read optimization)'
    )
    parser.add_argument(
        '--rescale_factor',
        type=float,
        default=2.0,
        help='Bounding-box expansion factor'
    )
    parser.add_argument(
        '--body_detector',
        type=str,
        default='vitdet',
        choices=['vitdet', 'regnety'],
        help='Body detector type'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (for testing)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker processes per GPU'
    )
    parser.add_argument(
        '--max_displacement',
        type=float,
        default=0.5,
        help='Outlier threshold: max allowed displacement between adjacent frames (m), default 0.5'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help='GPU IDs to use, comma-separated (e.g. "0,1,2")'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )

    args = parser.parse_args()

    # Download/check models
    print("[INFO] Checking models...")
    download_models(CACHE_DIR_HAMER)

    # Find sequences
    if args.sequence:
        sequences = [Path(args.sequence)]
    else:
        print(f"[INFO] Scanning dataset: {args.dataset_root}")
        sequences = find_all_sequences(args.dataset_root)
        print(f"[INFO] Found {len(sequences)} sequences")

    if len(sequences) == 0:
        print("[ERROR] No sequences found")
        return

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    
    if args.num_workers == 1 or len(sequences) == 1:
        # Single-process mode (single worker or single sequence)
        print("[INFO] Single-process mode")
        for idx, seq_dir in enumerate(sequences):
            print(f"\n[{idx + 1}/{len(sequences)}] Processing sequence: {seq_dir}")
            process_single_sequence(seq_dir, args, gpu_id=gpu_ids[0])
    else:
        # Multiprocess mode
        print(f"[INFO] Multiprocess mode: {args.num_workers} processes")
        print(f"[INFO] Using GPUs: {gpu_ids}")
        
        # Set multiprocessing start method to spawn to avoid CUDA init issues
        mp.set_start_method('spawn', force=True)
        
        # Assign GPU to each sequence
        seq_gpu_pairs = [(seq, gpu_ids[i % len(gpu_ids)], args) for i, seq in enumerate(sequences)]
        
        # Parallel processing
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.starmap(process_single_sequence_wrapper, seq_gpu_pairs),
                total=len(sequences),
                desc="Processing sequences"
            ))
        
        success_count = sum(results)
        print(f"\n[INFO] Successfully processed: {success_count}/{len(sequences)}")

    print("\n[INFO] All processing completed!")


if __name__ == '__main__':
    main()
