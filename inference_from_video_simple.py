#!/usr/bin/env python3
"""
Infer tactile pressure maps directly from an MP4 video (simplified version, run inside the hamer directory).
- Supports a single egocentric video input
- Uses HaMeR to estimate hand pose automatically
- Generates a tactile pressure prediction video
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Import chumpy early to avoid cyclic import issues
chumpy_path = Path(__file__).resolve().parents[1] / 'chumpy'
if str(chumpy_path) not in sys.path:
    sys.path.insert(0, str(chumpy_path))
try:
    import chumpy
except ImportError:
    pass

# HaMeR-related imports
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from vitpose_model import ViTPoseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.utils.config import load_config_with_base
from src.data.transforms import VideoTransform
from src.utils.vis_pressure import render_pressure_panel, PRESS_W, PRESS_H, pred_to_21

# Video layout constants
CAM_W, CAM_H = 480, 270
VIDEO_W, VIDEO_H = 1440, 630


def detect_both_hands_boxes(img, detector, cpm):
    """Detect both-hand bounding boxes in an image."""
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    
    if isinstance(detector, DefaultPredictor_Lazy):
        detector_predictions = detector(img)
        boxes = detector_predictions['instances'].pred_boxes.tensor.cpu().numpy()
        scores = detector_predictions['instances'].scores.cpu().numpy()
    else:
        boxes = detector(img)
        scores = np.ones(len(boxes))
    
    if len(boxes) == 0:
        return None, None
    
    # Append confidence score column (ViTPose expects 5 columns: x1, y1, x2, y2, score)
    boxes_with_scores = np.column_stack([boxes, scores])
    
    # Use ViTPose to detect keypoints
    pose_results = cpm.predict_pose(img, [boxes_with_scores])
    
    if len(pose_results) == 0:
        return None, None
    
    # Determine left/right hand
    right = []
    valid_boxes = []
    
    for i, result in enumerate(pose_results):
        if result is None or 'keypoints' not in result:
            continue
        
        kp = result['keypoints']
        if kp is None or len(kp) == 0:
            continue
        
        # Infer handedness from wrist and middle MCP x-coordinates
        wrist_x = kp[0, 0]
        middle_mcp_x = kp[9, 0]
        is_right = wrist_x < middle_mcp_x
        right.append(is_right)
        valid_boxes.append(boxes[i])
    
    if len(valid_boxes) == 0:
        return None, None
    
    return np.array(valid_boxes), np.array(right)


def extract_hamer_poses_from_video(
    video_path,
    detector,
    cpm,
    hamer_model,
    hamer_cfg,
    device,
    batch_size=8,
    max_frames=None
):
    """Extract HaMeR hand poses from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    poses = []
    frames = []
    
    print(f"Extracting hand poses from video: {total_frames} frames")
    
    with tqdm(total=total_frames, desc="Extracting poses") as pbar:
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            img_h, img_w = frame.shape[:2]
            left_pos = None
            right_pos = None
            
            boxes, right = detect_both_hands_boxes(frame, detector, cpm)
            
            if boxes is not None:
                dataset = ViTDetDataset(hamer_cfg, frame, boxes, right, rescale_factor=2.0)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        out = hamer_model(batch)
                    
                    pred_joints_3d = out['pred_keypoints_3d']
                    multiplier = (2 * batch['right'] - 1)
                    pred_cam = out['pred_cam'].clone()
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                    
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    
                    scaled_focal_length = (
                        hamer_cfg.EXTRA.FOCAL_LENGTH / hamer_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    )
                    
                    pred_cam_t_full = cam_crop_to_full(
                        pred_cam, box_center, box_size, img_size, scaled_focal_length
                    ).detach().cpu().numpy()
                    
                    bs = batch['img'].shape[0]
                    for n in range(bs):
                        joints_3d = pred_joints_3d[n].detach().cpu().numpy()
                        is_right_hand = int(batch['right'][n].detach().cpu().item())
                        
                        joints_3d[:, 0] = (2 * is_right_hand - 1) * joints_3d[:, 0]
                        joints_3d_cam = joints_3d + pred_cam_t_full[n][None, :]
                        
                        if is_right_hand:
                            right_pos = joints_3d_cam
                        else:
                            left_pos = joints_3d_cam
            
            poses.append({
                'left_pos': left_pos if left_pos is not None else np.zeros((21, 3)),
                'right_pos': right_pos if right_pos is not None else np.zeros((21, 3))
            })
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    return poses, frames


def infer_tactile_from_video(
    video_path, model, device, cfg, hamer_model, hamer_cfg, detector, cpm,
    clip_length=8, frame_interval=1, batch_size=16
):
    """Infer tactile pressure maps from a video."""
    poses, frames = extract_hamer_poses_from_video(
        video_path, detector, cpm, hamer_model, hamer_cfg, device
    )
    
    T = len(frames)
    print(f"Extracted {T} frames and pose entries")
    
    image_size = cfg['data']['image_size']
    transform = VideoTransform(
        image_size=image_size,
        normalize_mean=cfg['data'].get('mean', [0.485, 0.456, 0.406]),
        normalize_std=cfg['data'].get('std', [0.229, 0.224, 0.225]),
        use_augmentation=False
    )
    
    pred_maps_all = []
    
    print(f"Tactile inference: {(T - clip_length) // frame_interval + 1} clips")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, T - clip_length + 1, frame_interval), desc="Tactile inference"):
            end_idx = start_idx + clip_length
            
            clip_frames = np.stack([frames[i] for i in range(start_idx, end_idx)])
            frames_t = transform(clip_frames).unsqueeze(0).to(device)
            
            # Merge left and right hand poses (B, T, 42, 3) - 21 joints x 2 hands
            left_poses = np.stack([poses[i]['left_pos'] for i in range(start_idx, end_idx)])
            right_poses = np.stack([poses[i]['right_pos'] for i in range(start_idx, end_idx)])
            combined_poses = np.concatenate([left_poses, right_poses], axis=1)  # (T, 42, 3)
            
            poses_t = torch.from_numpy(combined_poses).float().unsqueeze(0).to(device)  # (1, T, 42, 3)
            
            with torch.cuda.amp.autocast():
                outputs = model(frames_t, poses=poses_t)
            
            tactile = outputs['tactile'].cpu().numpy()[0]  # (T, 2, 16, 16)
            pred_maps_all.append(tactile[-1])  # Take the last frame (2, 16, 16)
    
    pred_maps_all = np.array(pred_maps_all)
    return np.array(frames), pred_maps_all


def make_video_with_prediction(frames, pred_maps, output_path, fps=10, vmax=1.0):
    """Generate a video with predicted tactile maps."""
    import subprocess
    
    T = frames.shape[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{VIDEO_W}x{VIDEO_H}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    nan_mask_l = np.zeros((21, 21), dtype=bool)
    nan_mask_r = np.zeros((21, 21), dtype=bool)
    
    for t in range(T):
        cam_frame = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        cam_frame = cv2.resize(cam_frame, (CAM_W, CAM_H))
        
        row_top = np.zeros((CAM_H, VIDEO_W, 3), dtype=np.uint8)
        row_top[:, :CAM_W] = cam_frame
        
        if t < len(pred_maps) and not np.all(np.isnan(pred_maps[t])):
            pred_l21 = pred_to_21(pred_maps[t, 0], nan_mask_l)
            pred_r21 = pred_to_21(pred_maps[t, 1], nan_mask_r)
            pred_panel = render_pressure_panel(
                pred_l21, pred_r21, title='Predicted Pressure',
                w=PRESS_W, h=PRESS_H, vmax=vmax, global_max=vmax
            )
        else:
            pred_panel = np.zeros((PRESS_H, PRESS_W, 3), dtype=np.uint8)
            cv2.putText(pred_panel, 'No Prediction', (PRESS_W//2 - 80, PRESS_H//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        empty_panel = np.zeros((PRESS_H, PRESS_W, 3), dtype=np.uint8)
        row_bottom = np.hstack([empty_panel, pred_panel])
        
        frame_out = np.vstack([row_top, row_bottom])
        
        cv2.putText(frame_out, f'Frame {t+1}/{T}',
                   (10, VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(frame_out, 'Video Input',
                   (VIDEO_W - 120, 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 200, 120), 2, cv2.LINE_AA)
        
        ffmpeg_proc.stdin.write(frame_out.tobytes())
    
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Infer tactile pressure maps from an MP4 video')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--hamer_checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=str, default='output_tactile.mp4')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hamer_batch_size', type=int, default=8)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load tactile prediction model
    print("Loading tactile prediction model...")
    cfg = load_config_with_base(args.config)
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("✓ Tactile model loaded")
    
    # 2. Load HaMeR pose model
    print("Loading HaMeR pose estimation model...")
    hamer_model, hamer_cfg = load_hamer(args.hamer_checkpoint)
    hamer_model = hamer_model.to(device)
    hamer_model.eval()
    print("✓ HaMeR model loaded")
    
    # 3. Load detector
    print("Loading hand detector...")
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    print("✓ Detector loaded")
    
    # 4. Load ViTPose
    print("Loading ViTPose keypoint detector...")
    cpm = ViTPoseModel(device)
    print("✓ ViTPose loaded")
    
    # 5. Process video
    print(f"\nProcessing video: {args.video}")
    frames, pred_maps = infer_tactile_from_video(
        args.video, model, device, cfg, hamer_model, hamer_cfg, detector, cpm,
        clip_length=cfg['data'].get('clip_length', 8),
        frame_interval=cfg['data'].get('frame_interval', 1),
        batch_size=args.batch_size
    )
    
    # 6. Generate output video
    print(f"\nGenerating output video...")
    make_video_with_prediction(frames, pred_maps, args.output, fps=args.fps, vmax=1.0)
    
    print(f"\n✓ Done! Output video: {args.output}")


if __name__ == '__main__':
    main()
