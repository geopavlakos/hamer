"""
处理 TouchAnything 数据集，为每个序列生成 hamer_hands.json
数据格式与 rokoko_hands.json 保持一致
"""
from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
import json
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

from vitpose_model import ViTPoseModel


def detect_both_hands_boxes(img_cv2, detector, cpm):
    """
    检测图像中的左右手边界框
    返回: boxes (N, 4), is_right (N,)
    """
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


def process_one_frame(
    img_cv2,
    detector,
    cpm,
    model,
    model_cfg,
    device,
    batch_size=1,
    rescale_factor=2.0
):
    """
    处理单帧图像，返回左右手的 3D 关节点
    返回: left_pos (21, 3), right_pos (21, 3) 或 None
    """
    boxes, right = detect_both_hands_boxes(img_cv2, detector, cpm)
    if boxes is None:
        return None, None

    dataset = ViTDetDataset(
        model_cfg,
        img_cv2,
        boxes,
        right,
        rescale_factor=rescale_factor
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    left_pos = None
    right_pos = None

    for batch in dataloader:
        batch = recursive_to(batch, device)

        with torch.no_grad():
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

            if is_right_hand:
                right_pos = joints_3d_cam.tolist()
            else:
                left_pos = joints_3d_cam.tolist()

    return left_pos, right_pos


def process_video_sequence(
    video_path,
    detector,
    cpm,
    model,
    model_cfg,
    device,
    output_path,
    batch_size=1,
    rescale_factor=2.0,
    max_frames=None
):
    """
    处理视频序列，生成 hamer_hands.json
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    print(f"[INFO] 处理视频: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, 总帧数: {total_frames}")

    results = []
    frame_index = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames is not None and frame_index >= max_frames):
                break

            # 计算时间戳（假设从 0 开始）
            timestamp = frame_index / fps if fps > 0 else frame_index * 0.033

            left_pos, right_pos = process_one_frame(
                frame,
                detector,
                cpm,
                model,
                model_cfg,
                device,
                batch_size,
                rescale_factor
            )

            # 如果没有检测到手，使用空列表
            if left_pos is None:
                left_pos = []
            if right_pos is None:
                right_pos = []

            frame_data = {
                "ts": timestamp,
                "frame_index": frame_index,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "aligned_to_vive": False
            }

            results.append(frame_data)
            frame_index += 1
            pbar.update(1)

    cap.release()

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"[INFO] 已保存到: {output_path}")
    print(f"[INFO] 处理了 {len(results)} 帧")
    return True


def find_all_sequences(dataset_root):
    """
    扫描数据集，找到所有包含 chest.mp4 的序列
    """
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
    parser = argparse.ArgumentParser(description='Process TouchAnything dataset with HaMeR')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help='Path to pretrained model checkpoint'
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
        default=1,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--rescale_factor',
        type=float,
        default=2.0,
        help='Factor for padding the bbox'
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
        help='最大处理帧数（用于测试）'
    )

    args = parser.parse_args()

    # 加载模型
    print("[INFO] 加载 HaMeR 模型...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # 加载检测器
    print("[INFO] 加载人体检测器...")
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
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
    print("[INFO] 加载 ViTPose 模型...")
    cpm = ViTPoseModel(device)

    # 查找序列
    if args.sequence:
        sequences = [Path(args.sequence)]
    else:
        print(f"[INFO] 扫描数据集: {args.dataset_root}")
        sequences = find_all_sequences(args.dataset_root)
        print(f"[INFO] 找到 {len(sequences)} 个序列")

    # 处理每个序列
    for idx, seq_dir in enumerate(sequences):
        print(f"\n[{idx + 1}/{len(sequences)}] 处理序列: {seq_dir}")
        
        chest_video = seq_dir / "chest.mp4"
        output_json = seq_dir / "hamer_hands.json"

        if output_json.exists():
            print(f"[SKIP] 已存在 hamer_hands.json")
            continue

        success = process_video_sequence(
            chest_video,
            detector,
            cpm,
            model,
            model_cfg,
            device,
            output_json,
            batch_size=args.batch_size,
            rescale_factor=args.rescale_factor,
            max_frames=args.max_frames
        )

        if not success:
            print(f"[ERROR] 处理失败: {seq_dir}")

    print("\n[INFO] 全部处理完成！")


if __name__ == '__main__':
    main()
