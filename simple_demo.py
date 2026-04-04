from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

from vitpose_model import ViTPoseModel


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]


def project_points(points_3d, focal_length, img_w, img_h):
    """
    使用 HaMeR 原始思路做投影：
        u = f * X / Z + W/2
        v = f * Y / Z + H/2
    """
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    eps = 1e-9

    u = focal_length * (x / (z + eps)) + img_w / 2.0
    v = focal_length * (y / (z + eps)) + img_h / 2.0
    return np.stack([u, v], axis=1)


def draw_one_hand(canvas, joints_2d, is_right, edges=HAND_EDGES):
    """
    在图上绘制一只手。
    这里用不同颜色区分左右手：
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
        cv2.circle(canvas, (x, y), 2, point_color, -1)

    return canvas


def get_all_frame_items(dataset_root):
    """
    扫描数据集，假设结构：
        dataset_root/
            动作名/
                演示名/
                    data/
                        frame_xxx/
                            chest_color.png
    """
    dataset_root = Path(dataset_root)
    items = []

    for action_dir in sorted(dataset_root.iterdir()):
        if not action_dir.is_dir():
            continue

        for demo_dir in sorted(action_dir.iterdir()):
            if not demo_dir.is_dir():
                continue

            data_dir = demo_dir / "data"
            if not data_dir.exists() or not data_dir.is_dir():
                print(f"[WARN] 缺少 data 文件夹，跳过: {demo_dir}")
                continue

            for frame_dir in sorted(data_dir.iterdir(), key=lambda x: x.name):
                if not frame_dir.is_dir():
                    continue

                img_path = frame_dir / "chest_color.png"
                if not img_path.exists():
                    print(f"[WARN] 缺少 chest_color.png，跳过: {img_path}")
                    continue

                items.append({
                    "action_dir": action_dir,
                    "demo_dir": demo_dir,
                    "frame_dir": frame_dir,
                    "img_path": img_path,
                })

    return items


def detect_both_hands_boxes(img_cv2, detector, cpm):
    """
    检测整张图中的左右手 bbox。
    使用 ViTPose 的手部关键点：
        - 左手: keypoints[-42:-21]
        - 右手: keypoints[-21:]

    返回：
        boxes: (N, 4)
        is_right: (N,)
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


def process_one_image(
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
    对单张图同时估计双手，使用 HaMeR 原始相机估计方式。
    返回：
        overlay_img
        left_hand_results
        right_hand_results
    """
    overlay_img = img_cv2.copy()
    left_hand_results = []
    right_hand_results = []

    boxes, right = detect_both_hands_boxes(img_cv2, detector, cpm)
    if boxes is None:
        return overlay_img, left_hand_results, right_hand_results

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

    img_h, img_w = img_cv2.shape[:2]

    for batch in dataloader:
        batch = recursive_to(batch, device)

        with torch.no_grad():
            out = model(batch)

        pred_joints_3d = out['pred_keypoints_3d']   # (B, 21, 3)
        pred_joints_2d = out['pred_keypoints_2d']   # (B, 21, 2)

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
            joints_2d_crop = pred_joints_2d[n].detach().cpu().numpy()

            is_right_hand = int(batch['right'][n].detach().cpu().item())
            joints_3d[:, 0] = (2 * is_right_hand - 1) * joints_3d[:, 0]
            person_id = int(batch['personid'][n].detach().cpu().item())

            if scaled_focal_length.ndim == 0:
                focal_length_value = float(scaled_focal_length.detach().cpu().item())
            else:
                focal_length_value = float(scaled_focal_length[n].detach().cpu().item())

            joints_3d_cam = joints_3d + pred_cam_t_full[n][None, :]

            joints_2d_full = project_points(
                joints_3d_cam,
                focal_length=focal_length_value,
                img_w=img_w,
                img_h=img_h
            )

            overlay_img = draw_one_hand(
                overlay_img,
                joints_2d_full,
                is_right=is_right_hand
            )

            hand_info = {
                "person_id": person_id,
                "is_right": is_right_hand,
                "joints_3d_cam": joints_3d_cam.tolist(),
                "joints_2d_full": joints_2d_full.tolist(),
                "pred_keypoints_2d_crop": joints_2d_crop.tolist(),
                "pred_cam_t_full": pred_cam_t_full[n].tolist(),
                "focal_length": focal_length_value,
            }

            if is_right_hand:
                right_hand_results.append(hand_info)
            else:
                left_hand_results.append(hand_info)

    return overlay_img, left_hand_results, right_hand_results


def build_output_dir(out_root, dataset_root, frame_dir):
    """
    保持原始文件夹名一致，只移除中间的 data 层。
    例如：
        frame_dir = touchanything_datasets/拿水杯/action 1/data/0001
    输出：
        pose_data/拿水杯/action 1/0001
    """
    dataset_root = Path(dataset_root)
    frame_dir = Path(frame_dir)
    relative_parts = frame_dir.relative_to(dataset_root).parts

    relative_parts_wo_data = [p for p in relative_parts if p != "data"]
    save_dir = Path(out_root).joinpath(*relative_parts_wo_data)
    return save_dir


def main():
    parser = argparse.ArgumentParser(description='HaMeR dataset inference for both hands')
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
        help='数据集根目录，例如 touchanything_datasets'
    )
    parser.add_argument(
        '--out_root',
        type=str,
        default='pose_data',
        help='输出根目录'
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
        help='Using regnety improves runtime and reduces memory'
    )

    args = parser.parse_args()

    # 下载并加载模型
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # detector
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

    # keypoint detector
    cpm = ViTPoseModel(device)

    os.makedirs(args.out_root, exist_ok=True)

    all_items = get_all_frame_items(args.dataset_root)
    print(f"[INFO] 一共找到 {len(all_items)} 帧需要处理")

    for idx, item in enumerate(all_items):
        img_path = item["img_path"]
        frame_dir = item["frame_dir"]

        print(f"[{idx + 1}/{len(all_items)}] Processing: {img_path}")

        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"[WARN] 无法读取图像，跳过: {img_path}")
            continue

        overlay_img, left_hand_results, right_hand_results = process_one_image(
            img_cv2=img_cv2,
            detector=detector,
            cpm=cpm,
            model=model,
            model_cfg=model_cfg,
            device=device,
            batch_size=args.batch_size,
            rescale_factor=args.rescale_factor
        )

        save_dir = build_output_dir(
            out_root=args.out_root,
            dataset_root=args.dataset_root,
            frame_dir=frame_dir
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = save_dir / "overlay.jpg"
        left_json_path = save_dir / "left_hand.json"
        right_json_path = save_dir / "right_hand.json"

        cv2.imwrite(str(overlay_path), overlay_img)

        left_payload = {
            "source_image": str(img_path),
            "num_hands": len(left_hand_results),
            "hands": left_hand_results
        }

        right_payload = {
            "source_image": str(img_path),
            "num_hands": len(right_hand_results),
            "hands": right_hand_results
        }

        with open(left_json_path, 'w', encoding='utf-8') as f:
            json.dump(left_payload, f, ensure_ascii=False, indent=2)

        with open(right_json_path, 'w', encoding='utf-8') as f:
            json.dump(right_payload, f, ensure_ascii=False, indent=2)

    print("[INFO] 全部处理完成")


if __name__ == '__main__':
    main()