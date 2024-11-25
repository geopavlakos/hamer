from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
#from ultralytics import YOLO
import pprint
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.video_frame_processing import get_less_blurry_image
#from BIMEF import BIMEF            
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

#from vitpose_model import ViTPoseModel
from tqdm import tqdm

import json
from typing import Dict, Optional
from mmpose.apis import MMPoseInferencer

#def visualization(image, bboxes, keypoints):
    
#vizqimage m cv2ning.rectangle(image, start_point, end_point, color, thickness)
    
#    return viz_image
pose_model_cfg = './mm/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py'

#ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth'
#ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
#ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb32-140e_coco-384x288-45c3ba34_20220913.pth'
ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'
device = 'cuda'
mmdet_cfg = './rtmdet_m_640-8xb32_coco-person.py'
mmdet_ckpt = './rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='yolo', choices=['vitdet', 'regnety', 'yolo'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--video_path', type=str, default="", help='Video path')
    parser.add_argument('--yolo_model_path', type=str, default="yolo_weight_best.pt", help='Yolo model path')


    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    inferencer = MMPoseInferencer(
        # suppose the pose estimator is trained on custom dataset
        pose2d=pose_model_cfg,
        pose2d_weights=ckpt,
        det_model=mmdet_cfg,
        det_weights=mmdet_ckpt,
        det_cat_ids=[0]
    )

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    #img_paths = [img for emnd in args.file_type for img in Path(args.img_folder).glob(end)]
    
    cap = cv2.VideoCapture(args.video_path)
    # Iterate over all images in folder
    #pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    #for img_path in img_paths:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_skip = 1
    frames = np.array([None]*frame_skip)
    visualization = np.array([None]*frame_skip)
    image_size = (960,1920)
    resize = True
    visualization = True
    video=cv2.VideoWriter('video_rtm.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(960,1920))
    frame_num = 0
    frame_skip_count = 0
    try:
        with tqdm(total=total_frames, desc="Prcessing Video", unit="frame") as pbar:  
            while cap.isOpened():
                frame_num += 1
                
                #if frame_skip_count <= 4:
                flag, frame = cap.read()
                img_cv2 = frame # cv2.imread(str(img_path))
                
                if resize:
                    img_cv2 = cv2.resize(img_cv2, image_size,  
                        interpolation = cv2.INTER_LINEAR)
                    
                frames[frame_skip_count] = img_cv2
                frame_skip_count += 1
                    
                if frame_skip_count == frame_skip:
                    frame_skip_count = 0
                    idx = get_less_blurry_image(frames)
                               
                    #flag, frame = cap.read()
                
                    img_cv2 = frames[idx] # cv2.imread(str(img_path))
                   # Detect humans in image\
                    bboxes = []
                    is_right = []
                    result_generator = inferencer([img_cv2], return_vis=True, draw_bbox=True)
                    result = next(result_generator)
                    frames[idx] = result["visualization"][0]
                    vitposes_out = result["predictions"]
                    pprint.pp(len(vitposes_out))
                    # Use hands based on hand keypoint detections
                    for vitposes in vitposes_out[0]:
                        #pprint.pp(vitposes[0])
                        #continue
                        #print(vitposes)
                        if vitposes['bbox_score'] < 0.35:
                            continue
                        left_hand_keyp = np.array(vitposes['keypoints'][-42:-21])
                        right_hand_keyp = np.array(vitposes['keypoints'][-21:])
                        conf_left_hand_keyp = np.array(vitposes['keypoint_scores'][-42:-21])
                        conf_right_hand_keyp = np.array(vitposes['keypoint_scores'][-21:])

                        
                        # Rejecting not confident detections
                        #vitposes[0]['bbox_score']
                        valid = conf_left_hand_keyp > 0.4
                        keyp = left_hand_keyp
                        if sum(valid) > 3:
                            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                            bboxes.append(bbox)
                            is_right.append(0)
                    
                        valid = conf_right_hand_keyp > 0.4
                        keyp = right_hand_keyp
                        if sum(valid) > 3:
                            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                            bboxes.append(bbox)
                            is_right.append(1)

                    if len(bboxes) == 0:
                        continue

                    boxes = np.stack(bboxes)
                    right = np.stack(is_right)

                    # Run reconstruction on all detected hands
                    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

                    all_verts = []
                    all_cam_t = []
                    all_right = []
                    
                    for batch in dataloader:
                        batch = recursive_to(batch, device)
                        with torch.no_grad():
                            out = model(batch)

                        multiplier = (2*batch['right']-1)
                        pred_cam = out['pred_cam']
                        pred_cam[:,1] = multiplier*pred_cam[:,1]
                        box_center = batch["box_center"].float()
                        box_size = batch["box_size"].float()
                        img_size = batch["img_size"].float()
                        multiplier = (2*batch['right']-1)
                        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                        # Render the result
                        batch_size = batch['img'].shape[0]
                        for n in range(batch_size):
                            # Get filename from path img_path
                            #img_fn, _ = os.path.splitext(os.path.basename(img_path))
                            
                            person_id = int(batch['personid'][n])
                            #white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                            #input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                            #input_patch = input_patch.permute(1,2,0).numpy()

                            #regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                            #                        out['pred_cam_t'][n].detach().cpu().numpy(),
                            #                        batch['img'][n],
                            #                        mesh_base_color=LIGHT_BLUE,
                            #                        scene_bg_color=(1, 1, 1),
                            #                        )

                            #if args.side_view:
                            #    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                            #                            out['pred_cam_t'][n].detach().cpu().numpy(),
                            #                            white_img,
                            #                            mesh_base_color=LIGHT_BLUE,
                            #                            scene_bg_color=(1, 1, 1),
                            #                            side_view=True)
                            #    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                            #else:
                            #    final_img = np.concatenate([input_patch, regression_img], axis=1)

                            #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                            # Add all verts and cams to list
                            verts = out['pred_vertices'][n].detach().cpu().numpy()
                            is_right = batch['right'][n].cpu().numpy()
                            verts[:,0] = (2*is_right-1)*verts[:,0]
                            cam_t = pred_cam_t_full[n]
                            all_verts.append(verts)
                            all_cam_t.append(cam_t)
                            all_right.append(is_right)

                            # Save all meshes to disk
                            if args.save_mesh and False:
                                camera_translation = cam_t.copy()
                                tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                                tmesh.export(os.path.join(args.out_folder, f'{frame_num}_{person_id}.obj'))
                            
                    for i in range(frame_skip):
                        # Render front view
                        
                        if args.full_frame and len(all_verts) > 0:
                            misc_args = dict(
                                mesh_base_color=LIGHT_BLUE,
                                scene_bg_color=(1, 1, 1),
                                focal_length=scaled_focal_length,
                            )
                            #cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                            # Overlay image
                            #input_img = frames[i].astype(np.float32)[:,:,::-1]/255.0
                            #input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                            #input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                            #final_frame = 255*input_img_overlay[:, :, ::-1] #cv2.resize(255*input_img_overlay[:, :, ::-1], (512, 1024), 
                            #interpolation = cv2.INTER_LINEAR)
                            final_frame = frames[i]
                            final_frame = cv2.normalize(final_frame, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            video.write(final_frame)
                            #if frame_num == 300:
                            #    break
                    pbar.update(frame_skip)    
    except KeyboardInterrupt:

        print("stop at", frame_num)
        print("saving")
        cv2.destroyAllWindows()
        video.release()
    except Exception as error:
        print(error)
        print("error at", frame_num)
        print("saving")
        cv2.destroyAllWindows()
        video.release()

    
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    main()
