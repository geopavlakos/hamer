from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

# color for hand mesh
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ThreadPoolExecutor

# MediaPipe model
model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# ---- Create detector ----
landmarker = HandLandmarker.create_from_options(options)

def get_bounding_boxes(cv2_image):
    """
    Detect bounding boxes and handedness using MediaPipe.
    :param cv2_image: the image with hands
    :returns (bboxes, right) two numpy arrays with shape (N, 4) and (N,) where N is count of hands detected.
    """
    image = cv2_image
    bboxes = []
    right = []
    # ---- Create detector ----
    with HandLandmarker.create_from_options(options) as landmarker:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp = int(round(time.time()*1000))
        # Run detection
        result = landmarker.detect_for_video(mp_image, timestamp)

        h, w, _ = image.shape

        # ---- Process results ----
        for hand_landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness
        ):

            # Bounding box from landmarks
            xs = [lm.x * w for lm in hand_landmarks]
            ys = [lm.y * h for lm in hand_landmarks]

            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            label = handedness[0].category_name  # "Left" or "Right"
            is_right = handedness[0].category_name == "Right"
            
            bboxes.append([x_min, y_min, x_max, y_max])
            right.append(1 if is_right else 0)
    return np.array(bboxes), np.array(right)


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code for webcam using MediaPipe for bounding boxes')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video_device', type=str, default="0", help='Video device string for OpenCV.')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--save_crop', dest='save_crop', action='store_true', default=False, help='If set, saves cropped image with hand')

    args = parser.parse_args()


    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # worker thread for hand detection and image annotation.
    executor = ThreadPoolExecutor(max_workers=1)
    future = None   # keeps track of running task

    # Open the default camera
    cam = cv2.VideoCapture(int(args.video_device) if args.video_device.isnumeric() else args.video_device)
    img_fn = "debug_image"
    
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    def detect_hands(img_cv2):
        result_img = img_cv2
        boxes, right = get_bounding_boxes(img_cv2)
        
        if len(right) > 0: # check if any hands detected.
            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            
            for batch in dataloader:
                batch = recursive_to(batch, device)
                start = time.time()
                with torch.no_grad():
                    out = model(batch)
                end = time.time()
                print(f"Inference on hamer took {end-start}s")

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
                    person_id = int(batch['personid'][n])
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    if args.save_crop:
                        regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                batch['img'][n],
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                )

                        final_img = np.concatenate([input_patch, regression_img], axis=1)

                        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

            # Render front view
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                result_img = (255*input_img_overlay[:, :, ::-1]).astype(np.uint8)
                
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), result_img)

        return result_img
        
        
    while True:
        ret, img_cv2 = cam.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Display the result img if available.
        if future is not None and future.done():
            result_img = future.result()
            cv2.imshow('Camera', result_img)

        # Schedule hand detection on the current frame if it is not running.
        if future is None or future.done():
            future = executor.submit(detect_hands, img_cv2.copy())

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
