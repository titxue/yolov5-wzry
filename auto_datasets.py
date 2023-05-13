import os
import cv2
import torch
import argparse


def load_model_and_predict_video(args):
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model_path)
    model.eval()

    # Open the video file
    video = cv2.VideoCapture(args.video_path)

    # Create output directories if they don't exist
    os.makedirs(f"{args.output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{args.output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{args.output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{args.output_dir}/labels/val", exist_ok=True)

    frame_idx = 0
    save_idx = 0
    while video.isOpened():
        # Read frame from video
        ret, frame = video.read()

        # If the frame was not read correctly, we break the loop
        if not ret:
            break

        # Save and process every 30th frame
        if frame_idx % 30 == 0:
            # Resize frame to the closest multiple of 32
            height, width, _ = frame.shape
            new_height = (height // 32) * 32
            new_width = (width // 32) * 32
            frame = cv2.resize(frame, (new_width, new_height))

            # Convert the image to a tensor, add a batch dimension and normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

            # Perform inference
            results = model(frame_tensor)

            # Determine subset
            subset = 'val' if save_idx % 10 == 0 else 'train'

            # Save frame as image
            img_path = f"{args.output_dir}/images/{subset}/frame{save_idx}.png"
            cv2.imwrite(img_path, frame)

            # Save labels
            img_data = results.xyxy[0]
            with open(f"{args.output_dir}/labels/{subset}/frame{save_idx}.txt", 'w') as f:
                for *box, confidence, class_idx in img_data:
                    # only save the label if confidence > 0.9
                    if confidence > 0.9:
                        x_center = ((box[0] + box[2]) / 2) / width
                        y_center = ((box[1] + box[3]) / 2) / height
                        box_width = (box[2] - box[0]) / width
                        box_height = (box[3] - box[1]) / height
                        f.write(f"{int(class_idx)} {x_center} {y_center} {box_width} {box_height}\n")

            save_idx += 1

        frame_idx += 1

    # Release the video file
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='Z:\\code\\20230504_124619.mp4', help='Path to video file')
    parser.add_argument('--output_dir', type=str, default='runs_dataset', help='Path to output directory')
    parser.add_argument('--model_path', type=str, default='runs/train/exp7/weights/best.pt',
                        help='Path to trained model')
    args = parser.parse_args()
    load_model_and_predict_video(args)