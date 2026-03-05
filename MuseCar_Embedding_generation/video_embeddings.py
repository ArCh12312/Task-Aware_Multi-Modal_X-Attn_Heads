import os
import cv2
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from glob import glob


class DinoV2FeatureExtractor:
    def __init__(self, data_path, output_path, frame_rate=5, num_frames=8):
        self.data_path = data_path
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.num_frames = num_frames
        os.makedirs(output_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14"
        )
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def downsample_patches(self, patch_tokens):

        # 256 tokens → 16x16 grid
        patch_tokens = patch_tokens.view(16, 16, 768)  # (16,16,768)

        # Move channels first for interpolation
        patch_tokens = patch_tokens.permute(2, 0, 1).unsqueeze(0)  # (1,768,16,16)

        # Bilinear downsample to 8x8
        patch_tokens = F.interpolate(
            patch_tokens,
            size=(8, 8),
            mode="bilinear",
            align_corners=False
        )

        # Back to (64, 768)
        patch_tokens = patch_tokens.squeeze(0).permute(1, 2, 0)  # (8,8,768)
        patch_tokens = patch_tokens.reshape(-1, 768)  # (64,768)

        return patch_tokens

    def extract_scene_features(self, cap, start_sec, end_sec):
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % self.frame_rate == 0:
                frames.append(frame)

            current_frame += 1

        total_frames = len(frames)

        if total_frames == 0:
            return np.zeros((512, 768), dtype=np.float32)

        # Uniform sampling
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        all_tokens = []

        for frame in sampled_frames:
            frame = self.transform(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model.forward_features(frame)
                tokens = outputs["x_norm_patchtokens"]

            tokens = tokens.squeeze(0)
            tokens = self.downsample_patches(tokens)

            all_tokens.append(tokens.cpu().numpy())

        features = np.vstack(all_tokens)

        # Pad to 512
        if features.shape[0] < 512:
            pad = np.zeros((512 - features.shape[0], 768), dtype=np.float32)
            features = np.vstack((features, pad))

        return features

    def process_videos(self, transcript_dir):
        video_files = glob(os.path.join(self.data_path, "*.mp4"))

        for i, video in enumerate(video_files):
            video_id = os.path.splitext(os.path.basename(video))[0]
            transcript_file = os.path.join(transcript_dir, f"{video_id}.csv")

            if not os.path.exists(transcript_file):
                print(f"No transcript for {video_id}, skipping.")
                continue

            print(f"\nProcessing video: {video_id}")

            cap = cv2.VideoCapture(video)

            with open(transcript_file, newline="") as f:
                reader = csv.DictReader(f)

                for scene_idx, row in enumerate(reader, start=1):
                    start = float(row["start"]) / 1000
                    end = float(row["end"]) / 1000

                    # print(f"  Scene {scene_idx}: {start:.2f}s → {end:.2f}s")

                    features = self.extract_scene_features(cap, start, end)

                    save_path = os.path.join(
                        self.output_path,
                        f"{video_id}_scene_{scene_idx}.npy"
                    )
                    np.save(save_path, features)

                    if i==10:
                        break

            cap.release()

        print("All videos processed.")

def main():
    data_path = "D:/c2_muse_sent/videos" 
    output_path = "D:\c2_muse_sent\Video_output_512_DINOv2"
    transcript_dir="C:/Users/aryan/Documents/Study/Research/MuseCar_Classification/c2_muse_sent/transcription_segments1"

    num_frames = 8          
    frame_rate = 5          

    print(f"Input path: {data_path}")
    print(f"Output path: {output_path}")

    extractor = DinoV2FeatureExtractor(
        data_path=data_path,
        output_path=output_path,
        frame_rate=frame_rate,
        num_frames=num_frames
    )

    extractor.process_videos(transcript_dir)

    print("\nAll videos processed successfully!")


if __name__ == "__main__":
    main()