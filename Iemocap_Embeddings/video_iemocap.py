import os
import cv2
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

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_rate == 0:
                frames.append(frame)

            frame_count += 1

        cap.release()
        return frames

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

    def extract_features(self, video_path):
        raw_frames = self.extract_frames(video_path)
        total_frames = len(raw_frames)

        if total_frames == 0:
            print(f"No frames from {video_path}")
            return np.zeros((512, 768), dtype=np.float32)

        # Uniformly select 8 frames
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            sampled_frames = [raw_frames[i] for i in indices]
        else:
            sampled_frames = raw_frames

        all_tokens = []

        for frame in sampled_frames:
            frame = self.transform(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model.forward_features(frame)
                tokens = outputs["x_norm_patchtokens"]

            tokens = tokens.squeeze(0)

            tokens = self.downsample_patches(tokens)  # (64,768)

            all_tokens.append(tokens.cpu().numpy())

        features = np.vstack(all_tokens)

        # Pad if fewer than 8 frames
        if features.shape[0] < 512:
            pad = np.zeros((512 - features.shape[0], 768), dtype=np.float32)
            features = np.vstack((features, pad))

        return features

    def process_videos(self):
        video_files = glob(os.path.join(self.data_path, "*.avi"))

        for video in video_files:
            name = os.path.basename(video).split('.')[0]
            print(f"Processing {name}...")

            features = self.extract_features(video)
            np.save(os.path.join(self.output_path, f"{name}.npy"), features)

        print("DINOv2 extraction complete!")

def main():
    base_data_path = "D:\IEMOCAP_full_release\IEMOCAP_full_release"
    base_output_path = "D:\IEMOCAP_full_release\Video_output_512_DINOv2"

    num_frames = 8          # 8 frames
    frame_rate = 5          # sample every 5th frame

    for session_num in range(1, 6):
        data_path = os.path.join(
            base_data_path,
            f"Session{session_num}",
            "dialog",
            "avi",
            "DivX"
        )

        output_path = os.path.join(
            base_output_path,
            f"Session{session_num}"
        )

        print(f"\nProcessing Session{session_num}...")
        print(f"Input path: {data_path}")
        print(f"Output path: {output_path}")

        extractor = DinoV2FeatureExtractor(
            data_path=data_path,
            output_path=output_path,
            frame_rate=frame_rate,
            num_frames=num_frames
        )

        extractor.process_videos()

    print("\nAll sessions processed successfully!")


if __name__ == "__main__":
    main()