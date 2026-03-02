import os
import numpy as np
import torch
import torchaudio
from glob import glob


class HuBERTFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained HuBERT Base
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(self.device)
        self.model.eval()

        self.sample_rate = bundle.sample_rate  # 16000

    def extract_features(self, audio_path, target_length=512):
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.sample_rate
            )

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            # Extract hidden states
            features, _ = self.model.extract_features(waveform)
            embeddings = features[-1]  # last transformer layer

        embeddings = embeddings.squeeze(0).cpu().numpy()
        num_frames = embeddings.shape[0]
        feature_dim = embeddings.shape[1]  # 768 for hubert_base

        if num_frames >= target_length:
            indices = np.linspace(0, num_frames - 1, target_length, dtype=int)
            sampled_embeddings = embeddings[indices]
            print(f"Sampled {target_length} features from {num_frames} frames.")
        else:
            padding = np.zeros((target_length - num_frames, feature_dim), dtype=np.float32)
            sampled_embeddings = np.vstack([embeddings, padding])
            print(f"Padded {num_frames} features to {target_length}.")

        return sampled_embeddings

def main():
    base_data_path = "D:\\IEMOCAP_full_release\\IEMOCAP_full_release"
    base_output_path = "D:\\IEMOCAP_full_release\\Audio_output_512_HuBERT"

    extractor = HuBERTFeatureExtractor()

    for session_num in range(1, 6):
        data_path = os.path.join(
            base_data_path, f"Session{session_num}", "dialog", "wav"
        )
        output_path = os.path.join(base_output_path, f"Session{session_num}")
        os.makedirs(output_path, exist_ok=True)

        audio_files = glob(os.path.join(data_path, "*.wav"))

        for i, audio in enumerate(audio_files):
            audio_name = os.path.basename(audio).split(".")[0]
            print(f"Processing {audio_name} in Session{session_num}...")

            features = extractor.extract_features(audio)

            if i == 0:
                print(f"Shape of features for {audio_name}: {features.shape}")

            np.save(os.path.join(output_path, f"{audio_name}.npy"), features)

    print("HuBERT feature extraction for all sessions complete!")


if __name__ == "__main__":
    main()