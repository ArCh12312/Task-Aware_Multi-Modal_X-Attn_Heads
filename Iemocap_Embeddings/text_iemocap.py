import os
import numpy as np
import torch
from glob import glob
from transformers import BertTokenizer, BertModel

class BERTTextFeatureExtractor:
    def __init__(self, model_name="bert-base-uncased", output_dim=768, target_length=128):
        """
        - Truncate transcripts to 512 tokens (BERT max).
        - Extract all token embeddings (seq_len ≤ 512).
        - Downsample or pad to (128, 768).
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.output_dim = output_dim
        self.target_length = target_length

    def extract_features(self, text):
        if text.strip() == "":
            print("Warning: Empty transcript. Returning zeros.")
            return np.zeros((512, self.output_dim), dtype=np.float32)

        # Force exactly 512 tokens
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, 512, 768)

        hidden_states = hidden_states.squeeze(0).cpu().numpy()  # (512, 768)

        return hidden_states.astype(np.float32)
    
def main():
    base_data_path = "D:\IEMOCAP_full_release\scene_transcripts_clean"
    base_output_path = "D:/IEMOCAP_full_release/Text_Embeddings"

    extractor = BERTTextFeatureExtractor()

    for session_num in range(1, 6):
        transcript_dir = os.path.join(base_data_path, f"Session{session_num}")
        output_dir = os.path.join(base_output_path, f"Session{session_num}")
        os.makedirs(output_dir, exist_ok=True)

        transcript_files = glob(os.path.join(transcript_dir, "*.txt"))

        for i, txt_file in enumerate(transcript_files):
            file_name = os.path.basename(txt_file).split('.')[0]
            print(f"Processing {file_name} in Session{session_num}...")

            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

            features = extractor.extract_features(text)

            if i == 0:
                print(f"Example embedding shape: {features.shape}")  # should always be (128, 768)

            np.save(os.path.join(output_dir, f"{file_name}.npy"), features)

    print("Text feature extraction complete")


if __name__ == "__main__":
    main()
