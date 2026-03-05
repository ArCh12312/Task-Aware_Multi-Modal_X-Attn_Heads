import os
import csv
import numpy as np
import torch
from glob import glob
from transformers import BertTokenizer, BertModel

class BERTTextFeatureExtractor:
    def __init__(self, model_name="bert-base-uncased", output_dim=768, target_length=128):

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
    input_dir = "C:/Users/aryan/Documents/Study/Research/MuseCar_Classification/c2_muse_sent/transcription_segments1"
    output_dir = "D:\c2_muse_sent\Text_Embeddings_512"
    os.makedirs(output_dir, exist_ok=True)

    extractor = BERTTextFeatureExtractor()

    transcript_files = sorted(glob(os.path.join(input_dir, "*.csv")))
    if not transcript_files:
        print(f"No CSV files found in {input_dir}.")
        return

    for i, transcript_file in enumerate(transcript_files):
        file_name = os.path.basename(transcript_file).split('.')[0]
        print(f"\n=== Processing video {file_name} ===")

        with open(transcript_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader, start=1):
                text = row["transcript"].strip()
                if not text:
                    print(f"Skipping empty row in video {file_name}, scene {idx}")
                    continue

                features = extractor.extract_features(text)

                output_path = os.path.join(output_dir, f"{file_name}_scene{idx}.npy")
                np.save(output_path, features)
                print(f"Saved {output_path} (shape={features.shape})")

    print("\nBERT text feature extraction complete.")


if __name__ == "__main__":
    main()
