import os
import re
import csv
import numpy as np

def load_embeddings(base_path):
    embeddings = {}

    patterns = [
        r"^(\d+)_scene_(\d+)\.npy$",
        r"^(\d+)_scene(\d+)\.npy$",
        r"^(\d+)_(\d+)\.npy$"
    ]

    compiled_patterns = [re.compile(p) for p in patterns]

    for file in os.listdir(base_path):

        if not file.endswith(".npy"):
            continue

        for pattern in compiled_patterns:
            match = pattern.match(file)
            if match:
                video_id = int(match.group(1))
                segment_id = int(match.group(2))

                full_path = os.path.join(base_path, file)
                embeddings[(video_id, segment_id)] = np.load(full_path)

                break

    return embeddings

def load_labels(label_csv_path):
    labels = {}

    with open(label_csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video_id = row["id"]
            segment_id = row["segment_id"]

            labels[(video_id, segment_id)] = {
                "arousal": int(row["label_arousal"]),
                "valence": int(row["label_valence"]),
                "topic": int(row["label_topic"]),
            }

    return labels