from collections import defaultdict, Counter
import os
import csv

# Input and output paths
base_input_dir = "D:/IEMOCAP_full_release/IEMOCAP_full_release"
output_file = "D:/IEMOCAP_full_release/scene_emotions_binary.csv"

# Extract the emotion label from a single line of the form: <utterance_id> :<emotion>; ()
def extract_emotion(line):
    try:
        _, emotion_part = line.split(":", 1)
        emotion = emotion_part.strip().split(";")[0].strip()
        return emotion if emotion else None
    except ValueError:
        return None

# Map to hold per-scene emotion counts
scene_emotions_map = defaultdict(Counter)

# Iterate through all 5 sessions
for session_num in range(1, 6):
    input_dir = os.path.join(base_input_dir, f"Session{session_num}", "dialog", "EmoEvaluation", "Categorical")
    
    if not os.path.exists(input_dir):
        print(f"Warning: Directory {input_dir} does not exist. Skipping.")
        continue

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt") and "_e" in filename:
            full_path = os.path.join(input_dir, filename)
            scene_id = filename.split("_e")[0]  # Extract scene ID (before _e)

            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    emotion = extract_emotion(line)
                    if emotion:
                        scene_emotions_map[scene_id][emotion] += 1

# Collect all unique emotions across all scenes
all_emotions = sorted({emotion for counter in scene_emotions_map.values() for emotion in counter})

# Write new binary labels to CSV
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Header
    writer.writerow(["scene_id"] + all_emotions)

    # Process each scene
    for scene_id, emotions_counter in scene_emotions_map.items():
        total = sum(emotions_counter.values())

        # Normalized probabilities
        norm = {e: emotions_counter.get(e, 0) / total for e in all_emotions}

        # Apply threshold: >0.5 → 1 else 0
        binary = [1 if norm[e] > 0.1 else 0 for e in all_emotions]

        writer.writerow([scene_id] + binary)

print(f"Binary scene-level emotion labels saved to {output_file}")