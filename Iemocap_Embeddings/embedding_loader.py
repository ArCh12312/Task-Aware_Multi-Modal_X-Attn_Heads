import os
import numpy as np

def load_all_embeddings(base_embedding_dir):
    embeddings = []
    scene_names = []

    # Only loop over Session1 ... Session5
    for session_num in range(1, 6):
        session_dir = os.path.join(base_embedding_dir, f"Session{session_num}")

        if not os.path.exists(session_dir):
            print(f"Warning: {session_dir} does not exist, skipping.")
            continue

        for file_name in sorted(os.listdir(session_dir)):
            if file_name.endswith(".npy"):
                full_path = os.path.join(session_dir, file_name)

                emb = np.load(full_path)
                embeddings.append(emb)
                scene_names.append(os.path.splitext(file_name)[0])  # remove .npy extension

    return embeddings


# === Example usage ===
if __name__ == "__main__":
    base_embedding_dir = "Audio_output_128"
    scene_names, embeddings = load_all_embeddings(base_embedding_dir)

    print(f"Loaded {len(embeddings)} embeddings.")
    print(f"Example scene: {scene_names[0]}")
    print(f"Example embedding shape: {embeddings[0].shape}")