import os

def extract_text_from_transcript(file_path):
    """Extract only spoken text from an IEMOCAP transcript file."""
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            spoken = line.split(":", 1)[1].strip()
            lines.append(spoken)
    return "\n".join(lines)


def process_all_sessions(base_data_path, output_base_path):
    os.makedirs(output_base_path, exist_ok=True)

    for session_num in range(1, 6):
        print(f"Processing Session {session_num}...")

        transcript_dir = os.path.join(
            base_data_path,
            f"Session{session_num}",
            "dialog",
            "transcriptions"
        )

        output_dir = os.path.join(output_base_path, f"Session{session_num}")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(transcript_dir):
            print(f"Transcript directory missing: {transcript_dir}")
            continue

        for filename in os.listdir(transcript_dir):
            if not filename.endswith(".txt"):
                continue

            input_path = os.path.join(transcript_dir, filename)
            scene_id = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, scene_id + ".txt")

            clean_text = extract_text_from_transcript(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            print(f"✔ Saved cleaned text: {output_path}")

    print("Finished cleaning all sessions!")


# === Run ===
if __name__ == "__main__":
    base_data_path = "D:\IEMOCAP_full_release\IEMOCAP_full_release"
    output_base_path = "D:\IEMOCAP_full_release\scene_transcripts_clean"

    process_all_sessions(base_data_path, output_base_path)
