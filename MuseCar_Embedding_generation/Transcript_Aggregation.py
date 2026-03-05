import os
import pandas as pd
from glob import glob

transcript_directory = "c2_muse_sent/transcription_segments"
no_of_videos = 303

for i in range(no_of_videos):
    subdirectory = os.path.join(transcript_directory, str(i))
    csv_files = sorted(glob(os.path.join(subdirectory, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {subdirectory}")
        continue

    # Create an empty DataFrame for aggregated rows
    full_df = pd.DataFrame(columns=["start", "end", "transcript"])

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            start_time = df["start"].min()
            end_time = df["end"].max()
            transcript = " ".join(df["word"].astype(str).tolist())

            # Append row to full_df
            full_df = pd.concat([
                full_df,
                pd.DataFrame([{
                    "start": start_time,
                    "end": end_time,
                    "transcript": transcript
                }])
            ], ignore_index=True)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # Save aggregated results for this subdirectory
    output_file = os.path.join(transcript_directory, f"{i}.csv")
    full_df.to_csv(output_file, index=False)
    print(f"Saved aggregated transcript to {output_file}")