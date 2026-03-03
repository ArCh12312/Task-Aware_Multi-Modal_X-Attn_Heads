import os
from tqdm import tqdm

def delete_dot_underscore_files_with_progress(root_dir):
    # First collect all target files
    dot_underscore_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("._"):
                dot_underscore_files.append(os.path.join(dirpath, filename))

    # Show progress while deleting
    for file_path in tqdm(dot_underscore_files, desc="Deleting ._ files"):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Example usage:
delete_dot_underscore_files_with_progress("D:/IEMOCAP_full_release/IEMOCAP_full_release")
