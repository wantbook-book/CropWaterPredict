import os
from datetime import datetime
from pathlib import Path
def timestamp_filename(filepath: Path)->Path:
    # Get the last modified time of the file
    mtime = os.path.getmtime(filepath)
    
    # Convert timestamp to a human-readable format, suitable for filenames
    # You can adjust the format as per your requirements
    timestamp_str = datetime.fromtimestamp(mtime).strftime("%Y%m%d%H%M")
    
    # Create a new filename with the timestamp
    # Assuming the file has an extension, which we keep the same
    dirname = filepath.parent
    extension = filepath.suffix
    new_filename = dirname / f"{timestamp_str}{extension}"
    
    return new_filename

def use_mt_as_filename(files_dir: Path):
    for file in files_dir.iterdir():
        if file.is_file():
            new_file = timestamp_filename(file)
            os.rename(file, new_file)

if __name__ == '__main__':
    files_dirs = [
        files_dir for files_dir in Path('../data/sapflow_predict_data/sapflow_images').iterdir() if files_dir.is_dir()
    ]
    for files_dir in files_dirs:
        use_mt_as_filename(files_dir)

