# yolov11_pipeline/main.py


import os
import requests
from pathlib import Path

from yolov11_pipeline.core import YOLOv11Pipeline

def download_file(url, filename, directory):
    """파일이 존재하지 않으면 URL에서 다운로드"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename

    if not file_path.exists():
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"{filename} already exists.")
    return file_path
if __name__ == "__main__":
    pipeline = YOLOv11Pipeline("yolo11s")
    pipeline.run()