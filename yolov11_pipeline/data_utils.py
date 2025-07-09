# yolov11_pipeline/data_utils.py

from pathlib import Path
import requests

def download_utils():
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
        )
        open("notebook_utils.py", "w").write(r.text)


def download_sample_image():
    from notebook_utils import download_file
    IMAGE_PATH = Path("./data/coco_bike.jpg")

    if not IMAGE_PATH.exists():
        download_file(
            url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg",
            filename=IMAGE_PATH.name,
            directory=IMAGE_PATH.parent,
        )
    return IMAGE_PATH
