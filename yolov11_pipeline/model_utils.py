# yolov11_pipeline/model_utils.py

from pathlib import Path
from ultralytics import YOLO

def load_model(model_name):
    det_model = YOLO(f"{model_name}.pt")
    det_model.to("cpu")
    label_map = det_model.model.names
    return det_model, label_map


def export_openvino_model(det_model, model_name):
    export_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
    if not export_path.exists():
        det_model.export(format="openvino", dynamic=True, half=True)
    return export_path