# yolov11_pipeline/core.py

from yolov11_pipeline.data_utils import download_utils, download_sample_image
from yolov11_pipeline.model_utils import load_model, export_openvino_model
from yolov11_pipeline.quantization import quantize_model

class YOLOv11Pipeline:
    def __init__(self, model_name="yolo11m"):
        self.model_name = model_name
        self.image_path = None
        self.det_model = None
        self.label_map = None
        self.det_model_path = None
        self.int8_model_path = None

    def run(self):
        # 1. 준비
        download_utils()
        self.image_path = download_sample_image()

        # 2. 모델 로딩
        self.det_model, self.label_map = load_model(self.model_name)

        # 3. OpenVINO로 변환
        print("Exporting model to OpenVINO IR...")
        self.det_model_path = export_openvino_model(self.det_model, self.model_name)

        # 4. 양자화
        print("Quantizing model...")
        self.int8_model_path = quantize_model(
            self.model_name,
            self.det_model_path,
            self.det_model,
            self.label_map
        )


        print("Pipeline complete.")
