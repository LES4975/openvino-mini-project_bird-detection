# yolov11_pipeline/quantization.py

import shutil
from pathlib import Path
from zipfile import ZipFile

import openvino as ov
import nncf

from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset, DATASETS_DIR

from notebook_utils import download_file


def quantize_model(model_name, det_model_path, det_model, label_map, output_dir=None):
    # 1. 사용자 지정 output_dir이 없으면 기본 경로로 설정
    if output_dir is None:
        output_dir = Path(f"{model_name}_openvino_model_int8")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    int8_model_path = output_dir / f"{model_name}.xml"

    # 2. 기존 모델 존재 시 반환
    if int8_model_path.exists():
        return int8_model_path

    # 3. Dataset 경로 정의 및 다운로드
    DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
    LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
    CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

    OUT_DIR = DATASETS_DIR
    DATA_PATH = OUT_DIR / "val2017.zip"
    LABELS_PATH = OUT_DIR / "coco2017labels-segments.zip"
    CFG_PATH = OUT_DIR / "coco.yaml"

    if not (OUT_DIR / "coco/labels").exists():
        download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
        download_file(LABELS_URL, LABELS_PATH.name, LABELS_PATH.parent)
        download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)

        with ZipFile(LABELS_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)
        with ZipFile(DATA_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR / "coco/images")

    # 4. Validator 구성
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(CFG_PATH)
    det_validator = det_model.task_map[det_model.task]["validator"](args=args)
    det_validator.data = check_det_dataset(args.data)
    det_validator.stride = 32
    det_validator.is_coco = True
    det_validator.class_map = coco80_to_coco91_class()
    det_validator.names = label_map
    det_validator.metrics.names = det_validator.names
    det_validator.nc = 80
    dataloader = det_validator.get_dataloader(OUT_DIR / "coco", 1)

    def transform_fn(data_item: dict):
        return det_validator.preprocess(data_item)['img'].numpy()

    # 5. 양자화 수행
    quant_dataset = nncf.Dataset(dataloader, transform_fn)
    core = ov.Core()
    det_ov_model = core.read_model(det_model_path)

    ignored_scope = nncf.IgnoredScope(
        subgraphs=[
            nncf.Subgraph(
                inputs=[
                    "__module.model.23/aten::cat/Concat",
                    "__module.model.23/aten::cat/Concat_1",
                    "__module.model.23/aten::cat/Concat_2"
                ],
                outputs=["__module.model.23/aten::cat/Concat_7"]
            )
        ]
    )

    quant_model = nncf.quantize(
        det_ov_model,
        quant_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=ignored_scope
    )

    ov.save_model(quant_model, str(int8_model_path))
    
    metadata_src = det_model_path.parent / "metadata.yaml"
    if metadata_src.exists():
        shutil.copy(metadata_src, output_dir / "metadata.yaml")

    return int8_model_path
