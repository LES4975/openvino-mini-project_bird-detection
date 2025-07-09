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


def quantize_model(model_name, det_model_path, det_model, label_map):
    int8_model_path = Path(f"{model_name}_openvino_model_int8/{model_name}.xml")
    if int8_model_path.exists():
        return int8_model_path

    # Prepare dataset
    DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
    LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"
    CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

    OUT_DIR = DATASETS_DIR
    DATA_PATH = OUT_DIR / "val2017.zip"
    LABELS_PATH = OUT_DIR / " "
    CFG_PATH = OUT_DIR / ""

    if not (OUT_DIR / "coco/labels").exists():
        download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
        download_file(LABELS_URL, LABELS_PATH.name, LABELS_PATH.parent)
        download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)
        with ZipFile(LABELS_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)
        with ZipFile(DATA_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR / "coco/images")

    # Setup validator
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

    quant_dataset = nncf.Dataset(dataloader, transform_fn)
    core = ov.Core()
    det_ov_model = core.read_model(det_model_path)

    ignored_scope = nncf.IgnoredScope(
        subgraphs=[
            nncf.Subgraph(
                inputs=[
                    f"__module.model.23/aten::cat/Concat",
                    f"__module.model.23/aten::cat/Concat_1",
                    f"__module.model.23/aten::cat/Concat_2"
                ],
                outputs=[f"__module.model.23/aten::cat/Concat_7"]
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
    shutil.copy(det_model_path.parent / "metadata.yaml", int8_model_path.parent / "metadata.yaml")

    return int8_model_path