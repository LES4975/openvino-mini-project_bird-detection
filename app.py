import argparse
from bird_detect.bird_fear import BirdDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Bird Detection using YOLO11')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="./yolo11s_openvino_model_int8/yolo11s_quant.xml",
        help='Path to the YOLO11 model file (default: ./yolo11s_openvino_model_int8/yolo11s_quant.xml)'
    )
    
    parser.add_argument(
        '--image_path', 
        type=str, 
        default='./bird_detect/bird_image/IE002877093_STD.jpg',
        help='Path to the input image file (default: ./bird_detect/bird_image/IE002877093_STD.jpg)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize detector with model path
    detector = BirdDetector(args.model_path)
    
    # Run detection on the image
    detector.detect_start(args.image_path)

if __name__ == "__main__":
    main()